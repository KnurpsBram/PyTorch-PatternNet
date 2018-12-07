import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import networks

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def isnan(x):
    return x != x

def breadcast(a, b):
    # prepare tensor a to be broadcasted with tensor b by adding empty dimensions at the end
    for i in range(len(b.size()) - len(a.size())):
        a = a[..., None]
    return a

def outer_prod_along_batch(a, b):
    # outer product of two vectors where 0th dimension is batch
    # a is (n_batch, n)
    # b is (n_batch, m)
    # out must be (n_batch, n, m)
    return a[:, :, None] @ b[:, None, :]

def normalise(x):
    denom = torch.max(abs(x).view(x.size(0), -1), dim=1, keepdim=True)[0]
    return x / breadcast(denom, x)

class PatternNetSignalEstimator(object):
    def __init__(self, net, pkl_path=False):
        self.net = net
        self.prepare_for_correlation_stats()
        if pkl_path:
            self.load_from_pkl(pkl_path)

    def prepare_for_correlation_stats(self):
        def make_info_collector(layer):
            def modify_forward(forward_fn):
                def forward_info_collector(x):
                    # save the input and the output of the operation the layer performs
                    # so that we can use it later
                    layer.inp = x
                    x = forward_fn(x)
                    layer.outp = x
                    return x
                return forward_info_collector

            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                # initialise running means (Expected value of x, y and xy)
                layer.Ex = torch.zeros_like(layer.weight).to(layer.weight.device)
                layer.Ey = torch.zeros(layer.weight.size(0)).to(layer.weight.device) # init as floatTensor sometimes inits it with NaN values. wtf is that about
                layer.Exy = torch.zeros_like(layer.weight).to(layer.weight.device)
                layer.n = torch.zeros(layer.weight.size(0)).to(layer.weight.device)
                layer.n_all = torch.zeros(layer.weight.size(0)).to(layer.weight.device)

                # modify forward function so as to save input-output pairs
                layer.forward = modify_forward(layer.forward)
            return layer

        for layer in list(self.net.children()):
            layer = make_info_collector(layer)

    def load_from_pkl(self, pkl_path):
        print("apples!")
        self.net.load_state_dict(torch.load(pkl_path))

        for layer in list(self.net.children()):
            if isinstance(layer, nn.Conv2d):
                print(layer.a)

    def smart_vectorise_for_conv(self, inp, outp, kw, kh, stride=1):
        # for
        # output of size [ n_batch, n_channel_out, fm_out_height, fm_out_width]
        # input of size  [ n_batch, n_channel_in, fm_in_height, fm_out_width]
        #
        # m_out = fm_out_height * fm_out_width
        # kw is kernel width
        # kh is kernel height
        #
        # try to reshape as
        # outputs of shape [n_batch, n_channel_out, m_out ]
        # input of shape [ n_batch, 1, m_out, n_channel_in, kh, kw ]
        #
        # in such configuration, each item in reshaped output
        # has a tensor of size [n_channel_in, kw, kh] of inputs that contributed to it (it's the kernel shape)
        # because these 'input volumes' are the same for each out channel, we need the empty dimension to broadcast
        #
        # example:
        #      inp               outp
        # [[[1, 2, 3],
        #   [4, 5, 6],   -->  [[10, 11],
        #   [7, 8, 9]]]        [12, 13]]
        #
        #   resh_inp             resh_outp
        # [[ [[1, 2],          [[10, 11, 12, 13]]
        #     [3, 4]],
        #
        #    [[2, 3],
        #     [5, 6]],
        #
        #    [[4, 5],
        #     [7, 8]],
        #
        #    [[5, 6],
        #     [7, 8]] ]]

        # reshape output
        n_batch, n_channel_out, fm_out_h, fm_out_w = outp.size()
        resh_outp = outp.view(n_batch, n_channel_out, -1, 1, 1, 1)

        # reshape input
        #
        # as an intermediate step, we first reshape to
        #                   [[ [1, 2,  2, 3],
        # [[[1, 2, 3],         [4, 5,  5, 6],
        #   [4, 5, 6],   -->   [4, 5,  5, 6],
        #   [7, 8, 9]]]        [7, 8,  8, 9], ]]
        n_batch, n_channel_in, w_in, h_in = inp.size()
        resh_inp = inp.unfold(3, kw, stride).contiguous().view(n_batch, n_channel_in, w_in, -1)
        new_h = resh_inp.size(-1)
        resh_inp = resh_inp.permute(0, 1, 3, 2)
        resh_inp = resh_inp.unfold(3, kh, stride).contiguous().view(n_batch, n_channel_in, -1, new_h)
        resh_inp = resh_inp.permute(0, 1, 3, 2)

        # [ n_batch, n_channel_in, kh*fm_out_h, kw*fm_out_w] --> [ n_batch, 1, 1, n_channel_in, kh*fm_out_h, kw*fm_out_w]
        # we will fill up 2nd dimension up with m_out items (indexing starts at 0)
        resh_inp = resh_inp[:, None, None, :, :, :]

        slices = resh_inp.chunk(fm_out_w, dim=5)
        resh_inp = torch.cat(slices, dim=2)
        patches = resh_inp.chunk(fm_out_w, dim=4)
        resh_inp = torch.cat(patches, dim=2)

        # slices = resh_inp.chunk(fm_out_h, dim=4)
        # resh_inp = torch.cat(slices, dim=2)
        # patches = resh_inp.chunk(fm_out_w, dim=5)
        # resh_inp = torch.cat(patches, dim=2)
        return resh_inp, resh_outp

    def update_E_for_layer(self, layer, ex, ey, exy, n, n_all):
        # for the E-values of a particular layer
        # adds this batch's contribution
        # to the running mean
        layer.Ex += ex
        layer.Ey += ey
        layer.Exy += exy
        layer.n += n
        layer.n_all += n_all

    def update_E(self, inp):
        # processes an item
        # for each layer in the network, keep track of in and output that each filter performs
        # use that info to update the expected values needed to compute attribution of that layer
        y = self.net(inp)
        for layer in list(self.net.children()):
            if isinstance(layer, nn.Linear):
                # count items that have positive output
                mask = (layer.outp > 0).float()

                # the inputs that contributed to positive outputs
                inp_p = outer_prod_along_batch(mask, layer.inp)
                # the outputs that were positive
                outp_p = layer.outp*mask
                # xy-pairs of positive y's
                xy_p = inp_p * outp_p[:, :, None]

                # this batch' contribution to the running mean
                ex = inp_p.sum(dim=0)
                ey = layer.outp.sum(dim=0)
                exy = xy_p.sum(dim=0)
                n = mask.sum(dim=0)
                n_all = torch.ones(outp_p.size(1)).to(outp_p.device) * outp_p.size(0)

                self.update_E_for_layer(layer, ex, ey, exy, n, n_all)

            if isinstance(layer, nn.Conv2d):
                kw, kh = layer.kernel_size
                resh_inp, resh_outp = self.smart_vectorise_for_conv(layer.inp, layer.outp, kw, kh)
                # smart vectorise should return tensors such that for each output element
                # there is a related input patch of size weight that produced the output element
                # we check if it works by reproducing the output
                # i = np.random.randint(resh_inp.size(2))
                # assert torch.allclose(torch.sum(resh_inp[0, 0, i, :, :, :] * layer.weight[0]) + layer.bias[0], resh_outp[0, 0, i], rtol=1e-4), "smth wrong in smart_vectorise_for_conv"

                mask = (resh_outp > 0).float()
                resh_outp_p = resh_outp * mask
                resh_inp_p = resh_inp * mask
                xy = resh_inp_p * resh_outp_p

                # b = 0 if mask[0, 0, i] == 0 else layer.bias[0]
                # assert torch.allclose(F.relu(torch.sum(resh_inp[0, 0, i, :, :, :] * layer.weight[0]) + b), resh_outp[0, 0, i], rtol=1e-4), "smth wrong with mask in update_E for conv"

                # note that mean and sum get rid of the dimension they operate over
                # shape [1, 2, 3].mean(dim=0).mean(dim=1) becomes [2]
                ex = resh_inp_p.sum(dim=0).sum(dim=1)
                # ey = resh_outp.sum(dim=0).sum(dim=1).squeeze()
                ey = resh_outp.sum(dim=0).sum(dim=1).squeeze()
                exy = xy.sum(dim=0).sum(dim=1)
                n = mask.sum(dim=0).sum(dim=1).squeeze()
                n_all = torch.ones(mask.size(1)).to(mask.device) * mask.size(0)*mask.size(2)

                self.update_E_for_layer(layer, ex, ey, exy, n, n_all)

    def get_patterns(self):
        # calculates the patterns a for each of the neurons in the network
        # and replaces the weights with the patterns
        # signal estimation can now be performed with a simple backward pass
        # only call this when E-values are based on a full epoch
        for layer in list(self.net.children()):
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                Ex = layer.Ex / breadcast(layer.n, layer.Ex)
                Ey = layer.Ey / breadcast(layer.n_all, layer.Ey)
                Exy = layer.Exy / breadcast(layer.n, layer.Exy)
                Ex[isnan(Ex)] = 0
                Ey[isnan(Ey)] = 0
                Exy[isnan(Exy)] = 0
                ExEy = Ex * breadcast(Ey, Ex)

                num = Exy - ExEy
                denom = (layer.weight*Exy).view((num.size(0), -1)).sum(dim=1) - (layer.weight*ExEy).view((num.size(0), -1)).sum(dim=1)
                a = num/breadcast(denom, num)
                a[isnan(a)] = 0

                layer.a = nn.Parameter(a)
                layer.weight_store = layer.weight

        # log various stuff to a history txt file?

    def assign_patterns_for_signal(self):
        for layer in list(self.net.children()):
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                layer.weight_store.data.copy_(layer.weight)
                layer.weight.data.copy_(layer.a)

    def assign_patterns_for_attribution(self):
        for layer in list(self.net.children()):
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                layer.weight_store.data.copy_(layer.weight)
                layer.weight.data.copy_(layer.a * layer.weight)

    def restore_weights(self):
        for layer in list(self.net.children()):
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                layer.weight.data.copy_(layer.weight_store)

    def forward_backward(self, inp, replace_weight_fn, c=None):
        # ask the network what evidence it has for predicting class c (n_batch integers)
        # if no c is given, pick the most probable class
        self.restore_weights()
        self.net.zero_grad()

        inp = Variable(inp, requires_grad=True)
        outp = self.net(inp)

        # gather the gradient of the class to show
        if c is None:
            c = torch.argmax(outp, dim=1)
        one_hot_output = torch.zeros_like(outp).to(outp.device)
        one_hot_output[:, c] = 1

        replace_weight_fn()
        outp.backward(gradient=one_hot_output)
        signal = inp.grad

        # normalise per item in batch
        return normalise(signal)

    def get_signal(self, inp, c=None):
        return self.forward_backward(inp, self.assign_patterns_for_signal, c=c)

    def get_attribution(self, inp, c=None):
        at = self.forward_backward(inp, self.assign_patterns_for_attribution, c=c)
        heatmap = at.mean(dim=1, keepdim=True)
        norm_fact = heatmap.max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0]
        heatmap = heatmap/norm_fact # normalise
        return heatmap
