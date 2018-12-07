import sys
import numpy as np

from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import networks
from pytorch_patternNet import PatternNetSignalEstimator

import keras
import keras.backend
import keras.models
import innvestigate

# in this file we verify whether our PyTorch implementation
# outputs exactly the same values as the original iNNvestigate keras implementation
# given the same data and network weights
# note that sometimes we must reshape or transpose in order to match keras convention with pytorch convention

# tolerance for assert statements
rtol=1e-3
atol=1e-3

# initialise keras mininet
input_shape = (4, 4, 1)
keras_mininet = keras.models.Sequential([
    keras.layers.Conv2D(3, (3, 3), use_bias=False, activation="relu", padding='valid', input_shape=input_shape),
    keras.layers.MaxPooling2D((2, 2), (2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(2, use_bias=False, activation="linear"),
])

# initialise pytorch mininet
pytorch_mininet = networks.MiniNet() # pytorch models don't end with a softmax layer, because the nn.CrossEntropyLoss() function computes it for us

# randomly determine weights both networks should have
f1 = np.random.rand(3, 3)
f2 = np.random.rand(3, 3)
f3 = np.random.rand(3, 3)
dw = np.random.rand(3, 2)

# assign these weights to keras model according to keras convention
f1_k = f1[..., None, None]
f2_k = f2[..., None, None]
f3_k = f3[..., None, None]
f_k = np.concatenate((f1_k, f2_k, f3_k), 3)
keras_mininet.layers[0].set_weights([f_k])
keras_mininet.layers[3].set_weights([dw])

# assing the weights to the pytorch model according to pytorch convention
f1_p = torch.FloatTensor([f1]).view(1, 1, 3, 3)
f2_p = torch.FloatTensor([f2]).view(1, 1, 3, 3)
f3_p = torch.FloatTensor([f3]).view(1, 1, 3, 3)
f_p = torch.cat((f1_p, f2_p, f3_p), 0)
dw_p = torch.FloatTensor(dw).t() # pytorch convention is the transpose of keras convention
pytorch_mininet.conv.weight.data.copy_(f_p)
pytorch_mininet.dense.weight.data.copy_(dw_p)

# synthesize fake data
data_1 = np.random.rand(4, 4)
data_2 = np.random.rand(4, 4)

# prepare data for keras
data_1_k = data_1[None, :, :, None]
data_2_k = data_2[None, :, :, None]
data_k = np.concatenate((data_1_k, data_2_k), 0)

# prepare data for pytorch
data_1_p = torch.FloatTensor(data_1)[None, None, ...]
data_2_p = torch.FloatTensor(data_2)[None, None, ...]
data_p = torch.cat((data_1_p, data_2_p), 0)

# assert that both models have the same output
k_outp = keras_mininet.predict(data_k)
p_outp = pytorch_mininet(data_p).detach().numpy()
assert np.allclose(k_outp, p_outp, rtol=rtol, atol=atol), "The Keras and PyTorch models (not signal estimators) process items differently. Bad network initialisation"

# get the patterns with original keras implementation
signal_estimator_k = innvestigate.create_analyzer("pattern.net", keras_mininet, **{"pattern_type": "relu"})
signal_estimator_k.fit(data_k, batch_size=2, verbose=1)

# get the patterns with pytorch implementation
signal_estimator_p = PatternNetSignalEstimator(pytorch_mininet)
signal_estimator_p.update_E(data_p)
signal_estimator_p.get_patterns()

# assert that the same patterns have been learned
conv_pattern_compare_k = signal_estimator_k._patterns[0]
conv_pattern_compare_p = signal_estimator_p.net.conv.a.permute(2, 3, 1, 0).detach().numpy()
assert np.allclose(conv_pattern_compare_k, conv_pattern_compare_p, rtol=rtol, atol=atol), "The PyTorch implementation learns different patterns than the original Keras implementation"

conv_pattern_compare_k = signal_estimator_k._patterns[1]
conv_pattern_compare_p = signal_estimator_p.net.dense.a.t().detach().numpy()
assert np.allclose(conv_pattern_compare_k, conv_pattern_compare_p, rtol=rtol, atol=atol), "The PyTorch implementation learns different patterns than the original Keras implementation"

# assert that both implementations detect the same signal
signal_estimate_k = signal_estimator_k.analyze(data_k)
signal_estimate_p = signal_estimator_p.get_signal(data_p).permute(0, 2, 3, 1).detach().numpy()
assert np.allclose(signal_estimate_k, signal_estimate_p, atol=atol), "The PyTorch implementation estimates different signals than the original Keras implementation"
