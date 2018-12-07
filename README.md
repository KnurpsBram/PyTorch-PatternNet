# PatternNet for PyTorch

Image           |  Signal
:-------------------------:|:-------------------------:
![](images/mnist.png?raw=true "Image")  |  ![](images/mnist_signal.png?raw=true "Signal")

This is a PyTorch implementation of the PatternNet signal estimator / neural network explainer. It was introduced in [This Paper](https://arxiv.org/abs/1705.05598), and has since been adopted into the [iNNvestigate Package](https://arxiv.org/abs/1808.04260). Users are advised to check out [The Official iNNvestigate repo](https://github.com/albermax/innvestigate). It provides a nice set of options for methods that visualise the signal of the input; those parts that contributed most to the network's prediction. The iNNvestigate package uses the Keras framework. If you want to apply PatternNet to your PyTorch model you can either [convert it to a Keras model](https://github.com/nerox8664/pytorch2keras) and use the official package, or use this repo.

### A note on PatternNet
PatternNet computes the signal directions per neuron by learning it from data. It is required to run one epoch over a training set (may be the same or a different set that the model weights were trained on). This version assumes ReLU activations for all but the last layer. In principle the method can work for any piecewise linear activation function. This package does not work for recurrent models.

## Getting Started

### Prerequisites

```
Numpy
PyTorch
```

To run the mnist example:

```
dill
matplotlib
tqdm
```

To run the assertion file:

```
keras
innvestigate
```


## Running the tests

The mnist example can be used for reference on how to initialise and train the signal estimator and can be run with

```
python mnist.py
```

To make sure our implementation outputs the same patterns as the original package we've included a file that tests the two against each other. To run it, call

```
python assertion.py
```

