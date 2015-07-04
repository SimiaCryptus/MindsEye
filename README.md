# mindseye

An experimentation project dealing with neural networks and backpropigation techniques with the eventual goal of focusing on image processing tasks such as classification, autoencoders, and convolutional neural nets.

[Inspired by the recent work at Google.](http://googleresearch.blogspot.com/2015/06/inceptionism-going-deeper-into-neural.html)

## Background

Comming soon. (https://en.wikipedia.org/wiki/Backpropagation)

## How To Run

This project is a standard maven project with minimal dependencies & complexity. Currently the only entry points are junit tests, with some of the stable ones being:

1. [NetworkElementUnitTests](https://github.com/acharneski/mindseye/blob/master/src/test/java/com/simiacryptus/mindseye/test/NetworkElementUnitTests.java) - This demonstrates very basic "networks" and learning tasks that are intended to test one aspect of one component at a time.
2. [SimpleNetworkTests](https://github.com/acharneski/mindseye/blob/master/src/test/java/com/simiacryptus/mindseye/test/SimpleNetworkTests.java) - Very simple problems, such as boolean AND/XOR logic emulators, being solved with simple networks.

Additionally, I have prepared some code to ingest some training data sets for eventual use in r&d:

1. [MNIST](https://github.com/acharneski/mindseye/blob/master/src/test/java/com/simiacryptus/mindseye/data/TestMNIST.java) - Handwritten numbers sampled as 26x26 images
2. [CIFAR](https://github.com/acharneski/mindseye/blob/master/src/test/java/com/simiacryptus/mindseye/data/TestCIFAR.java) - Image categorization based on small thumbnail-type images




