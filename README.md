# MindsEye - Java 8 Neural Networks

This project was designed as a fully-featured neural network library developed using Java 8. Fast numeric calculations are performed using the JBLAS native library and Aparapi OpenCL kernels. In particular, the optimization strategy and component library are designed to be highly customizable and extendable.

**Project Website**
https://simiacryptus.github.io/MindsEye/

**JavaDoc**
https://simiacryptus.github.io/MindsEye/apidocs/index.html

**Maven**
http://mvnrepository.com/artifact/com.simiacryptus/mindseye

**Blog Articles** 
1. http://blog.simiacryptus.com/2017/05/a-unified-design-pattern-for-continuous.html
2. http://blog.simiacryptus.com/2015/10/re-anatomy-of-my-pet-brain.html 
3. http://blog.simiacryptus.com/2015/07/fun-with-deconvolutions-and.html


## Basic Use Cases
 
 Several major uses cases are illustrated by a library of scala notebooks:
 
 1. [Simple 2D Classification](https://github.com/acharneski/ImageLabs/blob/master/reports/MindsEyeDemo/2d_simple.md) - Simple regression on 2 classes XY points in various simple patterns
 1. [MNIST Logistic Regression](https://github.com/acharneski/ImageLabs/blob/master/reports/MindsEyeDemo/mnist_simple.md) - A very simple network is trained on the MNIST handwritten digit dataset
 1. [Image Deblurring](https://github.com/acharneski/ImageLabs/blob/master/reports/MindsEyeDemo/deconvolution.md) - A nonlinear convolution filter is inverted when we use the trainging algorithm to reconstruct an image.
 
 ## Features
 
 **Network Components**
1. *Activation Layers* – Sigmoid, Softmax, etc – Classical, stateless differentiable functions used to introduce nonlinear effects in neural networks.
1. *Basic* – Bias, Synapse Layers – Your basic state classes, performing vector addition and matrix multiplication.
1. *Loss Functions* – RMS and Entropy – These layers compare euclidean or probabilistic results with expected values and produce a differentiable error function that can then be optimized to achive supervised learning.
1. *Media* – Convolutional Neural Networks – These components, such as the ConvolutionalSynapseLayer and MaxPoolingLayer, are designed to work on 2d image data in ways that are position-independent and/or neighborhood-local.
1. *Meta* – These components enforce constraints on global behavior of the network across training samples. This differs greatly from other components that are designed to work identically whether they process the dataset one-at-a-time or in batch. For example, one component enforces sparsity constraint component.
1. *Reducers* – Avg, Sum, Max, etc – Like activation layers, these are simple stateless functions, but they usually serve more of a structural than a functional purpose.
1. *Utils* – A variety of useful methods that are more programmatic than mathematical in nature.
    1. Verbose Logging – This component wraps a id to provide verbose logging.
    1. Weight Extraction – Many of these components are stateful and contain weight vectors that we may want to normalize via an adjustment to our fitness function.
    1. Wrapper Layer – This provides a stub wrapper so that the inner implementation can be replaced as desired. This is useful when developing networks whose layout changes over time.
 
 **Training Algorithms**
 1. Gradient Descent
 1. L-BFGS
 1. OWL-QN
