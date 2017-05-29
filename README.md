# MindsEye - Java 8 Neural Networks
 
Welcome! MindsEye is a project I started for personal research into deep learning and neural networks. It has grown into something I think has promise and I hope might be of interest to researchers and developers with similar needs. It was designed as a fully-featured neural network library developed using Java 8: Fast numeric calculations are performed using the JBLAS native library and Aparapi OpenCL kernels. In particular, the optimization strategy and component library are designed to be highly customizable and extendable.

## Why MindsEye?
 
I created MindsEye because…. why not? I’ve wanted to build "SkyNet" ever since I was a kid.
 
You might be interested in MindsEye bacause:
1. I am a professional software developer, not a research scientist. I’ve worked with many APIs and software patterns and architectures, and I have an accordingly different perspective on writing software. You may find my approach more compatible to your thinking - or not.
1. You want detailed control over the optimization process itself. Perhaps you are a researcher looking for an A/B comparison, or you simply want/need certain customizations. I have paid particular attention to the design of the optimization package.
1. You really prefer Java and Scala to Python - There are other Java offerings, but the main deep learning tools are currently python-centric

Some features at a glance:
1. Support for SGD, L-BFGS, OWL-QN, and other optimization algorithms.
1. Suport for many layer types, including most/all the standard ones, and the ability to easily define more.
1. High visibility into the internals - Support for detailed metrics and json model serialization.
1. Convolution operations performed using OpenCL / GPU acceleration 
1. Ability to execute data-parallel training using Apache Spark
1. Excellent visibility including model JSON serialization and progress callbacks
1. Ability to arrange processing components in an arbitrary execution graph, allowing arbitrary topologies.

**Project Website**
https://simiacryptus.github.io/MindsEye/

**JavaDoc**
https://simiacryptus.github.io/MindsEye/apidocs/index.html

**Maven**
http://mvnrepository.com/artifact/com.simiacryptus/mindseye

**Blog Articles** 
1. http://blog.simiacryptus.com/2017/05/mindseye-12.html
2. http://blog.simiacryptus.com/2017/05/neural-networks-gpus-server-clusters.html
3. http://blog.simiacryptus.com/2017/05/autoencoders-and-interactive-research.html
4. http://blog.simiacryptus.com/2017/05/a-unified-design-pattern-for-continuous.html
5. http://blog.simiacryptus.com/2015/10/re-anatomy-of-my-pet-brain.html 
6. http://blog.simiacryptus.com/2015/07/fun-with-deconvolutions-and.html

## Getting Started

[This demo](https://github.com/SimiaCryptus/MindsEye/blob/master/reports/com.simiacryptus.mindseye.MNistDemo/basic.md) will show how to create and train a basic neural network.

[This demo](https://github.com/SimiaCryptus/MindsEye/blob/master/reports/com.simiacryptus.mindseye.MNistDemo/bellsAndWhistles.md) shows various options for the optimizer.

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
