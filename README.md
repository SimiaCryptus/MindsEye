# MindsEye - Java 8 Neural Networks
 
Welcome! This is MindsEye, a neural network library written in Java 8 and Scala.

**Project Website -** [https://simiacryptus.github.io/MindsEye/](https://simiacryptus.github.io/MindsEye/) 

**JavaDoc -** [https://simiacryptus.github.io/MindsEye/apidocs/index.html](https://simiacryptus.github.io/MindsEye/apidocs/index.html) 

**Maven -** [http://mvnrepository.com/artifact/com.simiacryptus/mindseye](http://mvnrepository.com/artifact/com.simiacryptus/mindseye) 

# Introduction

## Why MindsEye?

* You prefer Java/Scala to Python - Most current NN toolkits are either in Python or predominantly used via Python.

* You want fine-grained control over the optimization process - We have a highly customizable, modular optimization engine that runs many popular algorithms.

## Features

* Supports SGD, L-BFGS, OWL-QN, and other optimization algorithms

* Wide variety of components, including nested DAGs

* Convolution layers are GPU accelerated using Aparapi/OpenCL

* Leverage server clusters with Apache Spark

* Transparent, hackable model with JSON serialization

## Blog Articles

* [http://blog.simiacryptus.com/2017/05/mindseye-12.html](http://blog.simiacryptus.com/2017/05/mindseye-12.html)

* [http://blog.simiacryptus.com/2017/05/neural-networks-gpus-server-clusters.html](http://blog.simiacryptus.com/2017/05/neural-networks-gpus-server-clusters.html)

* [http://blog.simiacryptus.com/2017/05/autoencoders-and-interactive-research.html](http://blog.simiacryptus.com/2017/05/autoencoders-and-interactive-research.html)

* [http://blog.simiacryptus.com/2017/05/a-unified-design-pattern-for-continuous.html](http://blog.simiacryptus.com/2017/05/a-unified-design-pattern-for-continuous.html)

* [http://blog.simiacryptus.com/2015/10/re-anatomy-of-my-pet-brain.html](http://blog.simiacryptus.com/2015/10/re-anatomy-of-my-pet-brain.html)

* [http://blog.simiacryptus.com/2015/07/fun-with-deconvolutions-and.html](http://blog.simiacryptus.com/2015/07/fun-with-deconvolutions-and.html)

# Fundamentals

## Background

For background knowledge, the reader is referred to the following topics:

* Linear Algebra and Calculus

* Bayesian Statistics

* Backpropagation Learning

* Convolutional Networks

* Image Processing

* Quasi-Newtonian Optimization

## Foundation Types

A few central types form the basis for this entire library:

### Tensor

The [Tensor](https://github.com/SimiaCryptus/utilities/blob/master/java-util/src/main/java/com/simiacryptus/util/ml/Tensor.java) class provides facilities for efficiently using multidimensional arrays. It has a fairly simple API of get and set methods, as well as a variety of optimizations for index access, memory management, etc. These arrays are allocated as a single object, and they use a dense rectangular layout - this means all bounds are uniform (dimension 1 always has the same range, as does dimension 2, etc) and all values are stored in memory (no compression of sparseness)

Many components assume a specific "image tensor schema", which includes the following four dimensions: 

1. **Spacial dimension: X** - The first dimension, organizing rows of pixels into a spatial structure

2. **Spacial dimension: Y** - A second dimension with a local spatial structure

3. **Set dimension: Color Band** - This dimension organizes the color bands in a normal raster image, and represents an unstructured data vector.

4. **Sequence dimension: Batch Index** - This dimension allows many training examples to be stacked into a sequence of independent values.

### NNLayer

The [NNLayer](https://github.com/SimiaCryptus/MindsEye/blob/master/src/main/java/com/simiacryptus/mindseye/layers/NNLayer.java) class defines the fundamental unit of our neural networks: a single differentiable multivariate operation which may contain state. Its methods and the related classes define the core capabilities:

* eval() - The eval method performs the primary ("forward") calculation. 

* getJson() - Serializes the component.

* fromJson() - Static method to deserialize a generic NNLayer component.

A [NNResult](https://github.com/SimiaCryptus/MindsEye/blob/master/src/main/java/com/simiacryptus/mindseye/layers/NNResult.java) provides both the result data and a callback method to evaluate the gradient using a given DeltaSet. NNResult is also used as the input to the central eval method, which enables chaining these differentiable operations together. By chaining them together, we can form larger NNLayer components; they are composable.

A [DeltaSet](https://github.com/SimiaCryptus/MindsEye/blob/master/src/main/java/com/simiacryptus/mindseye/layers/DeltaSet.java) provides a buffer of pending deltas to apply to various double arrays. It keeps references to the layer and specific double[] data location, and accumulates values in an identically sized buffer collection. This gives us a convenient interface for working with and applying delta vectors.

# Component Library

There is a large and ever-changing library of components available, organized roughly into these categories: 

* [Activation Layers](https://github.com/SimiaCryptus/MindsEye/tree/master/src/main/java/com/simiacryptus/mindseye/layers/activation) – Sigmoid, Softmax, etc – Classical, stateless differentiable functions used to introduce nonlinear effects and perform basic computation.

* [Loss](https://github.com/SimiaCryptus/MindsEye/tree/master/src/main/java/com/simiacryptus/mindseye/layers/loss) - Various types of loss functions for comparing actual and desired outputs

* [Media](https://github.com/SimiaCryptus/MindsEye/tree/master/src/main/java/com/simiacryptus/mindseye/layers/media) - Components that assume the 4d image tensor schema, including convolutional network components.

* [Meta](https://github.com/SimiaCryptus/MindsEye/tree/master/src/main/java/com/simiacryptus/mindseye/layers/meta) - Components that operate across batch examples, breaking the usual rule where each given element in a batch is treated independently.

* [Reducers](https://github.com/SimiaCryptus/MindsEye/tree/master/src/main/java/com/simiacryptus/mindseye/layers/reducers) - Functions used to summarize multivariate data of potentially unspecified size, for example Sum, Average, and Max.

* [Synapse](https://github.com/SimiaCryptus/MindsEye/tree/master/src/main/java/com/simiacryptus/mindseye/layers/synapse) - These layers contain the majority of state that is learned by the network, including vector addition (BiasLayer) and matrix multiplication (DenseSynapseLayer)

* [Utilities](https://github.com/SimiaCryptus/MindsEye/tree/master/src/main/java/com/simiacryptus/mindseye/layers/util) - These non-functional components perform other tasks such as aggregating various statistics.

## OpenCL

This library uses Aparapi to compile Java code into OpenCL kernels. There are three main kernels, controlled by the central [ConvolutionController](https://github.com/SimiaCryptus/MindsEye/blob/master/src/main/java/com/simiacryptus/mindseye/opencl/ConvolutionController.java) class. OpenCL drivers must be installed on the host system; java emulation will be used as a fallback but the performance will be very poor.

## JSON Serialization

All components support serialization to/from json. Any network can be read via the [NNLayer::fromJson](https://github.com/SimiaCryptus/MindsEye/blob/master/src/main/java/com/simiacryptus/mindseye/layers/NNLayer.java#L102) method, and captured via its [getJson](https://github.com/SimiaCryptus/MindsEye/blob/master/src/main/java/com/simiacryptus/mindseye/layers/NNLayer.java#L115) method. JSON is intended to be the primary mechanism for serialization, though Kryo is also used (for cloning). A serialized model is shown in [one of the examples](https://simiacryptus.github.io/mindseye-scala/mnist/www/2017-06-11-18-07/model.json).

## Testing

Testing is mainly performed via finite differencing methods, implemented in [ComponentTestUtil](https://github.com/SimiaCryptus/MindsEye/blob/master/src/test/java/com/simiacryptus/mindseye/layers/ComponentTestUtil.java).

# Networks

In order for components to be useful, we need to be able to combine them into networks. The NNLayer/NNResult structures provide foundational support, which is then wrapped in more convenient network creation APIs outlined in this section. Each of these networks is itself also a NNLayer, and can thus be nested themselves into larger multi-level networks.

## Generic DAGs

The [base API](https://github.com/SimiaCryptus/MindsEye/blob/master/src/main/java/com/simiacryptus/mindseye/network/graph/DAGNetwork.java) to create networks supports arbitrary directed acyclic graphs (DAGs) and provides the most general structure with the most powerful and complex methods. Specialized networks will wrap a generic DAG with additional data and constraints to provide more convenient construction methods.

## Pipelines

The simplest and most common network is a [pipeline](https://github.com/SimiaCryptus/MindsEye/blob/master/src/main/java/com/simiacryptus/mindseye/network/PipelineNetwork.java), which is a single sequential chain of components. This simple structure also simplifies the creation API with a single, simple add() method.

## Supervised Networks

Supervised means that the ideal, desired output is given along with an input. [This type](https://github.com/SimiaCryptus/MindsEye/blob/master/src/main/java/com/simiacryptus/mindseye/network/SupervisedNetwork.java) of network has two multivariate inputs and a single univariate output. The [most common](https://github.com/SimiaCryptus/MindsEye/blob/master/src/main/java/com/simiacryptus/mindseye/network/SimpleLossNetwork.java) simply combines a loss component and a training component to provide a network.

## Composite Components

Several other networks act primarily as components themselves, intended for use in a larger network. A couple examples are normalization and inception layers.

### Regularization Layers

[Some networks](https://github.com/SimiaCryptus/mindseye-scala/blob/master/src/main/scala/util/NetworkMetaNormalizers.scala) use a meta layer to aggregate some signal statistic, such as the average value, and then uses that value in an operation to remove the related component. For example, it might subtract the average value so that the result has a zero mean.

### Inception Layer

The "[Inception](https://github.com/SimiaCryptus/MindsEye/blob/master/src/main/java/com/simiacryptus/mindseye/network/InceptionLayer.java)" layer was introduced by the GoogLeNet paper. In its general form, it is a set of parallel pipelines which process an input image using a given series of kernels in each pipeline. At the end of each pipeline is then several image tensors, all with the same spatial dimensions but an arbitrary number of bands. A single output image tensor is then formed by concatenating all the bands together.

# Optimization

Particular attention has been paid during the development of this library on the optimization components. We have identified commonalities in the typical optimization algorithms and captured them as individual components. First, we will review this structure, then review how major optimization algorithms are formed using them.

## Basic Structure

First we need to combine our network with training data and an execution method. This produces a [Trainable](https://github.com/SimiaCryptus/MindsEye/blob/master/src/main/java/com/simiacryptus/mindseye/opt/trainable/Trainable.java) object, a self-contained optimization problem that hides virtually all details of the network or data set, so we can give it to a generic optimizer. This object controls data sampling behavior, providing approximations with controlled randomization.


The **Iterator** defines the api for using the optimization process; the default [IterativeTrainer ](https://github.com/SimiaCryptus/MindsEye/blob/master/src/main/java/com/simiacryptus/mindseye/opt/IterativeTrainer.java)implementation will synchronously search in a loop using a single starting point and a single in-memory model. It moderates the interaction between the other components.

The Iterator will call the [Orienter](https://github.com/SimiaCryptus/MindsEye/blob/master/src/main/java/com/simiacryptus/mindseye/opt/OrientationStrategy.java)** **and [Stepper](https://github.com/SimiaCryptus/MindsEye/blob/master/src/main/java/com/simiacryptus/mindseye/opt/line/LineSearchStrategy.java)** **in a loop, passing the Monitor object around to provide debugging callbacks. The Orienter will determine the direction a given step should take, and the Stepper both determines the size of the step and performs the step, updating the model.

The Orienter is a strategy to determine the search direction. The simplest example for a differentiable function is "steepest descent" which blindly modifies the weights in the direction of the gradient. More advanced methods include the quasi-newton L-BFGS algorithm.

The Stepper searches along the given univariate function, also known as a "line search". On each step, it measures the fitness and derivative, and will iteratively attempt to find an approximate (or exact) minimum along that line. Our default implementation uses an adaptive step size window that is determined by what are known as the [Armijo and Wolfe](https://github.com/SimiaCryptus/MindsEye/blob/master/src/main/java/com/simiacryptus/mindseye/opt/line/ArmijoWolfeConditions.java) line search constraints, and takes a single “acceptable” step. These constraints basically ensure that the step size is not too large or small, respectively, by comparing both the value and gradient over the step.

Finally, the [Monitor](https://github.com/SimiaCryptus/MindsEye/blob/master/src/main/java/com/simiacryptus/mindseye/opt/TrainingMonitor.java) provides a way for the executing application to monitor the progress of the task, capturing logging and taking network snapshots, for example. It may also determine auxiliary stopping conditions.

## Standard Methods

### Stochastic Gradient Descent

The "gradient descent" part of SGD is defined by the simplest orientation strategy, which simply moves along the [raw gradient](https://github.com/SimiaCryptus/MindsEye/blob/master/src/main/java/com/simiacryptus/mindseye/opt/GradientDescent.java). The stochastic part is implemented in a trainable component ([StochasticArrayTrainable](https://github.com/SimiaCryptus/MindsEye/blob/master/src/main/java/com/simiacryptus/mindseye/opt/trainable/StochasticArrayTrainable.java)) that uses a random subset of the training data for evaluation.

### L1 and L2 normalization

Normalization factors could be implemented as part of the network itself, summing its own weights into the output. However, we can also encapsulate the trainable object to modify the optimized function. We have two such components, [one](https://github.com/SimiaCryptus/MindsEye/blob/master/src/main/java/com/simiacryptus/mindseye/opt/trainable/ConstL12Normalizer.java) uses constant factors and [the other ](https://github.com/SimiaCryptus/MindsEye/blob/master/src/main/java/com/simiacryptus/mindseye/opt/trainable/L12Normalizer.java)allows you to specify constants for each layer.

### Momentum

[Momentum](https://github.com/SimiaCryptus/MindsEye/blob/master/src/main/java/com/simiacryptus/mindseye/opt/MomentumStrategy.java) can be added to the orientation strategy by wrapping an underlying strategy such as Gradient Descent. Momentum will keep a delta vector in its memory, using it to accumulate new movement and modifying the returned vector (orientated line search).

### L-BFGS

[L-BFGS](https://github.com/SimiaCryptus/MindsEye/blob/master/src/main/java/com/simiacryptus/mindseye/opt/LBFGS.java) uses previously sampled points (along with their gradients) to estimate something known as the inverse Hessian. The Hessian is to a gradient as a second derivative is to a first, and knowing it can allow us to adapt our step since instead of a linear local model we now have a quadratic local model, which has a known minimum. This makes a more informed selection of both our step direction and our step size (since the orientated step is not necessarily a unit vector). It is implemented as an orientation strategy.

### OWL-QN

In the case of [OWL-QN](https://github.com/SimiaCryptus/MindsEye/blob/master/src/main/java/com/simiacryptus/mindseye/opt/OwlQn.java), the Orientation strategy passes a special "line search problem" to the stepper that isn’t a simple straight line. Instead, the line is constrained to the current “orthant”, and will follow the orthant boundaries where some weight values are zero. This “trust region” approach is very effective in finding a sparse model where many weights are zero.

OWL-QN is also implemented as an orientation strategy which wraps another "base" strategy. This new strategy will then wrap the line search object returned by the base strategy, and this new line search object will implement the orthant.

A very [similar strategy](https://github.com/SimiaCryptus/MindsEye/blob/master/src/main/java/com/simiacryptus/mindseye/opt/region/TrustRegionStrategy.java) is used in a more general fashion to support arbitrary per-layer trust regions, as discussed below.

## Trust Regions

[Trust Regions](https://github.com/SimiaCryptus/MindsEye/blob/master/src/main/java/com/simiacryptus/mindseye/opt/region/TrustRegion.java) can let us bound where each step takes us, controlling the search while minimally interfering with the overall optimization. They can cause the search to proceed with a different pattern, as in OWL-QN where we can promote sparse values. This can also be used to specify that a particular layer should not change at all, which is effectively a point-sized trust region. They can also enforce arbitrary constraints on the weights, such as "no weights are negative" in a particular bias layer.

Trust regions work by remapping a proposed point (i.e. Set of parameters) onto its nearest point within the trust region, which is a multidimensional volume determined by the point at the start of each step. The result is visualized by an otherwise uninterrupted line search being projected onto the surface of a volume when the line leaves the volume. 


For a valid trust region function, we have some requirements:

* The renormalized point, if it is different from the proposed point, must be on the volume’s boundary

* The vector from the normalized to the proposed point is perpendicular to the volume surface

* Every point in space must be reachable from some walk of points, each step of which falls within the respective trust region

* The point used to define the volume is within the volume

Many different trust regions are provided, including:

* [SingleOrthant](https://github.com/SimiaCryptus/MindsEye/blob/master/src/main/java/com/simiacryptus/mindseye/opt/region/SingleOrthant.java) - Enforces the constraint used in OWL-QN

* [StaticConstraint](https://github.com/SimiaCryptus/MindsEye/blob/master/src/main/java/com/simiacryptus/mindseye/opt/region/StaticConstraint.java) - Prevents changes to the weights

* [LinearSumConstraint](https://github.com/SimiaCryptus/MindsEye/blob/master/src/main/java/com/simiacryptus/mindseye/opt/region/LinearSumConstraint.java) - The sum of the absolute value of weights cannot increase (simplified explanation)

* [MeanVarianceGradient](https://github.com/SimiaCryptus/MindsEye/blob/master/src/main/java/com/simiacryptus/mindseye/opt/region/MeanVarianceGradient.java) - Allows only a regular offset and scaling factor to be applied equally to the weights. The preserves randomness or configuration, depending.

* [CompoundRegion](https://github.com/SimiaCryptus/MindsEye/blob/master/src/main/java/com/simiacryptus/mindseye/opt/region/CompoundRegion.java) - Allows multiple trust regions to be combined by intersection.

## Spark Execution

An [implementation](https://github.com/SimiaCryptus/MindsEye/blob/master/src/main/java/com/simiacryptus/mindseye/opt/trainable/SparkTrainable.java) of a Trainer is provided for data-parallel training using Spark. This allows an RDD of Tensors to be evaluated against the network across a cluster, performing a separate Spark operation for each training step. It is of use if a single step of training takes several minutes on a single machine.

## Monitoring

MindsEye supports extensive monitoring via a fairly simple API. The [monitor](https://github.com/SimiaCryptus/MindsEye/blob/master/src/main/java/com/simiacryptus/mindseye/opt/TrainingMonitor.java) class that is supplied to the trainer provides just two functions: a logging callback, used to monitor real-time activity and intercept verbose logging data, and a set completion event which can be used to copy the model and perform other tasks. Monitoring is supported by a few other components:

* A [monitored object](https://github.com/SimiaCryptus/utilities/blob/master/java-util/src/main/java/com/simiacryptus/util/MonitoredObject.java) class provides a basis for an on-demand metric state object, which can for example be provided to the user as a json object

* A [monitored "wrapper"](https://github.com/SimiaCryptus/MindsEye/blob/master/src/main/java/com/simiacryptus/mindseye/layers/util/MonitoringWrapper.java) class encapsulates another layer and reports statistics about the layer’s weights and performance.

* A [monitored "synapse"](https://github.com/SimiaCryptus/MindsEye/blob/master/src/main/java/com/simiacryptus/mindseye/layers/util/MonitoringSynapse.java) class provides a transparent pass-through or no-op while monitoring statistics of the forward and backpropagation signals.

By using the step completion event to capture all the metrics made available by the monitored network components, we produce a voluminous data stream that describes the dynamics of our learning process. This data is reported in the scala demonstration notebooks as both plots and raw data csv. An example of this can be viewed [here](https://simiacryptus.github.io/mindseye-scala/mnist/www/2017-06-11-18-07/metricsHistory.html).

# Walkthroughs

## Notebook Structure

For efficient demonstration and research, MindsEye includes utilities to produce notebook-style output reports. These reports interleave code, text, and data to summarize the entirety of a machine learning script. Combined with json serialization, we have an effective platform to develop AI models.

The full API of the notebook classes are undocumented and somewhat ad-hoc, but can be copied by example from a number of Java and Scala-based notebooks I’ve published, including:

* A java notebook demonstrating basic library use

* A java notebook demonstrating optimizer options

* A scala notebook, also demonstrating MNIST modeling

The Java notebooks are included in the main library, and serves the purpose of documentation as much as testing. The Scala notebook has many features built atop the basic APIs, which among other things records detailed metrics about the learning process. The Scala environment is intended for active research and iterative model development.

## How To Run

This is a standard Java library, using Git and Maven. The only requirements for the host system is that a JDK and OpenCL drivers are installed. OpenCL drivers can be downloaded from either Intel or NVidia, depending on which is suitable for your GPU. Below are the step-by-step CLI instructions for running the Scala demo:

### Setup EC2 Node

Creating an EC2 node for AI is very easy. The cost is currently about 0.9 $/hr.

1. Log into AWS and create a new virtual machine, of type [p2.xlarge](https://aws.amazon.com/ec2/instance-types/p2/)
(This is the only AWS instance type that supports OpenCL, and MindsEye can only leverage a single GPU. For larger scale, use Spark.)

2. Log into the EC2 instance and install needed software:
sudo apt-get update
sudo apt-get install maven git default-jdk nvidia-opencl-dev

### Run Example Report

Running any of these demos is as simple as cloning with Git and building with Maven:

1. Clone and build mindseye-scala:
```
git clone https://github.com/SimiaCryptus/mindseye-scala.git
cd mindseye-scala
mvn install -DskipTests
```

2. Execute the MNIST demo:
```
sudo java -Xmx16G -cp target/*.jar interactive.mnist.MnistDemo
```

3. Open a browser to http://localhost:port/index.html using the port logged at the application start, if the browser didn’t open itself

4. Monitor script progress. The server supports streaming results but does not reliably keep HTTP connections open, so the occasional stop/refresh is needed.

## Available Examples

* [Basic MNIST logistic regression](https://github.com/SimiaCryptus/MindsEye/blob/master/reports/com.simiacryptus.mindseye.MNistDemo/basic.md) - This markdown report demonstrates the basic construction and training of a network to recognize handwritten numbers.

* [Optimization Features](https://github.com/SimiaCryptus/MindsEye/blob/master/reports/com.simiacryptus.mindseye.MNistDemo/bellsAndWhistles.md) - This markdown report demonstrates the same process as above, but using many of the additional features of this library.

* [MNIST Research Notebook](https://simiacryptus.github.io/mindseye-scala/mnist/www/2017-06-11-18-07/index.html) - This HTML report demonstrates the additional features provided by the Scala notebook code, mostly around gathering metrics. (Follow the top row of links on the notebook for more data) There are also a few links that only work while the process is running in interactive mode.

