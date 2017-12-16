# NthPowerActivationLayer
## ZeroPowerTest
### Batch Execution
Code from [BatchingTester.java:66](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/BatchingTester.java#L66) executed in 0.00 seconds: 
```java
    return test(reference, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (120#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (60#)}
```



Code from [SingleDerivativeTester.java:77](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/SingleDerivativeTester.java#L77) executed in 0.00 seconds: 
```java
    return test(component, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ 1.844 ], [ -1.14 ], [ -0.744 ] ],
    	[ [ 1.268 ], [ -0.316 ], [ -1.808 ] ]
    ]
    Inputs Statistics: {meanExponent=0.009040577650570661, negative=4, min=-1.808, max=-1.808, mean=-0.14933333333333332, count=6.0, positive=2, stdDev=1.2971901257033305, zeros=0}
    Output: [
    	[ [ 1.0 ], [ 1.0 ], [ 1.0 ] ],
    	[ [ 1.0 ], [ 1.0 ], [ 1.0 ] ]
    ]
    Outputs Statistics: {meanExponent=0.0, negative=0, min=1.0, max=1.0, mean=1.0, count=6.0, positive=6, stdDev=0.0, zeros=0}
    Feedback for input 0
    Inputs Values: [
    	[ [ 1.844 ], [ -1.14 ], [ -0.744 ] ],
    	[ [ 1.268 ], [ -0.316 ], [ -1.808 ] ]
    ]
    Value Statistics: {meanExponent=0.009040577650570661, negative=4, min=-1.808, max=-1.808, mean=-0.14933333333333332, count=6.0, positive=2, stdDev=1.2971901257033305, zeros=0}
    Implemented Feedback: [ [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ] ]
    Implemented Statistics: {meanExponent=NaN, negative=0, min=0.0, max=0.0, mean=0.0, count=36.0, positive=0, stdDev=0.0, zeros=36}
    Measured Feedback: [ [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ] ]
    Measured Statistics: {meanExponent=NaN, negative=0, min=0.0, max=0.0, mean=0.0, count=36.0, positive=0, stdDev=0.0, zeros=36}
    Feedback Error: [ [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ] ]
    Error Statistics: {meanExponent=NaN, negative=0, min=0.0, max=0.0, mean=0.0, count=36.0, positive=0, stdDev=0.0, zeros=36}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (36#)
    relativeTol: 0.0000e+00 +- 0.0000e+00 [Infinity - -Infinity] (0#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (36#), relativeTol=0.0000e+00 +- 0.0000e+00 [Infinity - -Infinity] (0#)}
```



### Json Serialization
Code from [JsonTest.java:36](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/JsonTest.java#L36) executed in 0.00 seconds: 
```java
    JsonObject json = layer.getJson();
    NNLayer echo = NNLayer.fromJson(json);
    if ((echo == null)) throw new AssertionError("Failed to deserialize");
    if ((layer == echo)) throw new AssertionError("Serialization did not copy");
    if ((!layer.equals(echo))) throw new AssertionError("Serialization not equal");
    return new GsonBuilder().setPrettyPrinting().create().toJson(json);
```

Returns: 

```
    {
      "class": "com.simiacryptus.mindseye.layers.java.NthPowerActivationLayer",
      "id": "6f2b3537-d481-46b4-a4ed-9921c075d3b5",
      "isFrozen": false,
      "name": "NthPowerActivationLayer/6f2b3537-d481-46b4-a4ed-9921c075d3b5",
      "power": 0.0
    }
```



### Example Input/Output Pair
Code from [ReferenceIO.java:68](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/ReferenceIO.java#L68) executed in 0.00 seconds: 
```java
    SimpleEval eval = SimpleEval.run(layer, inputPrototype);
    return String.format("--------------------\nInput: \n[%s]\n--------------------\nOutput: \n%s\n--------------------\nDerivative: \n%s",
      Arrays.stream(inputPrototype).map(t -> t.prettyPrint()).reduce((a, b) -> a + ",\n" + b).get(),
      eval.getOutput().prettyPrint(),
      Arrays.stream(eval.getDerivative()).map(t -> t.prettyPrint()).reduce((a, b) -> a + ",\n" + b).get());
```

Returns: 

```
    --------------------
    Input: 
    [[
    	[ [ 1.552 ], [ -1.652 ], [ 1.736 ] ],
    	[ [ -0.868 ], [ -0.012 ], [ -1.82 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 1.0 ], [ 1.0 ], [ 1.0 ] ],
    	[ [ 1.0 ], [ 1.0 ], [ 1.0 ] ]
    ]
    --------------------
    Derivative: 
    [
    	[ [ 0.0 ], [ 0.0 ], [ 0.0 ] ],
    	[ [ 0.0 ], [ 0.0 ], [ 0.0 ] ]
    ]
```



### Input Learning
In this test, we use a network to learn this target input, given it's pre-evaluated output:

Code from [LearningTester.java:127](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L127) executed in 0.00 seconds: 
```java
    return Arrays.stream(input_target).map(x -> x.prettyPrint()).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [
    	[ [ -1.484 ], [ 0.132 ], [ 1.096 ], [ -1.788 ], [ 0.64 ], [ 1.924 ], [ 1.12 ], [ 0.352 ], ... ],
    	[ [ -1.9 ], [ -1.016 ], [ 1.82 ], [ -0.608 ], [ 0.748 ], [ 0.632 ], [ 0.12 ], [ -0.72 ], ... ],
    	[ [ 0.52 ], [ 0.844 ], [ -1.976 ], [ -0.436 ], [ -0.404 ], [ -0.492 ], [ 0.232 ], [ -0.8 ], ... ],
    	[ [ -1.268 ], [ -0.216 ], [ -0.092 ], [ 1.02 ], [ -0.676 ], [ -0.836 ], [ -0.244 ], [ 1.68 ], ... ],
    	[ [ 0.156 ], [ -0.9 ], [ 1.828 ], [ 0.9 ], [ 0.94 ], [ -0.376 ], [ 1.368 ], [ 1.124 ], ... ],
    	[ [ 0.288 ], [ -1.024 ], [ 0.364 ], [ 1.376 ], [ 1.096 ], [ 1.168 ], [ -0.804 ], [ -1.88 ], ... ],
    	[ [ -0.136 ], [ 0.5 ], [ -1.912 ], [ -0.912 ], [ -0.8 ], [ -0.74 ], [ -1.404 ], [ -0.984 ], ... ],
    	[ [ 0.652 ], [ -0.4 ], [ 1.784 ], [ 1.608 ], [ 0.708 ], [ -1.04 ], [ 0.54 ], [ 1.94 ], ... ],
    	...
    ]
```



First, we use a conjugate gradient descent method, which converges the fastest for purely linear functions.

Code from [LearningTester.java:225](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L225) executed in 0.00 seconds: 
```java
    return new IterativeTrainer(trainable)
      .setLineSearchFactory(label -> new QuadraticSearch())
      .setOrientation(new GradientDescent())
      .setMonitor(monitor)
      .setTimeout(30, TimeUnit.SECONDS)
      .setMaxIterations(250)
      .setTerminateThreshold(0)
      .run();
```

Returns: 

```
    0.0
```



This training run resulted in the following regressed input:

Code from [LearningTester.java:144](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L144) executed in 0.00 seconds: 
```java
    return Arrays.stream(input_gd).map(x -> x.prettyPrint()).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [
    	[ [ -1.996 ], [ -0.052 ], [ -0.432 ], [ 1.108 ], [ 1.056 ], [ 0.632 ], [ 0.96 ], [ -1.572 ], ... ],
    	[ [ -0.34 ], [ -0.948 ], [ 1.94 ], [ 0.824 ], [ 0.128 ], [ 0.432 ], [ -0.08 ], [ 1.708 ], ... ],
    	[ [ 1.068 ], [ 1.3 ], [ 0.908 ], [ -1.26 ], [ 0.42 ], [ 1.076 ], [ 0.1 ], [ -0.096 ], ... ],
    	[ [ 0.944 ], [ 1.164 ], [ -1.308 ], [ 1.868 ], [ 0.76 ], [ -0.644 ], [ -1.224 ], [ -0.816 ], ... ],
    	[ [ -0.004 ], [ 1.064 ], [ -1.4 ], [ 0.828 ], [ 1.92 ], [ -0.7 ], [ 0.476 ], [ -1.388 ], ... ],
    	[ [ 1.436 ], [ -0.2 ], [ 0.144 ], [ 0.008 ], [ -1.96 ], [ 1.636 ], [ 0.256 ], [ -1.436 ], ... ],
    	[ [ 0.008 ], [ -1.288 ], [ -0.904 ], [ 1.988 ], [ -1.02 ], [ 1.488 ], [ -0.268 ], [ 1.54 ], ... ],
    	[ [ 0.152 ], [ -0.132 ], [ -0.384 ], [ 1.76 ], [ 1.488 ], [ 0.496 ], [ 0.66 ], [ -0.124 ], ... ],
    	...
    ]
```



Next, we run the same optimization using L-BFGS, which is nearly ideal for purely second-order or quadratic functions.

Code from [LearningTester.java:249](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L249) executed in 0.00 seconds: 
```java
    return new IterativeTrainer(trainable)
      .setLineSearchFactory(label -> new ArmijoWolfeSearch())
      .setOrientation(new LBFGS())
      .setMonitor(monitor)
      .setTimeout(30, TimeUnit.SECONDS)
      .setMaxIterations(250)
      .setTerminateThreshold(0)
      .run();
```

Returns: 

```
    0.0
```



This training run resulted in the following regressed input:

Code from [LearningTester.java:154](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L154) executed in 0.00 seconds: 
```java
    return Arrays.stream(input_lbgfs).map(x -> x.prettyPrint()).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [
    	[ [ -1.996 ], [ -0.052 ], [ -0.432 ], [ 1.108 ], [ 1.056 ], [ 0.632 ], [ 0.96 ], [ -1.572 ], ... ],
    	[ [ -0.34 ], [ -0.948 ], [ 1.94 ], [ 0.824 ], [ 0.128 ], [ 0.432 ], [ -0.08 ], [ 1.708 ], ... ],
    	[ [ 1.068 ], [ 1.3 ], [ 0.908 ], [ -1.26 ], [ 0.42 ], [ 1.076 ], [ 0.1 ], [ -0.096 ], ... ],
    	[ [ 0.944 ], [ 1.164 ], [ -1.308 ], [ 1.868 ], [ 0.76 ], [ -0.644 ], [ -1.224 ], [ -0.816 ], ... ],
    	[ [ -0.004 ], [ 1.064 ], [ -1.4 ], [ 0.828 ], [ 1.92 ], [ -0.7 ], [ 0.476 ], [ -1.388 ], ... ],
    	[ [ 1.436 ], [ -0.2 ], [ 0.144 ], [ 0.008 ], [ -1.96 ], [ 1.636 ], [ 0.256 ], [ -1.436 ], ... ],
    	[ [ 0.008 ], [ -1.288 ], [ -0.904 ], [ 1.988 ], [ -1.02 ], [ 1.488 ], [ -0.268 ], [ 1.54 ], ... ],
    	[ [ 0.152 ], [ -0.132 ], [ -0.384 ], [ 1.76 ], [ 1.488 ], [ 0.496 ], [ 0.66 ], [ -0.124 ], ... ],
    	...
    ]
```



Code from [LearningTester.java:95](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L95) executed in 0.00 seconds: 
```java
    return TestUtil.compare(runs);
```

Code from [LearningTester.java:98](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L98) executed in 0.00 seconds: 
```java
    return TestUtil.compareTime(runs);
```

### Performance
Now we execute larger-scale runs to benchmark performance:

Code from [PerformanceTester.java:66](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/PerformanceTester.java#L66) executed in 0.15 seconds: 
```java
    test(component, inputPrototype);
```
Logging: 
```
    100 batches
    Input Dimensions:
    	[100, 100, 1]
    Performance:
    	Evaluation performance: 0.005171s +- 0.003502s [0.002829s - 0.012132s]
    	Learning performance: 0.013161s +- 0.001990s [0.010541s - 0.016140s]
    
```

### Function Plots
Code from [ActivationLayerTestBase.java:110](../../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/java/ActivationLayerTestBase.java#L110) executed in 0.00 seconds: 
```java
    return plot("Value Plot", plotData, x -> new double[]{x[0], x[1]});
```

Returns: 

![Result](etc/test.221.png)



Code from [ActivationLayerTestBase.java:114](../../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/java/ActivationLayerTestBase.java#L114) executed in 0.00 seconds: 
```java
    return plot("Derivative Plot", plotData, x -> new double[]{x[0], x[2]});
```

Returns: 

![Result](etc/test.222.png)



