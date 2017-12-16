# NthPowerActivationLayer
## ZeroPowerTest
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
      "id": "985f58f9-7e3c-471d-993d-cf54bbdd531e",
      "isFrozen": false,
      "name": "NthPowerActivationLayer/985f58f9-7e3c-471d-993d-cf54bbdd531e",
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
    	[ [ 0.4 ], [ 1.336 ], [ -1.984 ] ],
    	[ [ 1.74 ], [ 1.116 ], [ 1.344 ] ]
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
    	[ [ 1.304 ], [ 1.152 ], [ -1.052 ] ],
    	[ [ -0.06 ], [ -1.004 ], [ -0.82 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.18459256235380403, negative=4, min=-0.82, max=-0.82, mean=-0.08000000000000002, count=6.0, positive=2, stdDev=0.9814818728161344, zeros=0}
    Output: [
    	[ [ 1.0 ], [ 1.0 ], [ 1.0 ] ],
    	[ [ 1.0 ], [ 1.0 ], [ 1.0 ] ]
    ]
    Outputs Statistics: {meanExponent=0.0, negative=0, min=1.0, max=1.0, mean=1.0, count=6.0, positive=6, stdDev=0.0, zeros=0}
    Feedback for input 0
    Inputs Values: [
    	[ [ 1.304 ], [ 1.152 ], [ -1.052 ] ],
    	[ [ -0.06 ], [ -1.004 ], [ -0.82 ] ]
    ]
    Value Statistics: {meanExponent=-0.18459256235380403, negative=4, min=-0.82, max=-0.82, mean=-0.08000000000000002, count=6.0, positive=2, stdDev=0.9814818728161344, zeros=0}
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



### Performance
Now we execute larger-scale runs to benchmark performance:

Code from [PerformanceTester.java:66](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/PerformanceTester.java#L66) executed in 0.10 seconds: 
```java
    test(component, inputPrototype);
```
Logging: 
```
    100 batches
    Input Dimensions:
    	[100, 100, 1]
    Performance:
    	Evaluation performance: 0.005301s +- 0.004155s [0.002761s - 0.013591s]
    	Learning performance: 0.012613s +- 0.000269s [0.012263s - 0.013011s]
    
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
    	[ [ -0.56 ], [ -1.296 ], [ 0.968 ], [ 1.32 ], [ 1.128 ], [ 1.188 ], [ -1.716 ], [ -0.244 ], ... ],
    	[ [ -0.188 ], [ -1.088 ], [ -1.956 ], [ 0.464 ], [ -0.208 ], [ 1.292 ], [ -0.792 ], [ -1.82 ], ... ],
    	[ [ -0.412 ], [ -0.02 ], [ 1.428 ], [ -0.468 ], [ -1.012 ], [ 1.16 ], [ -0.22 ], [ 0.368 ], ... ],
    	[ [ -0.892 ], [ 0.492 ], [ -0.052 ], [ -1.636 ], [ -1.412 ], [ 1.28 ], [ 1.26 ], [ 1.352 ], ... ],
    	[ [ -1.008 ], [ -1.636 ], [ 1.288 ], [ 1.688 ], [ 0.86 ], [ 1.016 ], [ 0.504 ], [ -1.776 ], ... ],
    	[ [ -0.796 ], [ 0.292 ], [ -1.528 ], [ -0.232 ], [ -1.28 ], [ -0.492 ], [ -0.536 ], [ 0.192 ], ... ],
    	[ [ 0.832 ], [ -0.84 ], [ 1.648 ], [ -0.564 ], [ 0.16 ], [ -0.228 ], [ 0.608 ], [ 1.972 ], ... ],
    	[ [ 0.848 ], [ -1.58 ], [ -1.164 ], [ -1.08 ], [ 1.852 ], [ -1.884 ], [ -1.764 ], [ 1.996 ], ... ],
    	...
    ]
```



First, we use a conjugate gradient descent method, which converges the fastest for purely linear functions.

Code from [LearningTester.java:300](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L300) executed in 0.00 seconds: 
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
    	[ [ -1.484 ], [ -1.772 ], [ -0.312 ], [ -1.744 ], [ -0.212 ], [ -1.348 ], [ -0.632 ], [ 0.516 ], ... ],
    	[ [ 1.66 ], [ -1.116 ], [ -0.012 ], [ 1.084 ], [ 1.268 ], [ -1.252 ], [ 0.236 ], [ -1.144 ], ... ],
    	[ [ -1.092 ], [ 1.448 ], [ 0.28 ], [ 1.78 ], [ -0.228 ], [ -1.704 ], [ -0.688 ], [ -1.588 ], ... ],
    	[ [ 1.272 ], [ -1.86 ], [ 0.348 ], [ 1.04 ], [ -1.028 ], [ -1.38 ], [ 0.176 ], [ 0.292 ], ... ],
    	[ [ -0.936 ], [ 1.268 ], [ 1.056 ], [ 1.744 ], [ 1.136 ], [ -1.196 ], [ -1.0 ], [ -1.456 ], ... ],
    	[ [ 0.02 ], [ -1.512 ], [ 0.744 ], [ 0.476 ], [ -0.912 ], [ 1.752 ], [ -0.156 ], [ -0.068 ], ... ],
    	[ [ -1.328 ], [ 1.648 ], [ 0.544 ], [ 0.74 ], [ 1.396 ], [ -0.476 ], [ 0.752 ], [ 0.428 ], ... ],
    	[ [ -1.828 ], [ 0.984 ], [ 1.9 ], [ -0.232 ], [ 1.168 ], [ -0.776 ], [ 1.04 ], [ -0.836 ], ... ],
    	...
    ]
```



Next, we run the same optimization using L-BFGS, which is nearly ideal for purely second-order or quadratic functions.

Code from [LearningTester.java:324](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L324) executed in 0.00 seconds: 
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
    	[ [ -1.484 ], [ -1.772 ], [ -0.312 ], [ -1.744 ], [ -0.212 ], [ -1.348 ], [ -0.632 ], [ 0.516 ], ... ],
    	[ [ 1.66 ], [ -1.116 ], [ -0.012 ], [ 1.084 ], [ 1.268 ], [ -1.252 ], [ 0.236 ], [ -1.144 ], ... ],
    	[ [ -1.092 ], [ 1.448 ], [ 0.28 ], [ 1.78 ], [ -0.228 ], [ -1.704 ], [ -0.688 ], [ -1.588 ], ... ],
    	[ [ 1.272 ], [ -1.86 ], [ 0.348 ], [ 1.04 ], [ -1.028 ], [ -1.38 ], [ 0.176 ], [ 0.292 ], ... ],
    	[ [ -0.936 ], [ 1.268 ], [ 1.056 ], [ 1.744 ], [ 1.136 ], [ -1.196 ], [ -1.0 ], [ -1.456 ], ... ],
    	[ [ 0.02 ], [ -1.512 ], [ 0.744 ], [ 0.476 ], [ -0.912 ], [ 1.752 ], [ -0.156 ], [ -0.068 ], ... ],
    	[ [ -1.328 ], [ 1.648 ], [ 0.544 ], [ 0.74 ], [ 1.396 ], [ -0.476 ], [ 0.752 ], [ 0.428 ], ... ],
    	[ [ -1.828 ], [ 0.984 ], [ 1.9 ], [ -0.232 ], [ 1.168 ], [ -0.776 ], [ 1.04 ], [ -0.836 ], ... ],
    	...
    ]
```



Code from [LearningTester.java:96](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L96) executed in 0.00 seconds: 
```java
    return TestUtil.compare(runs);
```

Code from [LearningTester.java:99](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L99) executed in 0.00 seconds: 
```java
    return TestUtil.compareTime(runs);
```

### Function Plots
Code from [ActivationLayerTestBase.java:110](../../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/java/ActivationLayerTestBase.java#L110) executed in 0.00 seconds: 
```java
    return plot("Value Plot", plotData, x -> new double[]{x[0], x[1]});
```

Returns: 

![Result](etc/test.215.png)



Code from [ActivationLayerTestBase.java:114](../../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/java/ActivationLayerTestBase.java#L114) executed in 0.00 seconds: 
```java
    return plot("Derivative Plot", plotData, x -> new double[]{x[0], x[2]});
```

Returns: 

![Result](etc/test.216.png)



