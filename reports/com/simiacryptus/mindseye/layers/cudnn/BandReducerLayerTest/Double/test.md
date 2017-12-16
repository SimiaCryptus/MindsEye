# BandReducerLayer
## Double
### Batch Execution
Code from [BatchingTester.java:66](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/BatchingTester.java#L66) executed in 0.01 seconds: 
```java
    return test(reference, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (200#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (40#)}
```



Code from [SingleDerivativeTester.java:77](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/SingleDerivativeTester.java#L77) executed in 0.01 seconds: 
```java
    return test(component, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ -1.944, -1.912 ], [ 1.248, -1.256 ], [ 0.016, -1.388 ] ],
    	[ [ 1.032, -0.544 ], [ 0.52, 1.784 ], [ 1.684, 0.8 ] ],
    	[ [ 0.912, 1.528 ], [ -0.66, 0.984 ], [ -0.852, 1.356 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.056813155682690394, negative=7, min=1.356, max=1.356, mean=0.1837777777777778, count=18.0, positive=11, stdDev=1.231838713989762, zeros=0}
    Output: [
    	[ [ 1.684, 1.784 ] ]
    ]
    Outputs Statistics: {meanExponent=0.23886846860186747, negative=0, min=1.784, max=1.784, mean=1.734, count=2.0, positive=2, stdDev=0.04999999999999947, zeros=0}
    Feedback for input 0
    Inputs Values: [
    	[ [ -1.944, -1.912 ], [ 1.248, -1.256 ], [ 0.016, -1.388 ] ],
    	[ [ 1.032, -0.544 ], [ 0.52, 1.784 ], [ 1.684, 0.8 ] ],
    	[ [ 0.912, 1.528 ], [ -0.66, 0.984 ], [ -0.852, 1.356 ] ]
    ]
    Value Statistics: {meanExponent=-0.056813155682690394, negative=7, min=1.356, max=1.356, mean=0.1837777777777778, count=18.0, positive=11, stdDev=1.231838713989762, zeros=0}
    Implemented Feedback: [ [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 
```
...[skipping 67 bytes](etc/29.txt)...
```
    .0 ], ... ]
    Implemented Statistics: {meanExponent=0.0, negative=0, min=0.0, max=0.0, mean=0.05555555555555555, count=36.0, positive=2, stdDev=0.2290614236454256, zeros=34}
    Measured Feedback: [ [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.9999999999998899, 0.0 ], ... ]
    Measured Statistics: {meanExponent=-4.7830642341045674E-14, negative=0, min=0.0, max=0.0, mean=0.05555555555554944, count=36.0, positive=2, stdDev=0.22906142364540036, zeros=34}
    Feedback Error: [ [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ -1.1013412404281553E-13, 0.0 ], ... ]
    Error Statistics: {meanExponent=-12.958078098036825, negative=2, min=0.0, max=0.0, mean=-6.118562446823085E-15, count=36.0, positive=0, stdDev=2.5227479245189218E-14, zeros=34}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 6.1186e-15 +- 2.5227e-14 [0.0000e+00 - 1.1013e-13] (36#)
    relativeTol: 5.5067e-14 +- 0.0000e+00 [5.5067e-14 - 5.5067e-14] (2#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=6.1186e-15 +- 2.5227e-14 [0.0000e+00 - 1.1013e-13] (36#), relativeTol=5.5067e-14 +- 0.0000e+00 [5.5067e-14 - 5.5067e-14] (2#)}
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
      "class": "com.simiacryptus.mindseye.layers.cudnn.BandReducerLayer",
      "id": "e9ed631a-8624-486e-9b08-b4b026f7519c",
      "isFrozen": false,
      "name": "BandReducerLayer/e9ed631a-8624-486e-9b08-b4b026f7519c",
      "mode": 0
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
    	[ [ 1.38, 1.148 ], [ -0.12, 0.992 ], [ -1.992, -0.532 ] ],
    	[ [ -1.932, -0.288 ], [ 1.568, -0.288 ], [ 1.328, -0.884 ] ],
    	[ [ 1.196, 0.816 ], [ 0.68, -1.28 ], [ 1.984, 0.532 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 1.984, 1.148 ] ]
    ]
    --------------------
    Derivative: 
    [
    	[ [ 0.0, 1.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ] ],
    	[ [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ] ],
    	[ [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 1.0, 0.0 ] ]
    ]
```



[GPU Log](etc/cuda.log)

### Input Learning
In this test, we use a network to learn this target input, given it's pre-evaluated output:

Code from [LearningTester.java:127](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L127) executed in 0.01 seconds: 
```java
    return Arrays.stream(input_target).map(x -> x.prettyPrint()).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [
    	[ [ -1.18, -1.544, -0.12 ], [ 0.32, -1.704, 1.168 ], [ -0.304, -0.844, 0.056 ], [ -1.628, 1.24, -1.832 ], [ 0.116, 1.3, 1.596 ], [ -0.392, 1.948, -1.236 ], [ 1.192, 1.984, 0.652 ], [ -0.484, -0.704, 1.4 ], ... ],
    	[ [ -1.36, 1.64, 1.436 ], [ -0.648, 0.216, -1.288 ], [ -0.872, -1.376, 1.18 ], [ 1.228, -0.412, 0.312 ], [ 1.648, 0.764, -0.128 ], [ 0.608, 1.176, -1.608 ], [ -0.536, -1.748, -0.128 ], [ 0.476, 1.74, 0.624 ], ... ],
    	[ [ 1.248, 1.58, -1.908 ], [ 0.344, -1.012, 0.344 ], [ -1.784, 1.448, 1.008 ], [ -0.392, -0.716, 1.82 ], [ -1.38, 1.404, -0.944 ], [ -0.384, -1.64, 0.712 ], [ -0.928, -0.98, 1.256 ], [ 0.252, 0.592, -0.304 ], ... ],
    	[ [ -1.9, 0.516, -1.232 ], [ 0.884, 0.364, 0.188 ], [ -1.12, -0.208, 1.956 ], [ 0.28, -0.292, -0.34 ], [ 0.208, 1.644, 0.928 ], [ 0.452, -1.848, 1.312 ], [ -1.52, -0.388, 1.696 ], [ -1.876, -0.884, -1.62 ], ... ],
    	[ [ -0.02, -1.408, -0.2 ], [ 0.848, -1.284, -0.252 ], [ -1.208, 0.016, 0.696 ], [ -1.152, 1.48, 0.62 ], [ 0.4, 0.972, 0.836 ], [ -1.544, -1.432, -0.38 ], [ 1.672, 0.992, 1.18 ], [ -1.16, -1.636, -0.388 ], ... ],
    	[ [ -1.04, -1.58, 0.932 ], [ 0.04, -1.756, -0.448 ], [ -0.412, 0.58, -1.488 ], [ -0.528, -0.948, 0.052 ], [ 0.136, -0.432, 1.772 ], [ 1.02, -1.26, 0.968 ], [ 0.708, 0.492, -1.372 ], [ -1.064, 1.94, -0.744 ], ... ],
    	[ [ 1.068, -0.38, -1.156 ], [ 1.252, 1.12, -1.2 ], [ 0.944, 1.248, 0.872 ], [ -1.056, -1.964, -1.104 ], [ -1.512, 0.812, 1.74 ], [ 0.952, 1.468, -1.224 ], [ 1.584, 1.324, 1.324 ], [ 0.908, 0.376, 1.452 ], ... ],
    	[ [ -0.648, -1.852, -0.972 ], [ -1.988, -1.724, -0.232 ], [ 0.148, 1.444, 1.864 ], [ -1.128, -0.748, -0.436 ], [ -1.688, 0.112, 1.448 ], [ -1.736, -0.108, 1.728 ], [ 1.888, -0.808, -1.824 ], [ -1.116, -1.776, 0.908 ], ... ],
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

Code from [LearningTester.java:144](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L144) executed in 0.01 seconds: 
```java
    return Arrays.stream(input_gd).map(x -> x.prettyPrint()).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [
    	[ [ -1.208, 1.016, -0.356 ], [ -0.144, -1.3, -1.732 ], [ 1.988, 0.576, -1.384 ], [ -0.072, -1.764, -1.932 ], [ 1.136, 0.632, -1.18 ], [ -1.792, 0.784, 0.832 ], [ 1.836, -1.352, 1.108 ], [ 1.652, 1.972, -0.26 ], ... ],
    	[ [ 1.144, 1.86, 1.764 ], [ 1.508, -1.144, -1.444 ], [ -1.004, 1.956, -1.232 ], [ -0.568, 1.96, -1.652 ], [ -0.372, 1.068, -0.2 ], [ -0.184, -0.684, 0.764 ], [ -1.524, 1.488, 1.42 ], [ 0.888, -1.476, -1.868 ], ... ],
    	[ [ 0.548, 0.72, -0.92 ], [ -0.068, 0.572, -0.348 ], [ 1.144, 0.532, 0.308 ], [ 0.668, -1.652, -1.396 ], [ 1.104, -1.476, -1.724 ], [ -1.724, 1.62, 1.74 ], [ -0.34, 1.112, -0.572 ], [ -1.404, 0.024, 0.756 ], ... ],
    	[ [ -0.644, 0.48, -1.652 ], [ 0.92, 0.592, -1.608 ], [ -1.18, 1.836, -0.336 ], [ 1.04, 1.556, 0.228 ], [ 1.548, -0.556, -0.512 ], [ -0.5, 0.656, 0.552 ], [ -1.008, -1.212, 1.312 ], [ 0.024, 0.284, 1.636 ], ... ],
    	[ [ 1.06, -0.968, -1.376 ], [ -1.284, -0.264, 1.512 ], [ -0.492, 0.252, 0.196 ], [ 1.648, 0.964, -0.8 ], [ -1.164, -0.82, 1.852 ], [ -1.084, 0.156, 0.072 ], [ 0.308, 1.672, -1.456 ], [ -0.148, 1.968, 0.36 ], ... ],
    	[ [ -1.564, -0.552, 1.032 ], [ -1.0, -1.896, -0.444 ], [ 1.604, 1.636, 1.496 ], [ -0.372, 0.256, -0.848 ], [ 0.868, 1.592, -0.892 ], [ -1.848, -1.668, -1.604 ], [ 0.692, 0.22, 1.612 ], [ 0.204, 0.476, 0.868 ], ... ],
    	[ [ -1.04, -0.556, 1.268 ], [ -0.148, -1.968, -0.52 ], [ 1.588, -0.812, 0.536 ], [ 1.74, -0.12, 0.124 ], [ 1.48, 1.428, -0.468 ], [ -1.916, 0.104, -0.14 ], [ 1.348, 0.76, 1.592 ], [ -0.712, -1.288, 0.416 ], ... ],
    	[ [ 1.656, 1.94, -0.104 ], [ 1.52, 1.436, 0.668 ], [ 0.412, 0.896, 0.46 ], [ 0.368, -0.692, 0.86 ], [ 1.104, 1.048, -0.436 ], [ 1.752, -1.284, -0.292 ], [ 0.604, -1.144, -1.728 ], [ 1.956, -0.7, 1.24 ], ... ],
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

Code from [LearningTester.java:154](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L154) executed in 0.01 seconds: 
```java
    return Arrays.stream(input_lbgfs).map(x -> x.prettyPrint()).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [
    	[ [ -1.208, 1.016, -0.356 ], [ -0.144, -1.3, -1.732 ], [ 1.988, 0.576, -1.384 ], [ -0.072, -1.764, -1.932 ], [ 1.136, 0.632, -1.18 ], [ -1.792, 0.784, 0.832 ], [ 1.836, -1.352, 1.108 ], [ 1.652, 1.972, -0.26 ], ... ],
    	[ [ 1.144, 1.86, 1.764 ], [ 1.508, -1.144, -1.444 ], [ -1.004, 1.956, -1.232 ], [ -0.568, 1.96, -1.652 ], [ -0.372, 1.068, -0.2 ], [ -0.184, -0.684, 0.764 ], [ -1.524, 1.488, 1.42 ], [ 0.888, -1.476, -1.868 ], ... ],
    	[ [ 0.548, 0.72, -0.92 ], [ -0.068, 0.572, -0.348 ], [ 1.144, 0.532, 0.308 ], [ 0.668, -1.652, -1.396 ], [ 1.104, -1.476, -1.724 ], [ -1.724, 1.62, 1.74 ], [ -0.34, 1.112, -0.572 ], [ -1.404, 0.024, 0.756 ], ... ],
    	[ [ -0.644, 0.48, -1.652 ], [ 0.92, 0.592, -1.608 ], [ -1.18, 1.836, -0.336 ], [ 1.04, 1.556, 0.228 ], [ 1.548, -0.556, -0.512 ], [ -0.5, 0.656, 0.552 ], [ -1.008, -1.212, 1.312 ], [ 0.024, 0.284, 1.636 ], ... ],
    	[ [ 1.06, -0.968, -1.376 ], [ -1.284, -0.264, 1.512 ], [ -0.492, 0.252, 0.196 ], [ 1.648, 0.964, -0.8 ], [ -1.164, -0.82, 1.852 ], [ -1.084, 0.156, 0.072 ], [ 0.308, 1.672, -1.456 ], [ -0.148, 1.968, 0.36 ], ... ],
    	[ [ -1.564, -0.552, 1.032 ], [ -1.0, -1.896, -0.444 ], [ 1.604, 1.636, 1.496 ], [ -0.372, 0.256, -0.848 ], [ 0.868, 1.592, -0.892 ], [ -1.848, -1.668, -1.604 ], [ 0.692, 0.22, 1.612 ], [ 0.204, 0.476, 0.868 ], ... ],
    	[ [ -1.04, -0.556, 1.268 ], [ -0.148, -1.968, -0.52 ], [ 1.588, -0.812, 0.536 ], [ 1.74, -0.12, 0.124 ], [ 1.48, 1.428, -0.468 ], [ -1.916, 0.104, -0.14 ], [ 1.348, 0.76, 1.592 ], [ -0.712, -1.288, 0.416 ], ... ],
    	[ [ 1.656, 1.94, -0.104 ], [ 1.52, 1.436, 0.668 ], [ 0.412, 0.896, 0.46 ], [ 0.368, -0.692, 0.86 ], [ 1.104, 1.048, -0.436 ], [ 1.752, -1.284, -0.292 ], [ 0.604, -1.144, -1.728 ], [ 1.956, -0.7, 1.24 ], ... ],
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

Code from [PerformanceTester.java:66](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/PerformanceTester.java#L66) executed in 0.20 seconds: 
```java
    test(component, inputPrototype);
```
Logging: 
```
    100 batches
    Input Dimensions:
    	[100, 100, 3]
    Performance:
    	Evaluation performance: 0.018849s +- 0.001500s [0.017336s - 0.021585s]
    	Learning performance: 0.000627s +- 0.000247s [0.000392s - 0.001099s]
    
```

