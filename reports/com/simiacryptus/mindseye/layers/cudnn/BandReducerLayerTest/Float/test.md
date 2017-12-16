# BandReducerLayer
## Float
### Batch Execution
Code from [BatchingTester.java:66](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/BatchingTester.java#L66) executed in 0.01 seconds: 
```java
    return test(reference, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (200#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (40#)}
```



Code from [SingleDerivativeTester.java:77](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/SingleDerivativeTester.java#L77) executed in 0.02 seconds: 
```java
    return test(component, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ -1.34, -1.72 ], [ -1.476, -1.952 ], [ -1.596, -1.632 ] ],
    	[ [ -0.636, -0.508 ], [ -1.516, -1.728 ], [ 0.636, -1.172 ] ],
    	[ [ -0.032, -0.44 ], [ -1.808, -1.54 ], [ 0.272, 0.18 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.09327421315868117, negative=15, min=0.18, max=0.18, mean=-1.0004444444444447, count=18.0, positive=3, stdDev=0.8029068177719572, zeros=0}
    Output: [
    	[ [ 0.636, 0.18 ] ]
    ]
    Outputs Statistics: {meanExponent=-0.4706351896241401, negative=0, min=0.18, max=0.18, mean=0.40800000000000003, count=2.0, positive=2, stdDev=0.22799999999999995, zeros=0}
    Feedback for input 0
    Inputs Values: [
    	[ [ -1.34, -1.72 ], [ -1.476, -1.952 ], [ -1.596, -1.632 ] ],
    	[ [ -0.636, -0.508 ], [ -1.516, -1.728 ], [ 0.636, -1.172 ] ],
    	[ [ -0.032, -0.44 ], [ -1.808, -1.54 ], [ 0.272, 0.18 ] ]
    ]
    Value Statistics: {meanExponent=-0.09327421315868117, negative=15, min=0.18, max=0.18, mean=-1.0004444444444447, count=18.0, positive=3, stdDev=0.8029068177719572, zeros=0}
    Implemented Feedback: [ [ 0.0, 0.0 ], [ 
```
...[skipping 149 bytes](etc/30.txt)...
```
    ative=0, min=1.0, max=1.0, mean=0.05555555555555555, count=36.0, positive=2, stdDev=0.2290614236454256, zeros=34}
    Measured Feedback: [ [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 1.0000000000000009, 0.0 ], ... ]
    Measured Statistics: {meanExponent=3.857309866213147E-16, negative=0, min=1.0000000000000009, max=1.0000000000000009, mean=0.05555555555555561, count=36.0, positive=2, stdDev=0.22906142364542578, zeros=34}
    Feedback Error: [ [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 8.881784197001252E-16, 0.0 ], ... ]
    Error Statistics: {meanExponent=-15.05149978319906, negative=0, min=8.881784197001252E-16, max=8.881784197001252E-16, mean=4.9343245538895844E-17, count=36.0, positive=2, stdDev=2.03447413267655E-16, zeros=34}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 4.9343e-17 +- 2.0345e-16 [0.0000e+00 - 8.8818e-16] (36#)
    relativeTol: 4.4409e-16 +- 0.0000e+00 [4.4409e-16 - 4.4409e-16] (2#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=4.9343e-17 +- 2.0345e-16 [0.0000e+00 - 8.8818e-16] (36#), relativeTol=4.4409e-16 +- 0.0000e+00 [4.4409e-16 - 4.4409e-16] (2#)}
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
      "id": "55269726-01cf-44d8-90cf-a762ecfe78fa",
      "isFrozen": false,
      "name": "BandReducerLayer/55269726-01cf-44d8-90cf-a762ecfe78fa",
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
    	[ [ 1.876, -1.924 ], [ -1.112, 1.908 ], [ 1.348, 1.964 ] ],
    	[ [ -1.128, 1.12 ], [ 0.872, 0.472 ], [ 0.176, 1.66 ] ],
    	[ [ -0.22, 0.18 ], [ 0.236, -0.304 ], [ -1.356, 0.296 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 1.876, 1.964 ] ]
    ]
    --------------------
    Derivative: 
    [
    	[ [ 1.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 1.0 ] ],
    	[ [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ] ],
    	[ [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ] ]
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
    	[ [ 1.608, -1.096, -0.716 ], [ 0.86, -0.896, -0.152 ], [ -0.172, -0.716, 0.272 ], [ 1.744, -1.564, -0.644 ], [ -0.656, -0.512, -1.436 ], [ 1.272, 1.236, -1.992 ], [ -1.184, -1.132, -1.588 ], [ 1.228, -1.804, -0.04 ], ... ],
    	[ [ 0.592, 0.468, 1.568 ], [ -0.776, -1.328, 1.436 ], [ -1.656, -1.336, 0.34 ], [ 0.04, -0.86, -0.304 ], [ -1.684, -1.824, -0.236 ], [ 0.908, -1.296, 1.416 ], [ -0.264, 1.948, -0.668 ], [ 1.228, 0.688, -0.72 ], ... ],
    	[ [ 0.276, -1.156, 0.44 ], [ 1.756, -0.276, -1.564 ], [ -0.684, 1.456, 0.352 ], [ 0.336, 1.96, 1.56 ], [ -1.116, 0.876, 0.54 ], [ -1.48, 1.98, 0.164 ], [ 1.044, -1.608, -0.164 ], [ -0.824, 1.12, 0.768 ], ... ],
    	[ [ 0.432, 0.732, 1.076 ], [ 0.844, -0.416, -1.032 ], [ -0.452, 1.012, 1.62 ], [ 1.224, 1.916, -0.716 ], [ -1.94, 0.904, 1.104 ], [ 1.132, 0.396, -0.592 ], [ 0.72, -1.452, -1.74 ], [ -0.668, -1.64, -1.088 ], ... ],
    	[ [ 1.184, 0.068, 0.452 ], [ -0.716, 0.796, -0.908 ], [ 0.7, 1.14, 1.336 ], [ -1.712, 0.684, -0.016 ], [ -1.22, 1.476, -0.344 ], [ 1.924, 0.172, -1.108 ], [ -1.132, 0.472, 0.0 ], [ -1.176, -1.472, -0.52 ], ... ],
    	[ [ -1.416, -0.88, -1.108 ], [ -1.436, -0.364, -1.212 ], [ 0.428, 1.424, 0.096 ], [ 0.392, -1.676, -0.288 ], [ 0.952, -0.44, 0.232 ], [ -0.252, 0.088, -0.448 ], [ 0.668, 1.712, 1.404 ], [ 0.128, -0.408, 1.124 ], ... ],
    	[ [ -1.296, -1.228, 0.08 ], [ 1.444, -0.712, 1.38 ], [ 0.16, -0.908, 1.948 ], [ -1.428, -1.416, -0.9 ], [ 1.344, -0.444, -0.156 ], [ -1.86, -0.264, 1.588 ], [ -0.852, 1.08, 1.944 ], [ -1.872, -0.324, 0.2 ], ... ],
    	[ [ 1.436, 1.92, 0.54 ], [ -0.116, -1.36, -1.56 ], [ -1.956, 0.676, 1.312 ], [ -1.46, -0.048, 1.72 ], [ 0.02, 1.1, 0.328 ], [ -1.888, -0.936, -1.584 ], [ -0.352, 1.9, -0.78 ], [ 0.94, 1.964, -0.116 ], ... ],
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
    	[ [ -0.812, 0.908, -0.328 ], [ -1.52, -0.924, -0.828 ], [ -1.94, -0.868, 1.508 ], [ -1.308, -0.548, -1.228 ], [ 1.108, 0.56, -1.212 ], [ 0.888, -0.916, 0.572 ], [ 1.268, -1.48, 1.436 ], [ 1.868, -1.376, 1.164 ], ... ],
    	[ [ 0.468, 0.28, -0.716 ], [ 0.712, 0.576, -1.884 ], [ -0.988, -1.652, 1.24 ], [ -1.86, 1.692, -0.784 ], [ 1.16, -0.952, -1.5 ], [ 0.092, -1.14, 0.5 ], [ -0.888, -0.24, -0.048 ], [ -1.108, -1.54, -1.832 ], ... ],
    	[ [ -1.016, 1.816, 0.732 ], [ -1.344, -0.856, 0.708 ], [ -0.056, -0.448, -0.36 ], [ -0.744, -1.524, -1.196 ], [ -0.32, 1.604, 0.756 ], [ 0.504, -1.688, -0.204 ], [ -1.876, 1.896, -1.856 ], [ 1.58, -0.764, -0.532 ], ... ],
    	[ [ 1.488, 0.144, -1.516 ], [ 0.82, -0.936, 0.084 ], [ -0.68, 0.592, -0.892 ], [ -0.44, -0.608, -1.868 ], [ 0.08, -0.144, -0.348 ], [ 1.248, 0.172, 1.784 ], [ -0.844, 1.424, -1.28 ], [ 0.616, -0.86, 0.024 ], ... ],
    	[ [ -1.232, -1.084, 0.236 ], [ -0.204, -1.24, 1.684 ], [ -1.248, 1.796, 0.228 ], [ -1.496, -1.432, -1.756 ], [ 0.156, -0.74, -0.236 ], [ 0.348, -1.484, -1.84 ], [ -1.288, 1.664, 0.816 ], [ 0.956, -1.128, 0.024 ], ... ],
    	[ [ -0.324, 0.364, 1.448 ], [ 0.996, 1.916, 0.712 ], [ 0.348, -0.128, 0.316 ], [ -0.872, -0.84, 1.02 ], [ 1.184, -0.328, 1.128 ], [ 1.068, -0.944, 1.432 ], [ 0.58, -0.72, -0.076 ], [ 0.56, 0.288, 0.98 ], ... ],
    	[ [ 1.876, -0.46, 1.016 ], [ 0.848, -0.18, 1.072 ], [ -1.472, 0.896, 0.44 ], [ -1.82, 1.336, 0.468 ], [ -1.536, 0.732, -0.784 ], [ 1.836, -1.56, -1.104 ], [ 1.364, -0.508, -1.336 ], [ -1.432, -0.024, -1.504 ], ... ],
    	[ [ -1.372, -1.06, 1.168 ], [ -0.24, -1.58, 1.536 ], [ 0.74, -0.368, -1.148 ], [ -1.132, -0.628, 0.372 ], [ 0.044, 0.544, -1.38 ], [ 1.828, 0.652, 1.212 ], [ -1.044, 0.96, 1.912 ], [ 1.096, 1.896, 0.68 ], ... ],
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

Code from [LearningTester.java:154](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L154) executed in 0.02 seconds: 
```java
    return Arrays.stream(input_lbgfs).map(x -> x.prettyPrint()).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [
    	[ [ -0.812, 0.908, -0.328 ], [ -1.52, -0.924, -0.828 ], [ -1.94, -0.868, 1.508 ], [ -1.308, -0.548, -1.228 ], [ 1.108, 0.56, -1.212 ], [ 0.888, -0.916, 0.572 ], [ 1.268, -1.48, 1.436 ], [ 1.868, -1.376, 1.164 ], ... ],
    	[ [ 0.468, 0.28, -0.716 ], [ 0.712, 0.576, -1.884 ], [ -0.988, -1.652, 1.24 ], [ -1.86, 1.692, -0.784 ], [ 1.16, -0.952, -1.5 ], [ 0.092, -1.14, 0.5 ], [ -0.888, -0.24, -0.048 ], [ -1.108, -1.54, -1.832 ], ... ],
    	[ [ -1.016, 1.816, 0.732 ], [ -1.344, -0.856, 0.708 ], [ -0.056, -0.448, -0.36 ], [ -0.744, -1.524, -1.196 ], [ -0.32, 1.604, 0.756 ], [ 0.504, -1.688, -0.204 ], [ -1.876, 1.896, -1.856 ], [ 1.58, -0.764, -0.532 ], ... ],
    	[ [ 1.488, 0.144, -1.516 ], [ 0.82, -0.936, 0.084 ], [ -0.68, 0.592, -0.892 ], [ -0.44, -0.608, -1.868 ], [ 0.08, -0.144, -0.348 ], [ 1.248, 0.172, 1.784 ], [ -0.844, 1.424, -1.28 ], [ 0.616, -0.86, 0.024 ], ... ],
    	[ [ -1.232, -1.084, 0.236 ], [ -0.204, -1.24, 1.684 ], [ -1.248, 1.796, 0.228 ], [ -1.496, -1.432, -1.756 ], [ 0.156, -0.74, -0.236 ], [ 0.348, -1.484, -1.84 ], [ -1.288, 1.664, 0.816 ], [ 0.956, -1.128, 0.024 ], ... ],
    	[ [ -0.324, 0.364, 1.448 ], [ 0.996, 1.916, 0.712 ], [ 0.348, -0.128, 0.316 ], [ -0.872, -0.84, 1.02 ], [ 1.184, -0.328, 1.128 ], [ 1.068, -0.944, 1.432 ], [ 0.58, -0.72, -0.076 ], [ 0.56, 0.288, 0.98 ], ... ],
    	[ [ 1.876, -0.46, 1.016 ], [ 0.848, -0.18, 1.072 ], [ -1.472, 0.896, 0.44 ], [ -1.82, 1.336, 0.468 ], [ -1.536, 0.732, -0.784 ], [ 1.836, -1.56, -1.104 ], [ 1.364, -0.508, -1.336 ], [ -1.432, -0.024, -1.504 ], ... ],
    	[ [ -1.372, -1.06, 1.168 ], [ -0.24, -1.58, 1.536 ], [ 0.74, -0.368, -1.148 ], [ -1.132, -0.628, 0.372 ], [ 0.044, 0.544, -1.38 ], [ 1.828, 0.652, 1.212 ], [ -1.044, 0.96, 1.912 ], [ 1.096, 1.896, 0.68 ], ... ],
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

Code from [PerformanceTester.java:66](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/PerformanceTester.java#L66) executed in 0.35 seconds: 
```java
    test(component, inputPrototype);
```
Logging: 
```
    100 batches
    Input Dimensions:
    	[100, 100, 3]
    Performance:
    	Evaluation performance: 0.029661s +- 0.006907s [0.021086s - 0.042242s]
    	Learning performance: 0.003661s +- 0.004824s [0.000627s - 0.013248s]
    
```

