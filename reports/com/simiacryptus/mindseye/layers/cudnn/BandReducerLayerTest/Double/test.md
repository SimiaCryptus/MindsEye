# BandReducerLayer
## Double
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
      "id": "ab8a6b1e-0c89-4fda-8c05-3d07ee001d7b",
      "isFrozen": false,
      "name": "BandReducerLayer/ab8a6b1e-0c89-4fda-8c05-3d07ee001d7b",
      "mode": 0,
      "precision": "Double"
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
    	[ [ -0.556, -0.776 ], [ 0.756, -0.164 ], [ 1.572, 0.048 ] ],
    	[ [ 0.272, -1.68 ], [ -1.936, 1.812 ], [ 1.632, -0.012 ] ],
    	[ [ -0.348, -1.0 ], [ 0.996, 1.852 ], [ 0.752, -1.804 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 1.632, 1.852 ] ]
    ]
    --------------------
    Derivative: 
    [
    	[ [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ] ],
    	[ [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 1.0, 0.0 ] ],
    	[ [ 0.0, 0.0 ], [ 0.0, 1.0 ], [ 0.0, 0.0 ] ]
    ]
```



[GPU Log](etc/cuda.log)

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
    	[ [ 1.364, -0.276 ], [ -1.272, 1.36 ], [ 0.312, 0.808 ] ],
    	[ [ -0.644, -1.128 ], [ 1.34, -0.592 ], [ 0.82, -1.648 ] ],
    	[ [ 1.22, -1.352 ], [ -1.364, 0.776 ], [ 0.336, -1.248 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.05715142026131893, negative=9, min=-1.248, max=-1.248, mean=-0.066, count=18.0, positive=9, stdDev=1.0735952061492575, zeros=0}
    Output: [
    	[ [ 1.364, 1.36 ] ]
    ]
    Outputs Statistics: {meanExponent=0.13417663934533886, negative=0, min=1.36, max=1.36, mean=1.362, count=2.0, positive=2, stdDev=0.0020000000000287557, zeros=0}
    Feedback for input 0
    Inputs Values: [
    	[ [ 1.364, -0.276 ], [ -1.272, 1.36 ], [ 0.312, 0.808 ] ],
    	[ [ -0.644, -1.128 ], [ 1.34, -0.592 ], [ 0.82, -1.648 ] ],
    	[ [ 1.22, -1.352 ], [ -1.364, 0.776 ], [ 0.336, -1.248 ] ]
    ]
    Value Statistics: {meanExponent=-0.05715142026131893, negative=9, min=-1.248, max=-1.248, mean=-0.066, count=18.0, positive=9, stdDev=1.0735952061492575, zeros=0}
    Implemented Feedback: [ [ 1.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 
```
...[skipping 48 bytes](etc/46.txt)...
```
    .0 ], ... ]
    Implemented Statistics: {meanExponent=0.0, negative=0, min=0.0, max=0.0, mean=0.05555555555555555, count=36.0, positive=2, stdDev=0.2290614236454256, zeros=34}
    Measured Feedback: [ [ 0.9999999999998899, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], ... ]
    Measured Statistics: {meanExponent=-4.7830642341045674E-14, negative=0, min=0.0, max=0.0, mean=0.05555555555554944, count=36.0, positive=2, stdDev=0.22906142364540036, zeros=34}
    Feedback Error: [ [ -1.1013412404281553E-13, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], ... ]
    Error Statistics: {meanExponent=-12.958078098036825, negative=2, min=0.0, max=0.0, mean=-6.118562446823085E-15, count=36.0, positive=0, stdDev=2.5227479245189218E-14, zeros=34}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 6.1186e-15 +- 2.5227e-14 [0.0000e+00 - 1.1013e-13] (36#)
    relativeTol: 5.5067e-14 +- 0.0000e+00 [5.5067e-14 - 5.5067e-14] (2#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=6.1186e-15 +- 2.5227e-14 [0.0000e+00 - 1.1013e-13] (36#), relativeTol=5.5067e-14 +- 0.0000e+00 [5.5067e-14 - 5.5067e-14] (2#)}
```



### Performance
Now we execute larger-scale runs to benchmark performance:

Code from [PerformanceTester.java:66](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/PerformanceTester.java#L66) executed in 0.19 seconds: 
```java
    test(component, inputPrototype);
```
Logging: 
```
    100 batches
    Input Dimensions:
    	[100, 100, 3]
    Performance:
    	Evaluation performance: 0.019142s +- 0.001051s [0.018008s - 0.020870s]
    	Learning performance: 0.000621s +- 0.000112s [0.000500s - 0.000833s]
    
```

### Input Learning
In this test, we use a network to learn this target input, given it's pre-evaluated output:

Code from [LearningTester.java:127](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L127) executed in 0.01 seconds: 
```java
    return Arrays.stream(input_target).map(x -> x.prettyPrint()).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [
    	[ [ 1.172, -1.884, 0.424 ], [ 1.188, 0.808, -1.58 ], [ 1.376, -0.096, -1.244 ], [ -1.54, -0.528, -1.068 ], [ 1.06, 0.832, 0.872 ], [ 1.288, 0.016, 0.764 ], [ -1.676, -0.536, 0.632 ], [ -1.644, -1.54, -0.896 ], ... ],
    	[ [ -0.124, -0.516, -1.288 ], [ 1.932, -0.112, -1.544 ], [ 0.068, -1.64, 1.508 ], [ -0.02, 0.752, 0.952 ], [ 1.868, 0.928, 1.348 ], [ 1.824, -0.832, -0.78 ], [ -1.596, -0.124, -0.424 ], [ -1.756, 1.376, -0.164 ], ... ],
    	[ [ -1.588, -1.696, -1.992 ], [ 0.356, 1.388, 1.164 ], [ -1.628, 0.036, 0.78 ], [ -0.372, 1.788, -0.928 ], [ -1.112, -0.216, -1.292 ], [ -1.852, 0.14, 1.856 ], [ 1.18, 1.676, 0.432 ], [ 0.66, -1.524, -0.3 ], ... ],
    	[ [ 1.736, -0.44, -0.16 ], [ -0.724, -1.156, 0.9 ], [ -0.164, -0.28, -1.808 ], [ -0.852, 0.012, -0.352 ], [ -1.932, -0.42, 0.112 ], [ -0.908, -0.78, 1.872 ], [ -0.124, 1.004, -1.896 ], [ 1.572, 0.992, -0.944 ], ... ],
    	[ [ 0.204, -0.312, 0.968 ], [ 1.236, 0.256, 1.412 ], [ 0.792, 0.564, 0.112 ], [ -0.952, -0.38, 0.0 ], [ 0.544, 0.224, 0.072 ], [ 1.788, -1.6, 0.468 ], [ -0.212, 0.632, 1.932 ], [ -0.928, -1.564, -0.416 ], ... ],
    	[ [ -1.532, 1.82, 1.452 ], [ -1.9, -1.816, 0.008 ], [ -1.052, 0.352, -0.544 ], [ -1.496, -1.784, -1.132 ], [ -1.32, -0.152, -1.592 ], [ -1.288, 1.708, -1.668 ], [ 1.788, 1.516, -0.22 ], [ 0.336, 0.512, 0.944 ], ... ],
    	[ [ -0.844, -0.192, 0.384 ], [ -0.052, -0.872, 1.648 ], [ 0.976, 1.036, 0.104 ], [ 1.508, 0.732, 0.772 ], [ -0.628, -1.76, -1.36 ], [ 1.744, -0.592, -0.272 ], [ 0.068, -1.74, -0.868 ], [ 1.512, 0.74, -1.856 ], ... ],
    	[ [ -1.056, -0.852, 1.816 ], [ 0.292, -0.704, 0.508 ], [ 1.972, 0.976, -0.468 ], [ 0.548, 0.048, 0.968 ], [ 0.852, -0.524, -1.02 ], [ -1.08, -1.972, -0.736 ], [ -0.388, -0.924, 1.724 ], [ 1.848, -0.532, 0.832 ], ... ],
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

Code from [LearningTester.java:144](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L144) executed in 0.01 seconds: 
```java
    return Arrays.stream(input_gd).map(x -> x.prettyPrint()).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [
    	[ [ 0.872, 0.536, -1.66 ], [ 0.356, -0.912, -0.468 ], [ 1.764, -1.696, 1.196 ], [ 0.06, -1.336, 0.184 ], [ 1.732, 0.12, 1.752 ], [ 0.904, -1.56, -1.22 ], [ -1.4, -1.404, 0.036 ], [ -0.856, 0.532, 0.556 ], ... ],
    	[ [ 1.068, -1.984, 0.012 ], [ 1.56, -0.476, 1.996 ], [ -1.74, -0.812, -1.42 ], [ 0.116, -1.5, 1.776 ], [ 1.464, 0.916, -1.54 ], [ -0.684, -1.112, 0.016 ], [ -0.528, 1.6, -1.392 ], [ -1.44, -0.392, -0.288 ], ... ],
    	[ [ 0.992, 0.992, 0.672 ], [ -0.824, -1.976, -0.084 ], [ -1.324, 0.648, -0.412 ], [ 1.808, -1.796, -1.844 ], [ 0.068, -1.828, 0.064 ], [ 1.788, -0.164, 1.364 ], [ -1.772, 1.688, 0.872 ], [ 0.556, 0.684, 1.912 ], ... ],
    	[ [ -0.308, 0.404, 1.176 ], [ -1.764, -1.18, 0.68 ], [ -0.872, 1.968, 0.828 ], [ -0.612, -0.612, -0.628 ], [ 1.456, -0.864, 0.768 ], [ 1.66, 1.316, -1.596 ], [ -0.536, 0.612, -1.68 ], [ -1.104, 1.1, 0.66 ], ... ],
    	[ [ -0.496, -1.868, -1.088 ], [ -0.344, -0.296, 1.632 ], [ 0.68, 0.712, -1.312 ], [ 0.9, 0.6, -0.248 ], [ -1.064, -1.568, -0.928 ], [ -1.488, 1.236, -0.22 ], [ 0.952, 1.14, -0.152 ], [ 1.944, 0.292, 0.196 ], ... ],
    	[ [ -0.884, 1.144, 1.604 ], [ 0.64, 1.6, -0.096 ], [ 0.884, -0.624, 1.144 ], [ 0.74, -0.356, 1.9 ], [ -1.76, -1.456, 0.304 ], [ -0.744, 0.096, -1.384 ], [ -1.652, -1.88, -1.072 ], [ 0.148, 1.212, -0.496 ], ... ],
    	[ [ 1.584, 0.996, -1.076 ], [ 1.656, 0.024, 0.98 ], [ 0.228, 0.992, -0.428 ], [ -0.2, 0.42, 1.336 ], [ 1.292, 0.996, 1.656 ], [ 1.96, 0.296, 1.248 ], [ -1.86, 0.736, -0.62 ], [ -1.696, -0.46, -0.788 ], ... ],
    	[ [ -0.304, 1.848, 1.296 ], [ 0.256, 1.304, 0.952 ], [ 0.536, 1.144, -1.84 ], [ 0.524, 0.16, -1.664 ], [ 0.184, -0.316, -0.26 ], [ -1.2, -0.572, -0.72 ], [ -0.064, -1.02, 1.308 ], [ 0.668, -0.956, 0.2 ], ... ],
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

Code from [LearningTester.java:154](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L154) executed in 0.01 seconds: 
```java
    return Arrays.stream(input_lbgfs).map(x -> x.prettyPrint()).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [
    	[ [ 0.872, 0.536, -1.66 ], [ 0.356, -0.912, -0.468 ], [ 1.764, -1.696, 1.196 ], [ 0.06, -1.336, 0.184 ], [ 1.732, 0.12, 1.752 ], [ 0.904, -1.56, -1.22 ], [ -1.4, -1.404, 0.036 ], [ -0.856, 0.532, 0.556 ], ... ],
    	[ [ 1.068, -1.984, 0.012 ], [ 1.56, -0.476, 1.996 ], [ -1.74, -0.812, -1.42 ], [ 0.116, -1.5, 1.776 ], [ 1.464, 0.916, -1.54 ], [ -0.684, -1.112, 0.016 ], [ -0.528, 1.6, -1.392 ], [ -1.44, -0.392, -0.288 ], ... ],
    	[ [ 0.992, 0.992, 0.672 ], [ -0.824, -1.976, -0.084 ], [ -1.324, 0.648, -0.412 ], [ 1.808, -1.796, -1.844 ], [ 0.068, -1.828, 0.064 ], [ 1.788, -0.164, 1.364 ], [ -1.772, 1.688, 0.872 ], [ 0.556, 0.684, 1.912 ], ... ],
    	[ [ -0.308, 0.404, 1.176 ], [ -1.764, -1.18, 0.68 ], [ -0.872, 1.968, 0.828 ], [ -0.612, -0.612, -0.628 ], [ 1.456, -0.864, 0.768 ], [ 1.66, 1.316, -1.596 ], [ -0.536, 0.612, -1.68 ], [ -1.104, 1.1, 0.66 ], ... ],
    	[ [ -0.496, -1.868, -1.088 ], [ -0.344, -0.296, 1.632 ], [ 0.68, 0.712, -1.312 ], [ 0.9, 0.6, -0.248 ], [ -1.064, -1.568, -0.928 ], [ -1.488, 1.236, -0.22 ], [ 0.952, 1.14, -0.152 ], [ 1.944, 0.292, 0.196 ], ... ],
    	[ [ -0.884, 1.144, 1.604 ], [ 0.64, 1.6, -0.096 ], [ 0.884, -0.624, 1.144 ], [ 0.74, -0.356, 1.9 ], [ -1.76, -1.456, 0.304 ], [ -0.744, 0.096, -1.384 ], [ -1.652, -1.88, -1.072 ], [ 0.148, 1.212, -0.496 ], ... ],
    	[ [ 1.584, 0.996, -1.076 ], [ 1.656, 0.024, 0.98 ], [ 0.228, 0.992, -0.428 ], [ -0.2, 0.42, 1.336 ], [ 1.292, 0.996, 1.656 ], [ 1.96, 0.296, 1.248 ], [ -1.86, 0.736, -0.62 ], [ -1.696, -0.46, -0.788 ], ... ],
    	[ [ -0.304, 1.848, 1.296 ], [ 0.256, 1.304, 0.952 ], [ 0.536, 1.144, -1.84 ], [ 0.524, 0.16, -1.664 ], [ 0.184, -0.316, -0.26 ], [ -1.2, -0.572, -0.72 ], [ -0.064, -1.02, 1.308 ], [ 0.668, -0.956, 0.2 ], ... ],
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

