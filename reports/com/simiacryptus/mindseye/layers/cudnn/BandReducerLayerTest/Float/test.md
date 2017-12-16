# BandReducerLayer
## Float
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
      "id": "1158a0e4-d6c5-4ca0-b071-d9da9c5b009b",
      "isFrozen": false,
      "name": "BandReducerLayer/1158a0e4-d6c5-4ca0-b071-d9da9c5b009b",
      "mode": 0,
      "precision": "Float"
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
    	[ [ -1.208, 0.14 ], [ -1.168, 1.72 ], [ 1.56, -1.584 ] ],
    	[ [ 0.952, 1.892 ], [ -1.86, 1.36 ], [ -1.636, -1.876 ] ],
    	[ [ -1.612, -0.08 ], [ 1.164, 1.004 ], [ 1.096, 0.256 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 0.0, 0.0 ] ]
    ]
    --------------------
    Derivative: 
    [
    	[ [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ] ],
    	[ [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ] ],
    	[ [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ] ]
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
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (200#), relativeTol=0.0000e+00 +- 0.0000e+00 [Infinity - -Infinity] (0#)}
```



Code from [SingleDerivativeTester.java:77](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/SingleDerivativeTester.java#L77) executed in 0.02 seconds: 
```java
    return test(component, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ 0.692, -1.276 ], [ -0.952, -1.128 ], [ 0.756, -1.492 ] ],
    	[ [ -1.116, 0.444 ], [ -1.004, -1.968 ], [ 0.232, -1.44 ] ],
    	[ [ 1.476, -1.168 ], [ -1.128, 1.952 ], [ 0.948, -0.86 ] ]
    ]
    Inputs Statistics: {meanExponent=0.0019152658856245461, negative=11, min=-0.86, max=-0.86, mean=-0.3906666666666667, count=18.0, positive=7, stdDev=1.131641089548959, zeros=0}
    Output: [
    	[ [ 0.0, 0.0 ] ]
    ]
    Outputs Statistics: {meanExponent=NaN, negative=0, min=0.0, max=0.0, mean=0.0, count=2.0, positive=0, stdDev=0.0, zeros=2}
    Feedback for input 0
    Inputs Values: [
    	[ [ 0.692, -1.276 ], [ -0.952, -1.128 ], [ 0.756, -1.492 ] ],
    	[ [ -1.116, 0.444 ], [ -1.004, -1.968 ], [ 0.232, -1.44 ] ],
    	[ [ 1.476, -1.168 ], [ -1.128, 1.952 ], [ 0.948, -0.86 ] ]
    ]
    Value Statistics: {meanExponent=0.0019152658856245461, negative=11, min=-0.86, max=-0.86, mean=-0.3906666666666667, count=18.0, positive=7, stdDev=1.131641089548959, zeros=0}
    Implemented Feedback: [ [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], ... ]
    Implemented Statistics: {meanExponent=NaN, negative=0, min=0.0, max=0.0, mean=0.0, count=36.0, positive=0, stdDev=0.0, zeros=36}
    Measured Feedback: [ [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], ... ]
    Measured Statistics: {meanExponent=NaN, negative=0, min=0.0, max=0.0, mean=0.0, count=36.0, positive=0, stdDev=0.0, zeros=36}
    Feedback Error: [ [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], ... ]
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

Code from [PerformanceTester.java:66](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/PerformanceTester.java#L66) executed in 0.29 seconds: 
```java
    test(component, inputPrototype);
```
Logging: 
```
    100 batches
    Input Dimensions:
    	[100, 100, 3]
    Performance:
    	Evaluation performance: 0.026908s +- 0.005196s [0.021971s - 0.036382s]
    	Learning performance: 0.000762s +- 0.000180s [0.000514s - 0.001065s]
    
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
    	[ [ -1.904, -1.048, 1.624 ], [ 1.808, 0.908, 1.34 ], [ -1.032, 1.136, -0.912 ], [ 1.444, 0.508, 0.296 ], [ -1.704, 1.34, 1.668 ], [ 1.664, -0.644, 1.836 ], [ -1.52, 0.664, 0.908 ], [ -1.376, -1.584, 1.96 ], ... ],
    	[ [ -0.88, -0.896, -0.54 ], [ -0.224, -0.128, -1.436 ], [ -1.812, 0.132, 1.708 ], [ -1.756, 1.736, 1.54 ], [ 0.532, 1.584, -0.272 ], [ 1.256, -1.636, 1.252 ], [ -1.0, 0.812, 0.596 ], [ 1.1, 0.272, 1.068 ], ... ],
    	[ [ -1.992, -1.992, -1.332 ], [ 0.824, -0.704, 0.788 ], [ -0.292, 1.556, -0.772 ], [ 1.876, 1.36, -1.364 ], [ 0.012, -1.032, -0.816 ], [ 1.68, -0.236, -0.968 ], [ 0.06, 0.964, -1.996 ], [ -1.74, -1.044, -1.78 ], ... ],
    	[ [ -1.12, 0.612, 0.436 ], [ 1.128, -0.804, 0.46 ], [ 0.756, -1.112, -1.264 ], [ -0.756, 0.9, -0.288 ], [ -0.836, -1.436, -1.568 ], [ -1.816, 0.076, 0.688 ], [ 0.396, -1.472, -0.568 ], [ -1.656, 0.364, 0.696 ], ... ],
    	[ [ -1.08, 0.7, -0.684 ], [ 1.936, -1.62, 0.708 ], [ 0.276, -0.928, 0.876 ], [ -0.132, -1.6, 0.784 ], [ -1.012, -1.176, -1.772 ], [ -0.416, -1.964, -1.388 ], [ 1.256, -1.168, 0.968 ], [ -1.204, -0.176, 1.944 ], ... ],
    	[ [ -1.54, -0.504, 0.696 ], [ 0.1, 1.34, 0.1 ], [ 0.272, -1.92, -1.996 ], [ 0.164, -1.34, -1.868 ], [ 0.372, 1.6, -1.012 ], [ 1.364, -0.428, 1.668 ], [ 0.972, -0.924, -1.016 ], [ 1.852, 1.56, 1.0 ], ... ],
    	[ [ 0.608, -1.864, 0.808 ], [ 0.228, 0.576, 0.224 ], [ -1.872, -1.584, -0.348 ], [ -1.34, 1.184, -1.06 ], [ 1.72, 1.208, -0.888 ], [ 0.856, 0.004, 0.452 ], [ 1.04, -0.76, 1.32 ], [ 0.636, 0.532, -1.772 ], ... ],
    	[ [ 1.224, 1.452, 1.812 ], [ 0.136, 0.036, 1.004 ], [ 0.42, 0.744, 0.112 ], [ -0.224, -1.356, 0.668 ], [ -1.36, 1.26, -0.428 ], [ 1.1, -0.352, 1.14 ], [ -1.252, -1.512, 1.68 ], [ 1.304, 0.472, 1.408 ], ... ],
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
    	[ [ 1.056, 0.912, -0.28 ], [ 0.272, 0.576, 0.252 ], [ -0.488, 0.328, -1.644 ], [ 0.156, 1.244, 1.428 ], [ -1.312, 1.624, -1.748 ], [ -1.3, 0.36, 1.948 ], [ -0.152, 0.348, 1.932 ], [ 0.54, 1.752, 1.644 ], ... ],
    	[ [ -0.3, -0.936, -1.76 ], [ -1.68, 0.972, -1.524 ], [ -0.44, 0.168, -0.412 ], [ -1.788, -0.212, 1.688 ], [ -1.056, -1.976, -0.912 ], [ 0.168, -1.944, 0.228 ], [ -0.892, 1.824, 0.596 ], [ 0.912, 0.044, 0.984 ], ... ],
    	[ [ 0.4, 1.908, -0.164 ], [ 1.528, -0.196, 1.68 ], [ -0.588, -0.476, 1.232 ], [ 0.252, -1.52, -1.868 ], [ 1.88, 1.304, 1.864 ], [ 1.796, 1.352, 0.716 ], [ -1.008, -0.54, -1.26 ], [ 1.716, 0.744, 1.772 ], ... ],
    	[ [ -0.492, -1.828, -1.076 ], [ -1.536, -1.568, 1.252 ], [ -1.492, 0.892, 0.252 ], [ -0.544, -1.092, -1.896 ], [ -1.188, 1.024, -0.732 ], [ 1.356, 0.272, -1.832 ], [ 0.5, 0.428, 1.896 ], [ 0.824, -1.096, -1.652 ], ... ],
    	[ [ 0.672, 0.552, -0.016 ], [ -0.772, -1.768, -1.788 ], [ 1.512, 1.964, 1.788 ], [ 0.24, 1.74, 0.36 ], [ -1.68, 1.5, -0.344 ], [ 1.5, 0.42, -0.268 ], [ 1.608, 0.308, 0.264 ], [ -0.496, 0.116, 1.884 ], ... ],
    	[ [ 1.664, -1.288, -0.252 ], [ 0.5, 0.852, -1.212 ], [ 1.036, -1.332, -1.136 ], [ 0.244, -0.744, 0.98 ], [ 0.976, 0.528, 0.236 ], [ 0.816, -1.376, 1.344 ], [ 1.108, -0.044, 1.716 ], [ 0.444, -0.516, 0.52 ], ... ],
    	[ [ 1.72, -1.628, 1.376 ], [ -1.16, 0.28, 0.52 ], [ -1.796, -0.84, -0.904 ], [ 1.484, 1.708, 0.164 ], [ 1.7, -0.76, -0.752 ], [ -0.732, 1.76, 1.12 ], [ -0.232, -0.4, 1.792 ], [ -1.38, 0.76, 1.108 ], ... ],
    	[ [ -0.676, 1.652, 1.192 ], [ -0.288, 1.364, -1.628 ], [ 1.008, 0.28, -1.68 ], [ 0.244, 0.016, 1.556 ], [ 1.368, -0.004, -1.688 ], [ 1.148, 0.372, 1.504 ], [ -1.96, -0.164, -1.868 ], [ -1.536, -1.072, -0.92 ], ... ],
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
    	[ [ 1.056, 0.912, -0.28 ], [ 0.272, 0.576, 0.252 ], [ -0.488, 0.328, -1.644 ], [ 0.156, 1.244, 1.428 ], [ -1.312, 1.624, -1.748 ], [ -1.3, 0.36, 1.948 ], [ -0.152, 0.348, 1.932 ], [ 0.54, 1.752, 1.644 ], ... ],
    	[ [ -0.3, -0.936, -1.76 ], [ -1.68, 0.972, -1.524 ], [ -0.44, 0.168, -0.412 ], [ -1.788, -0.212, 1.688 ], [ -1.056, -1.976, -0.912 ], [ 0.168, -1.944, 0.228 ], [ -0.892, 1.824, 0.596 ], [ 0.912, 0.044, 0.984 ], ... ],
    	[ [ 0.4, 1.908, -0.164 ], [ 1.528, -0.196, 1.68 ], [ -0.588, -0.476, 1.232 ], [ 0.252, -1.52, -1.868 ], [ 1.88, 1.304, 1.864 ], [ 1.796, 1.352, 0.716 ], [ -1.008, -0.54, -1.26 ], [ 1.716, 0.744, 1.772 ], ... ],
    	[ [ -0.492, -1.828, -1.076 ], [ -1.536, -1.568, 1.252 ], [ -1.492, 0.892, 0.252 ], [ -0.544, -1.092, -1.896 ], [ -1.188, 1.024, -0.732 ], [ 1.356, 0.272, -1.832 ], [ 0.5, 0.428, 1.896 ], [ 0.824, -1.096, -1.652 ], ... ],
    	[ [ 0.672, 0.552, -0.016 ], [ -0.772, -1.768, -1.788 ], [ 1.512, 1.964, 1.788 ], [ 0.24, 1.74, 0.36 ], [ -1.68, 1.5, -0.344 ], [ 1.5, 0.42, -0.268 ], [ 1.608, 0.308, 0.264 ], [ -0.496, 0.116, 1.884 ], ... ],
    	[ [ 1.664, -1.288, -0.252 ], [ 0.5, 0.852, -1.212 ], [ 1.036, -1.332, -1.136 ], [ 0.244, -0.744, 0.98 ], [ 0.976, 0.528, 0.236 ], [ 0.816, -1.376, 1.344 ], [ 1.108, -0.044, 1.716 ], [ 0.444, -0.516, 0.52 ], ... ],
    	[ [ 1.72, -1.628, 1.376 ], [ -1.16, 0.28, 0.52 ], [ -1.796, -0.84, -0.904 ], [ 1.484, 1.708, 0.164 ], [ 1.7, -0.76, -0.752 ], [ -0.732, 1.76, 1.12 ], [ -0.232, -0.4, 1.792 ], [ -1.38, 0.76, 1.108 ], ... ],
    	[ [ -0.676, 1.652, 1.192 ], [ -0.288, 1.364, -1.628 ], [ 1.008, 0.28, -1.68 ], [ 0.244, 0.016, 1.556 ], [ 1.368, -0.004, -1.688 ], [ 1.148, 0.372, 1.504 ], [ -1.96, -0.164, -1.868 ], [ -1.536, -1.072, -0.92 ], ... ],
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

