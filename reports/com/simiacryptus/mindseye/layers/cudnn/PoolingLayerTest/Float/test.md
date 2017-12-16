# PoolingLayer
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
      "class": "com.simiacryptus.mindseye.layers.cudnn.PoolingLayer",
      "id": "a861c123-9a46-4a5d-a942-ae03f7d3bdbe",
      "isFrozen": false,
      "name": "PoolingLayer/a861c123-9a46-4a5d-a942-ae03f7d3bdbe",
      "mode": 0,
      "windowX": 2,
      "windowY": 2,
      "paddingX": 0,
      "paddingY": 0,
      "strideX": 2,
      "strideY": 2,
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
    	[ [ -0.564, 0.984 ], [ -0.136, 0.384 ], [ -0.316, -0.768 ], [ 1.884, -1.996 ] ],
    	[ [ 1.408, 1.196 ], [ 1.604, -0.756 ], [ 1.068, -1.72 ], [ -1.124, 0.776 ] ],
    	[ [ 0.716, 0.748 ], [ 0.544, 1.02 ], [ -0.072, -0.756 ], [ 1.064, 0.696 ] ],
    	[ [ 1.976, 0.548 ], [ -1.224, -1.4 ], [ -1.52, -1.568 ], [ -0.952, -1.864 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 0.0, 0.0 ], [ 0.0, 0.0 ] ],
    	[ [ 0.0, 0.0 ], [ 0.0, 0.0 ] ]
    ]
    --------------------
    Derivative: 
    [
    	[ [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ] ],
    	[ [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ] ],
    	[ [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ] ],
    	[ [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ] ]
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
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (400#), relativeTol=0.0000e+00 +- 0.0000e+00 [Infinity - -Infinity] (0#)}
```



Code from [SingleDerivativeTester.java:77](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/SingleDerivativeTester.java#L77) executed in 0.02 seconds: 
```java
    return test(component, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ 0.288, -1.448 ], [ -1.472, -1.664 ], [ -0.328, 1.872 ], [ -0.064, -0.548 ] ],
    	[ [ -0.712, 1.4 ], [ -1.092, -1.212 ], [ 0.932, 1.22 ], [ -0.008, -1.532 ] ],
    	[ [ -0.476, -0.352 ], [ -0.304, 1.864 ], [ -0.384, 0.228 ], [ 0.004, 1.744 ] ],
    	[ [ -0.136, -1.656 ], [ 0.792, -1.676 ], [ -1.292, 1.612 ], [ 1.836, -0.024 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.28723525804638933, negative=20, min=-0.024, max=-0.024, mean=-0.080875, count=32.0, positive=12, stdDev=1.1408548261610676, zeros=0}
    Output: [
    	[ [ 0.0, 0.0 ], [ 0.0, 0.0 ] ],
    	[ [ 0.0, 0.0 ], [ 0.0, 0.0 ] ]
    ]
    Outputs Statistics: {meanExponent=NaN, negative=0, min=0.0, max=0.0, mean=0.0, count=8.0, positive=0, stdDev=0.0, zeros=8}
    Feedback for input 0
    Inputs Values: [
    	[ [ 0.288, -1.448 ], [ -1.472, -1.664 ], [ -0.328, 1.872 ], [ -0.064, -0.548 ] ],
    	[ [ -0.712, 1.4 ], [ -1.092, -1.212 ], [ 0.932, 1.22 ], [ -0.008, -1.532 ] ],
    	[ [ -0.476, -0.352 ], [ -0.304, 1.864 ], [ -0.384, 0.228 ], [ 0.004, 1.744 ] ],
    	[ [ -0.136, -1.656 ], [ 0.792, -
```
...[skipping 902 bytes](etc/158.txt)...
```
    0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], ... ]
    Measured Statistics: {meanExponent=NaN, negative=0, min=0.0, max=0.0, mean=0.0, count=256.0, positive=0, stdDev=0.0, zeros=256}
    Feedback Error: [ [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], ... ]
    Error Statistics: {meanExponent=NaN, negative=0, min=0.0, max=0.0, mean=0.0, count=256.0, positive=0, stdDev=0.0, zeros=256}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (256#)
    relativeTol: 0.0000e+00 +- 0.0000e+00 [Infinity - -Infinity] (0#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (256#), relativeTol=0.0000e+00 +- 0.0000e+00 [Infinity - -Infinity] (0#)}
```



### Performance
Now we execute larger-scale runs to benchmark performance:

Code from [PerformanceTester.java:66](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/PerformanceTester.java#L66) executed in 0.21 seconds: 
```java
    test(component, inputPrototype);
```
Logging: 
```
    100 batches
    Input Dimensions:
    	[100, 100, 2]
    Performance:
    	Evaluation performance: 0.012043s +- 0.001610s [0.010430s - 0.015142s]
    	Learning performance: 0.013143s +- 0.001777s [0.011906s - 0.016587s]
    
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
    	[ [ 0.276, 0.936 ], [ 0.036, 0.176 ], [ -0.476, -0.452 ], [ -0.436, -1.432 ], [ 0.228, -0.788 ], [ 1.076, 0.528 ], [ 0.436, -1.008 ], [ -1.744, -1.452 ], ... ],
    	[ [ 0.28, 0.18 ], [ 1.344, 0.792 ], [ 1.764, 0.384 ], [ 1.08, -1.76 ], [ -1.616, 0.62 ], [ -1.956, -1.188 ], [ 0.604, -1.584 ], [ -0.308, -0.02 ], ... ],
    	[ [ -1.008, 1.088 ], [ -0.2, -0.512 ], [ 1.62, 0.528 ], [ -0.52, 1.288 ], [ 1.06, 1.08 ], [ -0.632, -0.24 ], [ -1.208, -1.104 ], [ -0.112, 1.064 ], ... ],
    	[ [ -0.944, -1.588 ], [ 1.34, 0.608 ], [ 1.188, 1.08 ], [ 1.86, 1.228 ], [ -1.704, -0.732 ], [ -1.18, 1.276 ], [ -0.356, 0.048 ], [ -0.088, 0.9 ], ... ],
    	[ [ -0.944, -1.032 ], [ -1.784, -0.284 ], [ 0.44, -1.0 ], [ 0.972, 0.616 ], [ 0.868, 0.716 ], [ 1.488, -0.784 ], [ 1.58, 1.592 ], [ 0.428, 1.444 ], ... ],
    	[ [ -1.532, 0.38 ], [ 1.54, -0.948 ], [ -1.876, -0.576 ], [ 1.62, -1.596 ], [ -1.268, -1.324 ], [ 1.352, 0.092 ], [ -0.072, 0.4 ], [ 1.052, 1.156 ], ... ],
    	[ [ -1.184, -0.476 ], [ 1.572, 1.804 ], [ -1.276, 0.248 ], [ 1.636, -1.9 ], [ -1.772, -0.436 ], [ 0.624, 0.96 ], [ 1.728, 1.456 ], [ -0.016, 0.904 ], ... ],
    	[ [ 1.876, 1.048 ], [ -0.772, -1.784 ], [ -1.732, 0.892 ], [ 1.196, -1.032 ], [ -0.2, 0.464 ], [ 0.4, 0.656 ], [ -0.76, 0.884 ], [ -1.416, -1.964 ], ... ],
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
    	[ [ -0.936, 0.32 ], [ -0.88, -0.116 ], [ 0.304, -1.472 ], [ 1.908, 0.724 ], [ 0.896, 1.3 ], [ 1.256, -1.184 ], [ 0.116, -0.2 ], [ 1.92, -0.356 ], ... ],
    	[ [ 1.656, -1.304 ], [ 0.816, -1.552 ], [ -1.004, -1.112 ], [ 0.824, 1.192 ], [ 0.436, 0.952 ], [ -0.288, 1.388 ], [ -0.6, -1.724 ], [ 1.64, 0.124 ], ... ],
    	[ [ -0.676, 1.992 ], [ 0.416, -0.452 ], [ 1.82, -1.78 ], [ 0.176, -0.932 ], [ -0.08, -0.956 ], [ -0.232, -1.856 ], [ -1.448, -0.528 ], [ 0.848, 1.732 ], ... ],
    	[ [ 1.392, 1.036 ], [ -1.24, 1.828 ], [ 0.012, -1.212 ], [ -0.2, -1.908 ], [ -0.252, 1.768 ], [ -1.82, -0.908 ], [ -1.068, 1.992 ], [ -1.004, 1.608 ], ... ],
    	[ [ -0.228, 0.244 ], [ 0.948, -1.236 ], [ -1.496, -0.44 ], [ 1.128, -1.388 ], [ -0.82, 1.248 ], [ -0.016, 0.696 ], [ 0.648, -1.348 ], [ 1.688, -1.564 ], ... ],
    	[ [ -1.208, -0.228 ], [ -1.68, 0.168 ], [ 0.316, -0.268 ], [ -0.26, -0.616 ], [ -0.972, 1.392 ], [ 0.428, -0.576 ], [ -1.144, 0.136 ], [ 1.768, -1.9 ], ... ],
    	[ [ -1.464, -0.472 ], [ -0.868, 0.644 ], [ -0.388, 0.964 ], [ 0.696, 1.2 ], [ 1.992, 1.136 ], [ -1.52, -0.26 ], [ -0.844, -0.484 ], [ -0.444, 1.988 ], ... ],
    	[ [ 1.956, 0.936 ], [ -1.956, 1.536 ], [ -0.424, -0.736 ], [ -1.148, 1.472 ], [ -0.6, 0.688 ], [ -1.564, -0.9 ], [ 1.176, 0.252 ], [ -1.444, -0.508 ], ... ],
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
    	[ [ -0.936, 0.32 ], [ -0.88, -0.116 ], [ 0.304, -1.472 ], [ 1.908, 0.724 ], [ 0.896, 1.3 ], [ 1.256, -1.184 ], [ 0.116, -0.2 ], [ 1.92, -0.356 ], ... ],
    	[ [ 1.656, -1.304 ], [ 0.816, -1.552 ], [ -1.004, -1.112 ], [ 0.824, 1.192 ], [ 0.436, 0.952 ], [ -0.288, 1.388 ], [ -0.6, -1.724 ], [ 1.64, 0.124 ], ... ],
    	[ [ -0.676, 1.992 ], [ 0.416, -0.452 ], [ 1.82, -1.78 ], [ 0.176, -0.932 ], [ -0.08, -0.956 ], [ -0.232, -1.856 ], [ -1.448, -0.528 ], [ 0.848, 1.732 ], ... ],
    	[ [ 1.392, 1.036 ], [ -1.24, 1.828 ], [ 0.012, -1.212 ], [ -0.2, -1.908 ], [ -0.252, 1.768 ], [ -1.82, -0.908 ], [ -1.068, 1.992 ], [ -1.004, 1.608 ], ... ],
    	[ [ -0.228, 0.244 ], [ 0.948, -1.236 ], [ -1.496, -0.44 ], [ 1.128, -1.388 ], [ -0.82, 1.248 ], [ -0.016, 0.696 ], [ 0.648, -1.348 ], [ 1.688, -1.564 ], ... ],
    	[ [ -1.208, -0.228 ], [ -1.68, 0.168 ], [ 0.316, -0.268 ], [ -0.26, -0.616 ], [ -0.972, 1.392 ], [ 0.428, -0.576 ], [ -1.144, 0.136 ], [ 1.768, -1.9 ], ... ],
    	[ [ -1.464, -0.472 ], [ -0.868, 0.644 ], [ -0.388, 0.964 ], [ 0.696, 1.2 ], [ 1.992, 1.136 ], [ -1.52, -0.26 ], [ -0.844, -0.484 ], [ -0.444, 1.988 ], ... ],
    	[ [ 1.956, 0.936 ], [ -1.956, 1.536 ], [ -0.424, -0.736 ], [ -1.148, 1.472 ], [ -0.6, 0.688 ], [ -1.564, -0.9 ], [ 1.176, 0.252 ], [ -1.444, -0.508 ], ... ],
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

