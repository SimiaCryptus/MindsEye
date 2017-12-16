# BinarySumLayer
## Float_Avg
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
      "class": "com.simiacryptus.mindseye.layers.cudnn.BinarySumLayer",
      "id": "70f2afb1-437a-434f-a44c-992487d0a30b",
      "isFrozen": false,
      "name": "BinarySumLayer/70f2afb1-437a-434f-a44c-992487d0a30b",
      "rightFactor": 0.5,
      "leftFactor": 0.5,
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
    	[ [ 1.852 ], [ -0.472 ] ],
    	[ [ 0.436 ], [ -0.888 ] ]
    ],
    [
    	[ [ 1.428 ], [ -1.172 ] ],
    	[ [ -1.94 ], [ -0.2 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 0.0 ], [ 0.0 ] ],
    	[ [ 0.0 ], [ 0.0 ] ]
    ]
    --------------------
    Derivative: 
    [
    	[ [ 0.0 ], [ 0.0 ] ],
    	[ [ 0.0 ], [ 0.0 ] ]
    ],
    [
    	[ [ 0.0 ], [ 0.0 ] ],
    	[ [ 0.0 ], [ 0.0 ] ]
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
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (120#), relativeTol=0.0000e+00 +- 0.0000e+00 [Infinity - -Infinity] (0#)}
```



Code from [SingleDerivativeTester.java:77](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/SingleDerivativeTester.java#L77) executed in 0.01 seconds: 
```java
    return test(component, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ -1.664 ], [ 0.34 ] ],
    	[ [ 0.06 ], [ 0.804 ] ]
    ],
    [
    	[ [ -1.384 ], [ 0.128 ] ],
    	[ [ 0.568 ], [ 0.768 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.3909901154677362, negative=1, min=0.804, max=0.804, mean=-0.11499999999999994, count=4.0, positive=3, stdDev=0.9329539109730984, zeros=0},
    {meanExponent=-0.2779860961222155, negative=1, min=0.768, max=0.768, mean=0.020000000000000018, count=4.0, positive=3, stdDev=0.8430136416452583, zeros=0}
    Output: [
    	[ [ 0.0 ], [ 0.0 ] ],
    	[ [ 0.0 ], [ 0.0 ] ]
    ]
    Outputs Statistics: {meanExponent=NaN, negative=0, min=0.0, max=0.0, mean=0.0, count=4.0, positive=0, stdDev=0.0, zeros=4}
    Feedback for input 0
    Inputs Values: [
    	[ [ -1.664 ], [ 0.34 ] ],
    	[ [ 0.06 ], [ 0.804 ] ]
    ]
    Value Statistics: {meanExponent=-0.3909901154677362, negative=1, min=0.804, max=0.804, mean=-0.11499999999999994, count=4.0, positive=3, stdDev=0.9329539109730984, zeros=0}
    Implemented Feedback: [ [ 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0 ] 
```
...[skipping 781 bytes](etc/52.txt)...
```
    in=0.768, max=0.768, mean=0.020000000000000018, count=4.0, positive=3, stdDev=0.8430136416452583, zeros=0}
    Implemented Feedback: [ [ 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0 ] ]
    Implemented Statistics: {meanExponent=NaN, negative=0, min=0.0, max=0.0, mean=0.0, count=16.0, positive=0, stdDev=0.0, zeros=16}
    Measured Feedback: [ [ 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0 ] ]
    Measured Statistics: {meanExponent=NaN, negative=0, min=0.0, max=0.0, mean=0.0, count=16.0, positive=0, stdDev=0.0, zeros=16}
    Feedback Error: [ [ 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0 ] ]
    Error Statistics: {meanExponent=NaN, negative=0, min=0.0, max=0.0, mean=0.0, count=16.0, positive=0, stdDev=0.0, zeros=16}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (32#)
    relativeTol: 0.0000e+00 +- 0.0000e+00 [Infinity - -Infinity] (0#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (32#), relativeTol=0.0000e+00 +- 0.0000e+00 [Infinity - -Infinity] (0#)}
```



### Performance
Now we execute larger-scale runs to benchmark performance:

Code from [PerformanceTester.java:66](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/PerformanceTester.java#L66) executed in 0.32 seconds: 
```java
    test(component, inputPrototype);
```
Logging: 
```
    100 batches
    Input Dimensions:
    	[100, 100, 1]
    	[100, 100, 1]
    Performance:
    	Evaluation performance: 0.013831s +- 0.004685s [0.010990s - 0.023165s]
    	Learning performance: 0.036046s +- 0.015994s [0.027317s - 0.068006s]
    
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
    	[ [ 0.916 ], [ 1.824 ], [ 0.592 ], [ 1.696 ], [ -1.992 ], [ -1.516 ], [ -0.152 ], [ -0.812 ], ... ],
    	[ [ 1.308 ], [ 0.072 ], [ -1.08 ], [ 0.084 ], [ -1.44 ], [ 1.512 ], [ -1.112 ], [ -0.308 ], ... ],
    	[ [ -0.2 ], [ -0.712 ], [ 0.752 ], [ 0.712 ], [ -1.572 ], [ -0.592 ], [ -0.596 ], [ 1.376 ], ... ],
    	[ [ -1.408 ], [ -0.644 ], [ 0.312 ], [ 0.26 ], [ -1.408 ], [ -1.536 ], [ 1.276 ], [ -0.86 ], ... ],
    	[ [ -1.864 ], [ -0.236 ], [ -1.704 ], [ 1.272 ], [ 0.28 ], [ -0.468 ], [ 1.52 ], [ -0.28 ], ... ],
    	[ [ -0.88 ], [ -1.644 ], [ -0.544 ], [ 1.928 ], [ -0.828 ], [ 0.96 ], [ -0.64 ], [ -0.052 ], ... ],
    	[ [ 1.116 ], [ -0.32 ], [ -0.044 ], [ 0.172 ], [ -0.248 ], [ 0.624 ], [ -1.024 ], [ 1.164 ], ... ],
    	[ [ -1.124 ], [ 0.04 ], [ -0.276 ], [ 1.984 ], [ -1.744 ], [ -0.768 ], [ -0.188 ], [ 1.62 ], ... ],
    	...
    ]
    [
    	[ [ 1.612 ], [ -0.428 ], [ 1.584 ], [ -1.34 ], [ -1.632 ], [ -1.448 ], [ 0.64 ], [ -0.032 ], ... ],
    	[ [ 0.628 ], [ -1.512 ], [ 0.06 ], [ 1.5 ], [ 0.36 ], [ 1.164 ], [ 0.372 ], [ -1.644 ], ... ],
    	[ [ 0.208 ], [ 1.788 ], [ -0.164 ], [ 0.7 ], [ -0.744 ], [ -0.196 ], [ -0.212 ], [ -0.388 ], ... ],
    	[ [ -0.636 ], [ -0.476 ], [ 1.892 ], [ 0.176 ], [ 0.712 ], [ -1.356 ], [ -1.548 ], [ 1.472 ], ... ],
    	[ [ 1.928 ], [ -1.652 ], [ -0.176 ], [ -1.968 ], [ 0.616 ], [ 1.108 ], [ 0.228 ], [ -0.076 ], ... ],
    	[ [ 1.472 ], [ -1.084 ], [ 0.32 ], [ -1.048 ], [ 1.652 ], [ -1.86 ], [ 1.836 ], [ 1.232 ], ... ],
    	[ [ -1.528 ], [ 0.664 ], [ 1.496 ], [ -0.364 ], [ 0.088 ], [ 0.44 ], [ 0.22 ], [ -1.828 ], ... ],
    	[ [ 0.48 ], [ 1.888 ], [ 1.076 ], [ -1.848 ], [ 1.864 ], [ -0.912 ], [ 1.936 ], [ -1.512 ], ... ],
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
    	[ [ -1.28 ], [ 0.92 ], [ 1.544 ], [ 0.796 ], [ -1.732 ], [ 1.984 ], [ 0.024 ], [ 1.388 ], ... ],
    	[ [ 1.464 ], [ 0.904 ], [ -0.66 ], [ -0.776 ], [ -1.16 ], [ 0.992 ], [ 1.484 ], [ 1.204 ], ... ],
    	[ [ -0.172 ], [ -1.232 ], [ 1.74 ], [ 1.044 ], [ 1.04 ], [ -1.56 ], [ -0.284 ], [ 1.424 ], ... ],
    	[ [ 0.724 ], [ 1.556 ], [ 0.524 ], [ 1.32 ], [ 1.78 ], [ 1.036 ], [ 0.984 ], [ -0.748 ], ... ],
    	[ [ 1.884 ], [ 0.408 ], [ 0.604 ], [ 0.34 ], [ 1.392 ], [ -0.464 ], [ -0.876 ], [ 1.624 ], ... ],
    	[ [ -1.852 ], [ 1.108 ], [ -1.316 ], [ -0.86 ], [ 1.336 ], [ 0.452 ], [ -1.676 ], [ 0.056 ], ... ],
    	[ [ -0.892 ], [ -0.02 ], [ -0.496 ], [ 1.872 ], [ -0.564 ], [ 1.084 ], [ 1.032 ], [ -1.724 ], ... ],
    	[ [ 0.836 ], [ 1.124 ], [ 1.076 ], [ -1.384 ], [ 1.86 ], [ 1.072 ], [ -1.056 ], [ -1.012 ], ... ],
    	...
    ]
    [
    	[ [ -0.772 ], [ -1.08 ], [ 1.332 ], [ -0.7 ], [ -1.748 ], [ -1.416 ], [ 0.296 ], [ -1.8 ], ... ],
    	[ [ 0.56 ], [ -0.272 ], [ 1.464 ], [ -0.684 ], [ 0.164 ], [ -1.86 ], [ -1.332 ], [ 1.052 ], ... ],
    	[ [ -0.7 ], [ 1.552 ], [ 1.648 ], [ 1.436 ], [ 0.284 ], [ -0.66 ], [ 0.868 ], [ 1.316 ], ... ],
    	[ [ 1.948 ], [ 1.876 ], [ 0.14 ], [ -0.496 ], [ 0.224 ], [ 1.612 ], [ 1.624 ], [ 0.328 ], ... ],
    	[ [ 1.82 ], [ -0.5 ], [ -1.096 ], [ 0.204 ], [ 0.856 ], [ -0.196 ], [ -0.264 ], [ 1.688 ], ... ],
    	[ [ -1.744 ], [ 1.904 ], [ 1.348 ], [ -0.14 ], [ -0.752 ], [ -1.676 ], [ 0.052 ], [ -0.492 ], ... ],
    	[ [ 1.312 ], [ 1.86 ], [ 1.616 ], [ -0.192 ], [ 0.828 ], [ -0.436 ], [ -1.1 ], [ 1.956 ], ... ],
    	[ [ 1.688 ], [ -0.172 ], [ -1.584 ], [ -1.856 ], [ 1.588 ], [ 1.36 ], [ 0.144 ], [ -0.228 ], ... ],
    	...
    ]
```



Next, we run the same optimization using L-BFGS, which is nearly ideal for purely second-order or quadratic functions.

Code from [LearningTester.java:324](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L324) executed in 0.01 seconds: 
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
    	[ [ -1.28 ], [ 0.92 ], [ 1.544 ], [ 0.796 ], [ -1.732 ], [ 1.984 ], [ 0.024 ], [ 1.388 ], ... ],
    	[ [ 1.464 ], [ 0.904 ], [ -0.66 ], [ -0.776 ], [ -1.16 ], [ 0.992 ], [ 1.484 ], [ 1.204 ], ... ],
    	[ [ -0.172 ], [ -1.232 ], [ 1.74 ], [ 1.044 ], [ 1.04 ], [ -1.56 ], [ -0.284 ], [ 1.424 ], ... ],
    	[ [ 0.724 ], [ 1.556 ], [ 0.524 ], [ 1.32 ], [ 1.78 ], [ 1.036 ], [ 0.984 ], [ -0.748 ], ... ],
    	[ [ 1.884 ], [ 0.408 ], [ 0.604 ], [ 0.34 ], [ 1.392 ], [ -0.464 ], [ -0.876 ], [ 1.624 ], ... ],
    	[ [ -1.852 ], [ 1.108 ], [ -1.316 ], [ -0.86 ], [ 1.336 ], [ 0.452 ], [ -1.676 ], [ 0.056 ], ... ],
    	[ [ -0.892 ], [ -0.02 ], [ -0.496 ], [ 1.872 ], [ -0.564 ], [ 1.084 ], [ 1.032 ], [ -1.724 ], ... ],
    	[ [ 0.836 ], [ 1.124 ], [ 1.076 ], [ -1.384 ], [ 1.86 ], [ 1.072 ], [ -1.056 ], [ -1.012 ], ... ],
    	...
    ]
    [
    	[ [ -0.772 ], [ -1.08 ], [ 1.332 ], [ -0.7 ], [ -1.748 ], [ -1.416 ], [ 0.296 ], [ -1.8 ], ... ],
    	[ [ 0.56 ], [ -0.272 ], [ 1.464 ], [ -0.684 ], [ 0.164 ], [ -1.86 ], [ -1.332 ], [ 1.052 ], ... ],
    	[ [ -0.7 ], [ 1.552 ], [ 1.648 ], [ 1.436 ], [ 0.284 ], [ -0.66 ], [ 0.868 ], [ 1.316 ], ... ],
    	[ [ 1.948 ], [ 1.876 ], [ 0.14 ], [ -0.496 ], [ 0.224 ], [ 1.612 ], [ 1.624 ], [ 0.328 ], ... ],
    	[ [ 1.82 ], [ -0.5 ], [ -1.096 ], [ 0.204 ], [ 0.856 ], [ -0.196 ], [ -0.264 ], [ 1.688 ], ... ],
    	[ [ -1.744 ], [ 1.904 ], [ 1.348 ], [ -0.14 ], [ -0.752 ], [ -1.676 ], [ 0.052 ], [ -0.492 ], ... ],
    	[ [ 1.312 ], [ 1.86 ], [ 1.616 ], [ -0.192 ], [ 0.828 ], [ -0.436 ], [ -1.1 ], [ 1.956 ], ... ],
    	[ [ 1.688 ], [ -0.172 ], [ -1.584 ], [ -1.856 ], [ 1.588 ], [ 1.36 ], [ 0.144 ], [ -0.228 ], ... ],
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

