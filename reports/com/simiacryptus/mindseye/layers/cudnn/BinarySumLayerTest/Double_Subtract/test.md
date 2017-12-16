# BinarySumLayer
## Double_Subtract
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
      "id": "27b1f16d-e3ee-4e1b-9c95-79567bb5aad8",
      "isFrozen": false,
      "name": "BinarySumLayer/27b1f16d-e3ee-4e1b-9c95-79567bb5aad8",
      "rightFactor": -1.0,
      "leftFactor": 1.0,
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
    	[ [ 0.4 ], [ 1.072 ] ],
    	[ [ 1.1 ], [ 1.168 ] ]
    ],
    [
    	[ [ 1.14 ], [ 0.764 ] ],
    	[ [ -1.992 ], [ 1.34 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ -0.7399999999999999 ], [ 0.30800000000000005 ] ],
    	[ [ 3.092 ], [ -0.17200000000000015 ] ]
    ]
    --------------------
    Derivative: 
    [
    	[ [ 1.0 ], [ 1.0 ] ],
    	[ [ 1.0 ], [ 1.0 ] ]
    ],
    [
    	[ [ -1.0 ], [ -1.0 ] ],
    	[ [ -1.0 ], [ -1.0 ] ]
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
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (120#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (120#)}
```



Code from [SingleDerivativeTester.java:77](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/SingleDerivativeTester.java#L77) executed in 0.01 seconds: 
```java
    return test(component, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ 0.152 ], [ 0.096 ] ],
    	[ [ 1.244 ], [ -0.736 ] ]
    ],
    [
    	[ [ -1.012 ], [ 1.116 ] ],
    	[ [ 0.916 ], [ -0.348 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.46854674608084007, negative=1, min=-0.736, max=-0.736, mean=0.189, count=4.0, positive=3, stdDev=0.7033256713642692, zeros=0},
    {meanExponent=-0.1109201438200571, negative=2, min=-0.348, max=-0.348, mean=0.168, count=4.0, positive=2, stdDev=0.8827321224471215, zeros=0}
    Output: [
    	[ [ 1.164 ], [ -1.02 ] ],
    	[ [ 0.32799999999999996 ], [ -0.388 ] ]
    ]
    Outputs Statistics: {meanExponent=-0.20518531965458162, negative=2, min=-0.388, max=-0.388, mean=0.02099999999999999, count=4.0, positive=2, stdDev=0.8141959223675834, zeros=0}
    Feedback for input 0
    Inputs Values: [
    	[ [ 0.152 ], [ 0.096 ] ],
    	[ [ 1.244 ], [ -0.736 ] ]
    ]
    Value Statistics: {meanExponent=-0.46854674608084007, negative=1, min=-0.736, max=-0.736, mean=0.189, count=4.0, positive=3, stdDev=0.7033256713642692, zeros=0}
    Implemented Feedback: [ [ 1.0, 0.0, 0.0, 0.0 ], [ 0.0, 1.0, 0.0, 0.0 ], [ 
```
...[skipping 1488 bytes](etc/50.txt)...
```
    0, stdDev=0.4330127018922193, zeros=12}
    Measured Feedback: [ [ -0.9999999999998899, 0.0, 0.0, 0.0 ], [ 0.0, -0.9999999999998899, 0.0, 0.0 ], [ 0.0, 0.0, -0.9999999999998899, 0.0 ], [ 0.0, 0.0, 0.0, -0.9999999999998899 ] ]
    Measured Statistics: {meanExponent=-4.7830642341045674E-14, negative=4, min=-0.9999999999998899, max=-0.9999999999998899, mean=-0.24999999999997247, count=16.0, positive=0, stdDev=0.4330127018921716, zeros=12}
    Feedback Error: [ [ 1.1013412404281553E-13, 0.0, 0.0, 0.0 ], [ 0.0, 1.1013412404281553E-13, 0.0, 0.0 ], [ 0.0, 0.0, 1.1013412404281553E-13, 0.0 ], [ 0.0, 0.0, 0.0, 1.1013412404281553E-13 ] ]
    Error Statistics: {meanExponent=-12.958078098036825, negative=0, min=1.1013412404281553E-13, max=1.1013412404281553E-13, mean=2.7533531010703882E-14, count=16.0, positive=4, stdDev=4.7689474622312385E-14, zeros=12}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 2.7534e-14 +- 4.7689e-14 [0.0000e+00 - 1.1013e-13] (32#)
    relativeTol: 5.5067e-14 +- 0.0000e+00 [5.5067e-14 - 5.5067e-14] (8#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=2.7534e-14 +- 4.7689e-14 [0.0000e+00 - 1.1013e-13] (32#), relativeTol=5.5067e-14 +- 0.0000e+00 [5.5067e-14 - 5.5067e-14] (8#)}
```



### Performance
Now we execute larger-scale runs to benchmark performance:

Code from [PerformanceTester.java:66](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/PerformanceTester.java#L66) executed in 0.28 seconds: 
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
    	Evaluation performance: 0.011949s +- 0.001481s [0.010439s - 0.013812s]
    	Learning performance: 0.032817s +- 0.009612s [0.026724s - 0.051903s]
    
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
    	[ [ 1.288 ], [ -0.736 ], [ 1.728 ], [ 0.48 ], [ -0.82 ], [ 0.08 ], [ -0.48 ], [ -1.52 ], ... ],
    	[ [ 0.288 ], [ -0.304 ], [ 0.028 ], [ 1.056 ], [ 1.976 ], [ -0.78 ], [ -0.244 ], [ -0.312 ], ... ],
    	[ [ 1.616 ], [ -1.408 ], [ 1.86 ], [ 0.012 ], [ -0.376 ], [ -0.644 ], [ 1.5 ], [ -1.264 ], ... ],
    	[ [ 1.204 ], [ 1.08 ], [ 0.616 ], [ -0.448 ], [ 1.588 ], [ 1.188 ], [ 1.744 ], [ 1.832 ], ... ],
    	[ [ -0.964 ], [ -1.584 ], [ 1.3 ], [ 0.408 ], [ -1.752 ], [ 0.004 ], [ -1.0 ], [ 0.18 ], ... ],
    	[ [ -1.016 ], [ 1.188 ], [ -1.636 ], [ 0.376 ], [ -0.648 ], [ -1.688 ], [ -0.18 ], [ 1.696 ], ... ],
    	[ [ 1.912 ], [ 0.064 ], [ 0.012 ], [ 0.632 ], [ 1.516 ], [ -1.736 ], [ -0.272 ], [ 1.152 ], ... ],
    	[ [ 0.744 ], [ 1.232 ], [ -0.908 ], [ 0.856 ], [ -1.824 ], [ -1.236 ], [ -1.924 ], [ -0.56 ], ... ],
    	...
    ]
    [
    	[ [ -0.528 ], [ 1.284 ], [ 0.112 ], [ 0.416 ], [ -1.876 ], [ 0.872 ], [ -1.88 ], [ -0.216 ], ... ],
    	[ [ 0.58 ], [ 0.124 ], [ -1.252 ], [ 1.296 ], [ -1.888 ], [ 1.364 ], [ 0.428 ], [ 0.88 ], ... ],
    	[ [ 1.048 ], [ -1.288 ], [ -0.628 ], [ 0.144 ], [ 0.904 ], [ 1.4 ], [ -1.856 ], [ -0.88 ], ... ],
    	[ [ -0.888 ], [ 0.572 ], [ 1.496 ], [ -0.36 ], [ -1.288 ], [ 0.288 ], [ 0.944 ], [ 0.468 ], ... ],
    	[ [ -0.576 ], [ 1.54 ], [ 0.512 ], [ -1.896 ], [ -1.888 ], [ 1.44 ], [ 0.236 ], [ -0.968 ], ... ],
    	[ [ 1.948 ], [ -0.572 ], [ 0.848 ], [ 0.244 ], [ 1.86 ], [ -1.424 ], [ 0.448 ], [ -1.472 ], ... ],
    	[ [ -1.896 ], [ -1.352 ], [ 1.68 ], [ -0.284 ], [ -1.432 ], [ 0.128 ], [ -1.644 ], [ 0.94 ], ... ],
    	[ [ 0.016 ], [ 1.692 ], [ -0.496 ], [ 1.808 ], [ 1.988 ], [ -0.48 ], [ 0.324 ], [ 1.812 ], ... ],
    	...
    ]
```



First, we use a conjugate gradient descent method, which converges the fastest for purely linear functions.

Code from [LearningTester.java:300](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L300) executed in 0.02 seconds: 
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
Logging: 
```
    Zero gradient: 0.0
    Constructing line search parameters: GD
    F(0.0) = LineSearchPoint{point=PointSample{avg=2.7225582175999965}, derivative=0.0}
    Iteration 1 failed, aborting. Error: 2.7225582175999965 Total: 249392891095756.1200; Orientation: 0.0005; Line Search: 0.0038
    
```

Returns: 

```
    2.7225582175999965
```



This training run resulted in the following regressed input:

Code from [LearningTester.java:144](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L144) executed in 0.01 seconds: 
```java
    return Arrays.stream(input_gd).map(x -> x.prettyPrint()).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [
    	[ [ 1.088 ], [ 1.792 ], [ -0.872 ], [ 0.464 ], [ -0.94 ], [ -1.0 ], [ -1.072 ], [ -1.988 ], ... ],
    	[ [ 1.08 ], [ 0.468 ], [ -1.2 ], [ 1.156 ], [ -0.812 ], [ 0.292 ], [ 0.908 ], [ 0.016 ], ... ],
    	[ [ 1.628 ], [ -0.32 ], [ 0.408 ], [ 1.684 ], [ 0.82 ], [ 0.908 ], [ 0.38 ], [ -0.216 ], ... ],
    	[ [ -0.312 ], [ 1.296 ], [ -1.644 ], [ 0.268 ], [ -1.012 ], [ -0.212 ], [ -1.944 ], [ 1.396 ], ... ],
    	[ [ 1.232 ], [ 0.8 ], [ 0.472 ], [ 1.016 ], [ 1.648 ], [ 1.164 ], [ 1.928 ], [ -0.448 ], ... ],
    	[ [ -0.208 ], [ -0.688 ], [ 1.324 ], [ -1.056 ], [ -0.636 ], [ -0.812 ], [ -0.228 ], [ 0.388 ], ... ],
    	[ [ -0.068 ], [ -0.7 ], [ 1.608 ], [ 1.712 ], [ 0.752 ], [ -0.516 ], [ -1.868 ], [ 1.34 ], ... ],
    	[ [ -0.072 ], [ 1.668 ], [ -0.82 ], [ 0.08 ], [ -0.12 ], [ 1.016 ], [ 1.688 ], [ -0.304 ], ... ],
    	...
    ]
    [
    	[ [ -0.676 ], [ -0.292 ], [ 0.668 ], [ 0.584 ], [ -0.812 ], [ 0.736 ], [ 0.832 ], [ 1.856 ], ... ],
    	[ [ -0.868 ], [ 1.096 ], [ 1.748 ], [ -1.696 ], [ 0.2 ], [ 0.4 ], [ -0.748 ], [ -0.364 ], ... ],
    	[ [ -1.384 ], [ -1.968 ], [ 1.084 ], [ 0.208 ], [ 1.008 ], [ -1.496 ], [ -1.208 ], [ -0.676 ], ... ],
    	[ [ -1.448 ], [ -1.768 ], [ 1.144 ], [ -0.312 ], [ -1.984 ], [ -1.452 ], [ 0.556 ], [ 1.356 ], ... ],
    	[ [ -0.276 ], [ 1.54 ], [ 0.036 ], [ -0.944 ], [ -1.996 ], [ -1.644 ], [ -1.96 ], [ -0.616 ], ... ],
    	[ [ 1.66 ], [ -1.388 ], [ 0.828 ], [ -0.836 ], [ 1.728 ], [ -1.984 ], [ 1.504 ], [ 0.568 ], ... ],
    	[ [ 0.796 ], [ 1.0 ], [ 1.812 ], [ 0.46 ], [ 0.692 ], [ 0.712 ], [ 0.828 ], [ -0.512 ], ... ],
    	[ [ 0.192 ], [ 0.152 ], [ 1.444 ], [ -1.304 ], [ -0.96 ], [ -1.644 ], [ -1.756 ], [ -0.7 ], ... ],
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
Logging: 
```
    LBFGS Accumulation History: 1 points
    Constructing line search parameters: GD
    th(0)=2.7225582175999965;dx=0.0 (ERROR: Starting derivative negative)
    Iteration 1 failed, aborting. Error: 2.7225582175999965 Total: 249392914330899.1000; Orientation: 0.0006; Line Search: 0.0051
    
```

Returns: 

```
    2.7225582175999965
```



This training run resulted in the following regressed input:

Code from [LearningTester.java:154](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L154) executed in 0.01 seconds: 
```java
    return Arrays.stream(input_lbgfs).map(x -> x.prettyPrint()).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [
    	[ [ 1.088 ], [ 1.792 ], [ -0.872 ], [ 0.464 ], [ -0.94 ], [ -1.0 ], [ -1.072 ], [ -1.988 ], ... ],
    	[ [ 1.08 ], [ 0.468 ], [ -1.2 ], [ 1.156 ], [ -0.812 ], [ 0.292 ], [ 0.908 ], [ 0.016 ], ... ],
    	[ [ 1.628 ], [ -0.32 ], [ 0.408 ], [ 1.684 ], [ 0.82 ], [ 0.908 ], [ 0.38 ], [ -0.216 ], ... ],
    	[ [ -0.312 ], [ 1.296 ], [ -1.644 ], [ 0.268 ], [ -1.012 ], [ -0.212 ], [ -1.944 ], [ 1.396 ], ... ],
    	[ [ 1.232 ], [ 0.8 ], [ 0.472 ], [ 1.016 ], [ 1.648 ], [ 1.164 ], [ 1.928 ], [ -0.448 ], ... ],
    	[ [ -0.208 ], [ -0.688 ], [ 1.324 ], [ -1.056 ], [ -0.636 ], [ -0.812 ], [ -0.228 ], [ 0.388 ], ... ],
    	[ [ -0.068 ], [ -0.7 ], [ 1.608 ], [ 1.712 ], [ 0.752 ], [ -0.516 ], [ -1.868 ], [ 1.34 ], ... ],
    	[ [ -0.072 ], [ 1.668 ], [ -0.82 ], [ 0.08 ], [ -0.12 ], [ 1.016 ], [ 1.688 ], [ -0.304 ], ... ],
    	...
    ]
    [
    	[ [ -0.676 ], [ -0.292 ], [ 0.668 ], [ 0.584 ], [ -0.812 ], [ 0.736 ], [ 0.832 ], [ 1.856 ], ... ],
    	[ [ -0.868 ], [ 1.096 ], [ 1.748 ], [ -1.696 ], [ 0.2 ], [ 0.4 ], [ -0.748 ], [ -0.364 ], ... ],
    	[ [ -1.384 ], [ -1.968 ], [ 1.084 ], [ 0.208 ], [ 1.008 ], [ -1.496 ], [ -1.208 ], [ -0.676 ], ... ],
    	[ [ -1.448 ], [ -1.768 ], [ 1.144 ], [ -0.312 ], [ -1.984 ], [ -1.452 ], [ 0.556 ], [ 1.356 ], ... ],
    	[ [ -0.276 ], [ 1.54 ], [ 0.036 ], [ -0.944 ], [ -1.996 ], [ -1.644 ], [ -1.96 ], [ -0.616 ], ... ],
    	[ [ 1.66 ], [ -1.388 ], [ 0.828 ], [ -0.836 ], [ 1.728 ], [ -1.984 ], [ 1.504 ], [ 0.568 ], ... ],
    	[ [ 0.796 ], [ 1.0 ], [ 1.812 ], [ 0.46 ], [ 0.692 ], [ 0.712 ], [ 0.828 ], [ -0.512 ], ... ],
    	[ [ 0.192 ], [ 0.152 ], [ 1.444 ], [ -1.304 ], [ -0.96 ], [ -1.644 ], [ -1.756 ], [ -0.7 ], ... ],
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

