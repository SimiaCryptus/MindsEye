# MeanSqLossLayer
## MeanSqLossLayerTest
### Batch Execution
Code from [BatchingTester.java:66](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/BatchingTester.java#L66) executed in 0.00 seconds: 
```java
    return test(reference, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (130#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (130#)}
```



Code from [SingleDerivativeTester.java:77](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/SingleDerivativeTester.java#L77) executed in 0.00 seconds: 
```java
    return test(component, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ 0.208 ], [ -1.524 ], [ 0.888 ] ],
    	[ [ 0.748 ], [ -1.344 ], [ 1.004 ] ]
    ],
    [
    	[ [ 0.78 ], [ 0.7 ], [ 0.26 ] ],
    	[ [ -0.864 ], [ -0.064 ], [ -0.272 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.09108402547729792, negative=2, min=1.004, max=1.004, mean=-0.003333333333333336, count=6.0, positive=4, stdDev=1.0430862327194665, zeros=0},
    {meanExponent=-0.44509523147124425, negative=3, min=-0.272, max=-0.272, mean=0.09000000000000001, count=6.0, positive=3, stdDev=0.5689135845334217, zeros=0}
    Output: [ 1.9221440000000003 ]
    Outputs Statistics: {meanExponent=0.2837859203058675, negative=0, min=1.9221440000000003, max=1.9221440000000003, mean=1.9221440000000003, count=1.0, positive=1, stdDev=0.0, zeros=0}
    Feedback for input 0
    Inputs Values: [
    	[ [ 0.208 ], [ -1.524 ], [ 0.888 ] ],
    	[ [ 0.748 ], [ -1.344 ], [ 1.004 ] ]
    ]
    Value Statistics: {meanExponent=-0.09108402547729792, negative=2, min=1.004, max=1.004, mean=-0.003333333333333336, count=6.0, positive=4, stdDev=1.0430862327194665, zeros=0}
    Implement
```
...[skipping 1748 bytes](etc/290.txt)...
```
    1111111111111128, count=6.0, positive=3, stdDev=0.4610896380542374, zeros=0}
    Measured Feedback: [ [ 0.1906833333342739 ], [ -0.5373166666688967 ], [ 0.7413499999997519 ], [ 0.4266833333343989 ], [ -0.20931666666612614 ], [ -0.425316666665676 ] ]
    Measured Statistics: {meanExponent=-0.4233019617020286, negative=3, min=-0.425316666665676, max=-0.425316666665676, mean=0.031127777777954318, count=6.0, positive=3, stdDev=0.46108963805462805, zeros=0}
    Feedback Error: [ [ 1.6666667607218022E-5 ], [ 1.6666664436670864E-5 ], [ 1.6666666418529985E-5 ], [ 1.666666773225689E-5 ], [ 1.666666720717691E-5 ], [ 1.6666667657316836E-5 ] ]
    Error Statistics: {meanExponent=-4.778151245783731, negative=0, min=1.6666667657316836E-5, max=1.6666667657316836E-5, mean=1.6666666843194917E-5, count=6.0, positive=6, stdDev=1.1593828079623528E-12, zeros=0}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 1.6667e-05 +- 1.3258e-12 [1.6667e-05 - 1.6667e-05] (12#)
    relativeTol: 2.4898e-05 +- 1.2300e-05 [1.1241e-05 - 4.3708e-05] (12#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=1.6667e-05 +- 1.3258e-12 [1.6667e-05 - 1.6667e-05] (12#), relativeTol=2.4898e-05 +- 1.2300e-05 [1.1241e-05 - 4.3708e-05] (12#)}
```



### Json Serialization
Code from [JsonTest.java:36](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/JsonTest.java#L36) executed in 0.00 seconds: 
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
      "class": "com.simiacryptus.mindseye.layers.java.MeanSqLossLayer",
      "id": "91f7c46b-1d23-4f63-a021-61ec0fca9bad",
      "isFrozen": false,
      "name": "MeanSqLossLayer/91f7c46b-1d23-4f63-a021-61ec0fca9bad"
    }
```



### Example Input/Output Pair
Code from [ReferenceIO.java:68](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/ReferenceIO.java#L68) executed in 0.00 seconds: 
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
    	[ [ 0.88 ], [ 1.272 ], [ 0.68 ] ],
    	[ [ -0.956 ], [ 1.08 ], [ 0.32 ] ]
    ],
    [
    	[ [ 0.904 ], [ -1.428 ], [ 1.704 ] ],
    	[ [ 1.64 ], [ -0.728 ], [ 1.548 ] ]
    ]]
    --------------------
    Output: 
    [ 3.309202666666667 ]
    --------------------
    Derivative: 
    [
    	[ [ -0.008000000000000007 ], [ 0.9 ], [ -0.3413333333333333 ] ],
    	[ [ -0.8653333333333333 ], [ 0.6026666666666667 ], [ -0.4093333333333333 ] ]
    ],
    [
    	[ [ 0.008000000000000007 ], [ -0.9 ], [ 0.3413333333333333 ] ],
    	[ [ 0.8653333333333333 ], [ -0.6026666666666667 ], [ 0.4093333333333333 ] ]
    ]
```



### Input Learning
In this test, we use a network to learn this target input, given it's pre-evaluated output:

Code from [LearningTester.java:127](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L127) executed in 0.00 seconds: 
```java
    return Arrays.stream(input_target).map(x -> x.prettyPrint()).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [
    	[ [ 0.824 ], [ -1.248 ], [ -0.052 ] ],
    	[ [ 1.156 ], [ -0.756 ], [ -0.084 ] ]
    ]
    [
    	[ [ -0.54 ], [ -0.86 ], [ -1.568 ] ],
    	[ [ 0.412 ], [ 0.032 ], [ -0.908 ] ]
    ]
```



First, we use a conjugate gradient descent method, which converges the fastest for purely linear functions.

Code from [LearningTester.java:225](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L225) executed in 0.00 seconds: 
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
    F(0.0) = LineSearchPoint{point=PointSample{avg=1.0549864503751114}, derivative=0.0}
    Iteration 1 failed, aborting. Error: 1.0549864503751114 Total: 239673924571627.0600; Orientation: 0.0000; Line Search: 0.0001
    
```

Returns: 

```
    1.0549864503751114
```



This training run resulted in the following regressed input:

Code from [LearningTester.java:144](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L144) executed in 0.00 seconds: 
```java
    return Arrays.stream(input_gd).map(x -> x.prettyPrint()).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [
    	[ [ 1.156 ], [ 0.824 ], [ -0.756 ] ],
    	[ [ -0.084 ], [ -0.052 ], [ -1.248 ] ]
    ]
    [
    	[ [ -0.908 ], [ -0.86 ], [ 0.412 ] ],
    	[ [ 0.032 ], [ -0.54 ], [ -1.568 ] ]
    ]
```



Next, we run the same optimization using L-BFGS, which is nearly ideal for purely second-order or quadratic functions.

Code from [LearningTester.java:249](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L249) executed in 0.00 seconds: 
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
    th(0)=1.0549864503751114;dx=0.0 (ERROR: Starting derivative negative)
    Iteration 1 failed, aborting. Error: 1.0549864503751114 Total: 239673926644275.0600; Orientation: 0.0001; Line Search: 0.0002
    
```

Returns: 

```
    1.0549864503751114
```



This training run resulted in the following regressed input:

Code from [LearningTester.java:154](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L154) executed in 0.00 seconds: 
```java
    return Arrays.stream(input_lbgfs).map(x -> x.prettyPrint()).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [
    	[ [ 1.156 ], [ 0.824 ], [ -0.756 ] ],
    	[ [ -0.084 ], [ -0.052 ], [ -1.248 ] ]
    ]
    [
    	[ [ -0.908 ], [ -0.86 ], [ 0.412 ] ],
    	[ [ 0.032 ], [ -0.54 ], [ -1.568 ] ]
    ]
```



Code from [LearningTester.java:95](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L95) executed in 0.00 seconds: 
```java
    return TestUtil.compare(runs);
```

Code from [LearningTester.java:98](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L98) executed in 0.00 seconds: 
```java
    return TestUtil.compareTime(runs);
```

### Performance
Now we execute larger-scale runs to benchmark performance:

Code from [PerformanceTester.java:66](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/PerformanceTester.java#L66) executed in 0.00 seconds: 
```java
    test(component, inputPrototype);
```
Logging: 
```
    100 batches
    Input Dimensions:
    	[2, 3, 1]
    	[2, 3, 1]
    Performance:
    	Evaluation performance: 0.000271s +- 0.000112s [0.000186s - 0.000488s]
    	Learning performance: 0.000041s +- 0.000004s [0.000039s - 0.000050s]
    
```

