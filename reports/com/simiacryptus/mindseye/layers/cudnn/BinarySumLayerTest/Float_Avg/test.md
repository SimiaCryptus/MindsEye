# BinarySumLayer
## Float_Avg
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
    	[ [ 1.336 ], [ 1.96 ] ],
    	[ [ 0.204 ], [ -0.208 ] ]
    ],
    [
    	[ [ -1.716 ], [ 0.448 ] ],
    	[ [ 1.264 ], [ 0.796 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.2385609920288342, negative=1, min=-0.208, max=-0.208, mean=0.823, count=4.0, positive=3, stdDev=0.8663226881480133, zeros=0},
    {meanExponent=-0.027886140201283518, negative=1, min=0.796, max=0.796, mean=0.198, count=4.0, positive=3, stdDev=1.142350208998974, zeros=0}
    Output: [
    	[ [ -0.18999999999999995 ], [ 1.204 ] ],
    	[ [ 0.734 ], [ 0.29400000000000004 ] ]
    ]
    Outputs Statistics: {meanExponent=-0.32664413044928436, negative=1, min=0.29400000000000004, max=0.29400000000000004, mean=0.5105, count=4.0, positive=3, stdDev=0.5168333870794339, zeros=0}
    Feedback for input 0
    Inputs Values: [
    	[ [ 1.336 ], [ 1.96 ] ],
    	[ [ 0.204 ], [ -0.208 ] ]
    ]
    Value Statistics: {meanExponent=-0.2385609920288342, negative=1, min=-0.208, max=-0.208, mean=0.823, count=4.0, positive=3, stdDev=0.8663226881480133, zeros=0}
    Implemented Feedback: [ [ 0.5, 0.0, 0.0, 0.0 ], [ 0.0
```
...[skipping 1517 bytes](etc/38.txt)...
```
    positive=4, stdDev=0.21650635094610965, zeros=12}
    Measured Feedback: [ [ 0.49999999999994493, 0.0, 0.0, 0.0 ], [ 0.0, 0.49999999999994493, 0.0, 0.0 ], [ 0.0, 0.0, 0.49999999999994493, 0.0 ], [ 0.0, 0.0, 0.0, 0.5000000000000004 ] ]
    Measured Statistics: {meanExponent=-0.301029995664017, negative=0, min=0.5000000000000004, max=0.5000000000000004, mean=0.1249999999999897, count=16.0, positive=4, stdDev=0.21650635094609183, zeros=12}
    Feedback Error: [ [ -5.5067062021407764E-14, 0.0, 0.0, 0.0 ], [ 0.0, -5.5067062021407764E-14, 0.0, 0.0 ], [ 0.0, 0.0, -5.5067062021407764E-14, 0.0 ], [ 0.0, 0.0, 0.0, 4.440892098500626E-16 ] ]
    Error Statistics: {meanExponent=-13.782463514991367, negative=3, min=4.440892098500626E-16, max=4.440892098500626E-16, mean=-1.0297318553398327E-14, count=16.0, positive=1, stdDev=2.15069536196907E-14, zeros=12}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 1.0381e-14 +- 2.1539e-14 [0.0000e+00 - 5.5955e-14] (32#)
    relativeTol: 4.1522e-14 +- 2.3718e-14 [4.4409e-16 - 5.5955e-14] (8#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=1.0381e-14 +- 2.1539e-14 [0.0000e+00 - 5.5955e-14] (32#), relativeTol=4.1522e-14 +- 2.3718e-14 [4.4409e-16 - 5.5955e-14] (8#)}
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
      "class": "com.simiacryptus.mindseye.layers.cudnn.BinarySumLayer",
      "id": "7a3d3e6b-429b-4417-8609-cba012e10991",
      "isFrozen": false,
      "name": "BinarySumLayer/7a3d3e6b-429b-4417-8609-cba012e10991",
      "rightFactor": 0.5,
      "leftFactor": 0.5
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
    	[ [ 0.164 ], [ -1.616 ] ],
    	[ [ 1.392 ], [ -1.596 ] ]
    ],
    [
    	[ [ -1.096 ], [ 1.26 ] ],
    	[ [ 1.132 ], [ 1.256 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ -0.466 ], [ -0.17800000000000005 ] ],
    	[ [ 1.262 ], [ -0.17000000000000004 ] ]
    ]
    --------------------
    Derivative: 
    [
    	[ [ 0.5 ], [ 0.5 ] ],
    	[ [ 0.5 ], [ 0.5 ] ]
    ],
    [
    	[ [ 0.5 ], [ 0.5 ] ],
    	[ [ 0.5 ], [ 0.5 ] ]
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
    	[ [ -0.056 ], [ 0.552 ], [ -0.564 ], [ -0.252 ], [ -0.112 ], [ -1.668 ], [ 0.668 ], [ -0.996 ], ... ],
    	[ [ -1.9 ], [ -1.988 ], [ -0.976 ], [ 1.928 ], [ -1.416 ], [ -1.124 ], [ 0.444 ], [ -0.312 ], ... ],
    	[ [ -0.816 ], [ 1.744 ], [ -0.076 ], [ 0.432 ], [ -0.744 ], [ -1.548 ], [ 0.616 ], [ -0.176 ], ... ],
    	[ [ -0.56 ], [ -1.124 ], [ 0.664 ], [ -1.652 ], [ -0.992 ], [ -0.428 ], [ 1.924 ], [ 0.104 ], ... ],
    	[ [ -1.724 ], [ -0.824 ], [ 1.696 ], [ 0.64 ], [ -1.96 ], [ -1.308 ], [ 0.828 ], [ -1.54 ], ... ],
    	[ [ 0.748 ], [ 0.252 ], [ -0.088 ], [ -1.22 ], [ -0.8 ], [ 0.668 ], [ 1.98 ], [ -0.656 ], ... ],
    	[ [ -0.084 ], [ 0.64 ], [ -1.544 ], [ -1.296 ], [ 1.292 ], [ -0.888 ], [ -0.392 ], [ 1.884 ], ... ],
    	[ [ 1.3 ], [ -1.812 ], [ -0.364 ], [ -1.38 ], [ 0.436 ], [ -1.092 ], [ 0.272 ], [ 1.684 ], ... ],
    	...
    ]
    [
    	[ [ -1.564 ], [ 1.132 ], [ 0.132 ], [ 1.048 ], [ -1.472 ], [ 1.312 ], [ -0.74 ], [ -0.616 ], ... ],
    	[ [ 0.956 ], [ -0.592 ], [ -1.32 ], [ -1.796 ], [ -0.8 ], [ -0.048 ], [ 1.468 ], [ -0.732 ], ... ],
    	[ [ 0.1 ], [ 0.84 ], [ 0.464 ], [ 0.4 ], [ 1.524 ], [ -0.272 ], [ 1.12 ], [ 0.2 ], ... ],
    	[ [ 0.2 ], [ 1.044 ], [ 0.66 ], [ 1.384 ], [ 1.604 ], [ -0.88 ], [ 1.64 ], [ 0.828 ], ... ],
    	[ [ 0.352 ], [ 0.092 ], [ -1.724 ], [ -0.132 ], [ 1.08 ], [ -1.104 ], [ -1.268 ], [ -1.652 ], ... ],
    	[ [ 0.516 ], [ -0.24 ], [ 0.516 ], [ 0.084 ], [ 0.728 ], [ -0.488 ], [ 0.072 ], [ 1.436 ], ... ],
    	[ [ -0.532 ], [ 1.248 ], [ 1.872 ], [ -1.92 ], [ -0.42 ], [ 0.296 ], [ 1.856 ], [ -0.212 ], ... ],
    	[ [ -0.856 ], [ 1.692 ], [ -0.316 ], [ 1.624 ], [ -0.876 ], [ -0.532 ], [ 1.4 ], [ -1.48 ], ... ],
    	...
    ]
```



First, we use a conjugate gradient descent method, which converges the fastest for purely linear functions.

Code from [LearningTester.java:225](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L225) executed in 0.13 seconds: 
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
    Constructing line search parameters: GD
    F(0.0) = LineSearchPoint{point=PointSample{avg=2.0170228824000023}, derivative=-8.0680915296E-4}
    New Minimum: 2.0170228824000023 > 2.017022882399917
    F(1.0E-10) = LineSearchPoint{point=PointSample{avg=2.017022882399917}, derivative=-8.068091529599839E-4}, delta = -8.526512829121202E-14
    New Minimum: 2.017022882399917 > 2.017022882399435
    F(7.000000000000001E-10) = LineSearchPoint{point=PointSample{avg=2.017022882399435}, derivative=-8.068091529598871E-4}, delta = -5.6710192097853E-13
    New Minimum: 2.017022882399435 > 2.0170228823960485
    F(4.900000000000001E-9) = LineSearchPoint{point=PointSample{avg=2.0170228823960485}, derivative=-8.068091529592094E-4}, delta = -3.9537262352951075E-12
    New Minimum: 2.0170228823960485 > 2.017022882372333
    F(3.430000000000001E-8) = LineSearchPoint{point=PointSample{avg=2.017022882372333}, derivative=-8.068091529544653E-4}, delta = -2.7669422308918E-11
    New Minimum: 2.017022882372333 > 2.017022882206293
    F(2.4010000000000004E-7) = LineS
```
...[skipping 7680 bytes](etc/39.txt)...
```
     gradient: 1.3374868086673423E-141
    F(0.0) = LineSearchPoint{point=PointSample{avg=4.47217740839788E-279}, derivative=-1.788870963359152E-282}
    New Minimum: 4.47217740839788E-279 > 4.251817505319342E-304
    F(5000.000000001542) = LineSearchPoint{point=PointSample{avg=4.251817505319342E-304}, derivative=5.515778312884602E-295}, delta = -4.47217740839788E-279
    4.251817505319342E-304 <= 4.47217740839788E-279
    Converged to right
    Iteration 12 complete. Error: 4.251817505319342E-304 Total: 239463962679432.0300; Orientation: 0.0003; Line Search: 0.0045
    Zero gradient: 4.123987150959296E-154
    F(0.0) = LineSearchPoint{point=PointSample{avg=4.251817505319342E-304}, derivative=-1.700727002127737E-307}
    New Minimum: 4.251817505319342E-304 > 0.0
    F(5000.000000001542) = LineSearchPoint{point=PointSample{avg=0.0}, derivative=5.2435E-320}, delta = -4.251817505319342E-304
    0.0 <= 4.251817505319342E-304
    Converged to right
    Iteration 13 complete. Error: 0.0 Total: 239463968528614.0300; Orientation: 0.0003; Line Search: 0.0039
    
```

Returns: 

```
    0.0
```



Training Converged

Next, we run the same optimization using L-BFGS, which is nearly ideal for purely second-order or quadratic functions.

Code from [LearningTester.java:249](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L249) executed in 0.09 seconds: 
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
    th(0)=2.0170228824000023;dx=-8.0680915296E-4
    New Minimum: 2.0170228824000023 > 2.0152850392602533
    WOLFE (weak): th(2.154434690031884)=2.0152850392602533; dx=-8.064615094345257E-4 delta=0.0017378431397490068
    New Minimum: 2.0152850392602533 > 2.013547945095792
    WOLFE (weak): th(4.308869380063768)=2.013547945095792; dx=-8.061138659090512E-4 delta=0.003474937304210446
    New Minimum: 2.013547945095792 > 2.0066070581906477
    WOLFE (weak): th(12.926608140191302)=2.0066070581906477; dx=-8.047232918071533E-4 delta=0.010415824209354607
    New Minimum: 2.0066070581906477 > 1.9755213642211344
    WOLFE (weak): th(51.70643256076521)=1.9755213642211344; dx=-7.984657083486132E-4 delta=0.041501518178867824
    New Minimum: 1.9755213642211344 > 1.8138293890668566
    WOLFE (weak): th(258.53216280382605)=1.8138293890668566; dx=-7.650919299030658E-4 delta=0.20319349333314563
    New Minimum: 1.8138293890668566 > 0.9596405809473888
    END: th(1551.1929768229563)=0.9
```
...[skipping 2555 bytes](etc/40.txt)...
```
    4.5103280863223737E-10 delta=1.1163062013647935E-5
    Iteration 6 complete. Error: 1.127582021580577E-7 Total: 239464041649508.9700; Orientation: 0.0005; Line Search: 0.0081
    LBFGS Accumulation History: 1 points
    th(0)=1.127582021580577E-7;dx=-4.510328086322315E-11
    New Minimum: 1.127582021580577E-7 > 9.941941573957754E-8
    WOLF (strong): th(9694.956105143481)=9.941941573957754E-8; dx=4.2351584770158075E-11 delta=1.3338786418480164E-8
    New Minimum: 9.941941573957754E-8 > 1.0492351171124814E-10
    END: th(4847.478052571741)=1.0492351171124814E-10; dx=-1.3758480465324794E-12 delta=1.1265327864634645E-7
    Iteration 7 complete. Error: 1.0492351171124814E-10 Total: 239464051245870.9700; Orientation: 0.0005; Line Search: 0.0074
    LBFGS Accumulation History: 1 points
    th(0)=1.0492351171124814E-10;dx=-4.196940468449917E-14
    MAX ALPHA: th(0)=1.0492351171124814E-10;th'(0)=-4.196940468449917E-14;
    Iteration 8 failed, aborting. Error: 1.0492351171124814E-10 Total: 239464059227830.9400; Orientation: 0.0006; Line Search: 0.0055
    
```

Returns: 

```
    1.0492351171124814E-10
```



Training Converged

Code from [LearningTester.java:95](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L95) executed in 0.00 seconds: 
```java
    return TestUtil.compare(runs);
```

Returns: 

![Result](etc/test.33.png)



Code from [LearningTester.java:98](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L98) executed in 0.00 seconds: 
```java
    return TestUtil.compareTime(runs);
```

Returns: 

![Result](etc/test.34.png)



### Performance
Now we execute larger-scale runs to benchmark performance:

Code from [PerformanceTester.java:66](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/PerformanceTester.java#L66) executed in 0.27 seconds: 
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
    	Evaluation performance: 0.012217s +- 0.000935s [0.011286s - 0.013947s]
    	Learning performance: 0.027703s +- 0.001321s [0.025524s - 0.029442s]
    
```

