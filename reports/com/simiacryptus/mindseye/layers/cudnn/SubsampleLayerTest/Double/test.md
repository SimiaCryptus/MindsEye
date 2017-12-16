# SubsampleLayer
## Double
### Batch Execution
Code from [BatchingTester.java:66](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/BatchingTester.java#L66) executed in 0.01 seconds: 
```java
    return test(reference, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (160#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (160#)}
```



Code from [SingleDerivativeTester.java:77](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/SingleDerivativeTester.java#L77) executed in 0.02 seconds: 
```java
    return test(component, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ -0.112 ], [ -1.44 ] ],
    	[ [ -1.632 ], [ 0.18 ] ]
    ],
    [
    	[ [ 0.864 ], [ -0.94 ] ],
    	[ [ -1.912 ], [ 0.048 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.3311067064283551, negative=3, min=0.18, max=0.18, mean=-0.751, count=4.0, positive=1, stdDev=0.7946640799734187, zeros=0},
    {meanExponent=-0.2819073196514349, negative=2, min=0.048, max=0.048, mean=-0.485, count=4.0, positive=2, stdDev=1.0425022781749689, zeros=0}
    Output: [
    	[ [ -0.112, 0.864 ], [ -1.44, -0.94 ] ],
    	[ [ -1.632, -1.912 ], [ 0.18, 0.048 ] ]
    ]
    Outputs Statistics: {meanExponent=-0.306507013039895, negative=5, min=0.048, max=0.048, mean=-0.6179999999999999, count=8.0, positive=3, stdDev=0.9363973515554175, zeros=0}
    Feedback for input 0
    Inputs Values: [
    	[ [ -0.112 ], [ -1.44 ] ],
    	[ [ -1.632 ], [ 0.18 ] ]
    ]
    Value Statistics: {meanExponent=-0.3311067064283551, negative=3, min=0.18, max=0.18, mean=-0.751, count=4.0, positive=1, stdDev=0.7946640799734187, zeros=0}
    Implemented Feedback: [ [ 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0
```
...[skipping 1884 bytes](etc/177.txt)...
```
    9999999998899, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9999999999998899, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0000000000000286 ] ]
    Measured Statistics: {meanExponent=-3.276302567614995E-14, negative=0, min=1.0000000000000286, max=1.0000000000000286, mean=0.12499999999999056, count=32.0, positive=4, stdDev=0.33071891388304886, zeros=28}
    Feedback Error: [ [ 0.0, 0.0, 0.0, 0.0, -1.1013412404281553E-13, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, -1.1013412404281553E-13, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.1013412404281553E-13, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.864375403532904E-14 ] ]
    Error Statistics: {meanExponent=-13.104301089584563, negative=3, min=2.864375403532904E-14, max=2.864375403532904E-14, mean=-9.429956815409923E-15, count=32.0, positive=1, stdDev=3.2769779210413227E-14, zeros=28}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 1.1220e-14 +- 3.2201e-14 [0.0000e+00 - 1.1013e-13] (64#)
    relativeTol: 4.4881e-14 +- 1.7643e-14 [1.4322e-14 - 5.5067e-14] (8#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=1.1220e-14 +- 3.2201e-14 [0.0000e+00 - 1.1013e-13] (64#), relativeTol=4.4881e-14 +- 1.7643e-14 [1.4322e-14 - 5.5067e-14] (8#)}
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
      "class": "com.simiacryptus.mindseye.layers.cudnn.SubsampleLayer",
      "id": "91b17d4d-dc5b-475d-acb4-37fca9c830e9",
      "isFrozen": false,
      "name": "SubsampleLayer/91b17d4d-dc5b-475d-acb4-37fca9c830e9",
      "maxBands": -1
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
    	[ [ -1.728 ], [ -0.58 ] ],
    	[ [ 1.016 ], [ 0.044 ] ]
    ],
    [
    	[ [ 0.968 ], [ -1.172 ] ],
    	[ [ -0.408 ], [ 0.436 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ -1.728, 0.968 ], [ -0.58, -1.172 ] ],
    	[ [ 1.016, -0.408 ], [ 0.044, 0.436 ] ]
    ]
    --------------------
    Derivative: 
    [
    	[ [ 1.0 ], [ 1.0 ] ],
    	[ [ 1.0 ], [ 1.0 ] ]
    ],
    [
    	[ [ 1.0 ], [ 1.0 ] ],
    	[ [ 1.0 ], [ 1.0 ] ]
    ]
```



[GPU Log](etc/cuda.log)

### Input Learning
In this test, we use a network to learn this target input, given it's pre-evaluated output:

Code from [LearningTester.java:127](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L127) executed in 0.00 seconds: 
```java
    return Arrays.stream(input_target).map(x -> x.prettyPrint()).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [
    	[ [ -0.004 ], [ -1.744 ] ],
    	[ [ -1.792 ], [ 1.476 ] ]
    ]
    [
    	[ [ -0.976 ], [ -1.0 ] ],
    	[ [ 1.604 ], [ -1.044 ] ]
    ]
```



First, we use a conjugate gradient descent method, which converges the fastest for purely linear functions.

Code from [LearningTester.java:225](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L225) executed in 0.03 seconds: 
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
    F(0.0) = LineSearchPoint{point=PointSample{avg=4.986811999999999}, derivative=-3.7754659999999998}
    New Minimum: 4.986811999999999 > 4.986811999622454
    F(1.0E-10) = LineSearchPoint{point=PointSample{avg=4.986811999622454}, derivative=-3.7754659998112268}, delta = -3.7754510628928983E-10
    New Minimum: 4.986811999622454 > 4.986811997357174
    F(7.000000000000001E-10) = LineSearchPoint{point=PointSample{avg=4.986811997357174}, derivative=-3.7754659986785866}, delta = -2.642824625809226E-9
    New Minimum: 4.986811997357174 > 4.9868119815002165
    F(4.900000000000001E-9) = LineSearchPoint{point=PointSample{avg=4.9868119815002165}, derivative=-3.7754659907501087}, delta = -1.8499782150627198E-8
    New Minimum: 4.9868119815002165 > 4.986811870501518
    F(3.430000000000001E-8) = LineSearchPoint{point=PointSample{avg=4.986811870501518}, derivative=-3.7754659352507582}, delta = -1.294984803834609E-7
    New Minimum: 4.986811870501518 > 4.986811093510668
    F(2.4010000000000004E-7) = LineSearch
```
...[skipping 1753 bytes](etc/178.txt)...
```
    00000000000004) = LineSearchPoint{point=PointSample{avg=1.211346}, derivative=6.109557304512237E-16}, delta = -3.775465999999999
    Right bracket at 2.0000000000000004
    Converged to right
    Iteration 1 complete. Error: 1.211346 Total: 239632763127135.2500; Orientation: 0.0001; Line Search: 0.0185
    Zero gradient: 3.284083995358298E-16
    F(0.0) = LineSearchPoint{point=PointSample{avg=1.211346}, derivative=-1.078520768856852E-31}
    F(2.0000000000000004) = LineSearchPoint{point=PointSample{avg=1.211346}, derivative=9.244463733058732E-33}, delta = 0.0
    1.211346 <= 1.211346
    F(1.8421052631578951) = LineSearchPoint{point=PointSample{avg=1.211346}, derivative=-1.8488927466117464E-32}, delta = 0.0
    Left bracket at 1.8421052631578951
    F(1.947368421052632) = LineSearchPoint{point=PointSample{avg=1.211346}, derivative=-1.8488927466117464E-32}, delta = 0.0
    Left bracket at 1.947368421052632
    Converged to right
    Iteration 2 failed, aborting. Error: 1.211346 Total: 239632769921590.2500; Orientation: 0.0000; Line Search: 0.0058
    
```

Returns: 

```
    1.211346
```



This training run resulted in the following regressed input:

Code from [LearningTester.java:144](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L144) executed in 0.00 seconds: 
```java
    return Arrays.stream(input_gd).map(x -> x.prettyPrint()).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [
    	[ [ -0.49 ], [ -1.372 ] ],
    	[ [ -0.09399999999999996 ], [ 0.21599999999999997 ] ]
    ]
    [
    	[ [ -0.976 ], [ -1.0 ] ],
    	[ [ 1.604 ], [ -1.044 ] ]
    ]
```



Next, we run the same optimization using L-BFGS, which is nearly ideal for purely second-order or quadratic functions.

Code from [LearningTester.java:249](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L249) executed in 0.07 seconds: 
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
    th(0)=4.986811999999999;dx=-3.7754659999999998
    New Minimum: 4.986811999999999 > 1.23385728538526
    WOLF (strong): th(2.154434690031884)=1.23385728538526; dx=0.29153146071795794 delta=3.752954714614739
    END: th(1.077217345015942)=2.0150745909873367; dx=-1.741967269641021 delta=2.971737409012662
    Iteration 1 complete. Error: 1.23385728538526 Total: 239632778271457.2200; Orientation: 0.0000; Line Search: 0.0031
    LBFGS Accumulation History: 1 points
    th(0)=2.0150745909873367;dx=-0.8037285909873362
    New Minimum: 2.0150745909873367 > 1.232023738017238
    WOLF (strong): th(2.3207944168063896)=1.232023738017238; dx=0.12891582230820192 delta=0.7830508529700986
    END: th(1.1603972084031948)=1.3529896710970426; dx=-0.3374063843395671 delta=0.662084919890294
    Iteration 2 complete. Error: 1.232023738017238 Total: 239632782790359.2200; Orientation: 0.0000; Line Search: 0.0034
    LBFGS Accumulation History: 1 points
    th(0)=1.3529896710970426;dx=-0.141
```
...[skipping 4966 bytes](etc/179.txt)...
```
    2835566932.1600; Orientation: 0.0001; Line Search: 0.0035
    LBFGS Accumulation History: 1 points
    th(0)=1.211346000000001;dx=-8.082386781293902E-16
    New Minimum: 1.211346000000001 > 1.2113460000000005
    WOLF (strong): th(3.5065668783071047)=1.2113460000000005; dx=6.088328108641791E-16 delta=4.440892098500626E-16
    New Minimum: 1.2113460000000005 > 1.211346
    END: th(1.7532834391535523)=1.211346; dx=-9.970293430267332E-17 delta=8.881784197001252E-16
    Iteration 13 complete. Error: 1.211346 Total: 239632840025419.1600; Orientation: 0.0001; Line Search: 0.0033
    LBFGS Accumulation History: 1 points
    th(0)=1.211346;dx=-1.229918262705537E-17
    WOLF (strong): th(3.777334662770819)=1.211346; dx=1.0929881806395802E-17 delta=0.0
    Armijo: th(1.8886673313854094)=1.2113460000000003; dx=-6.846503276713764E-19 delta=-2.220446049250313E-16
    END: th(0.6295557771284698)=1.211346; dx=-8.427671935554358E-18 delta=0.0
    Iteration 14 failed, aborting. Error: 1.211346 Total: 239632846431448.1600; Orientation: 0.0002; Line Search: 0.0048
    
```

Returns: 

```
    1.211346
```



This training run resulted in the following regressed input:

Code from [LearningTester.java:154](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L154) executed in 0.00 seconds: 
```java
    return Arrays.stream(input_lbgfs).map(x -> x.prettyPrint()).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [
    	[ [ -0.48999999939893635 ], [ -1.3719999964777174 ] ],
    	[ [ -0.09400000204064823 ], [ 0.2159999975165928 ] ]
    ]
    [
    	[ [ -0.976 ], [ -1.0 ] ],
    	[ [ 1.604 ], [ -1.044 ] ]
    ]
```



Code from [LearningTester.java:95](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L95) executed in 0.00 seconds: 
```java
    return TestUtil.compare(runs);
```

Returns: 

![Result](etc/test.104.png)



Code from [LearningTester.java:98](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L98) executed in 0.00 seconds: 
```java
    return TestUtil.compareTime(runs);
```

Returns: 

![Result](etc/test.105.png)



### Performance
Now we execute larger-scale runs to benchmark performance:

Code from [PerformanceTester.java:66](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/PerformanceTester.java#L66) executed in 0.01 seconds: 
```java
    test(component, inputPrototype);
```
Logging: 
```
    100 batches
    Input Dimensions:
    	[2, 2, 1]
    	[2, 2, 1]
    Performance:
    	Evaluation performance: 0.000672s +- 0.000300s [0.000341s - 0.001094s]
    	Learning performance: 0.000321s +- 0.000069s [0.000234s - 0.000416s]
    
```

