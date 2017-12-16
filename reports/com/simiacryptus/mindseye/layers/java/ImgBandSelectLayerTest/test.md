# ImgBandSelectLayer
## ImgBandSelectLayerTest
### Batch Execution
Code from [BatchingTester.java:66](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/BatchingTester.java#L66) executed in 0.00 seconds: 
```java
    return test(reference, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (200#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (160#)}
```



Code from [SingleDerivativeTester.java:77](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/SingleDerivativeTester.java#L77) executed in 0.03 seconds: 
```java
    return test(component, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ -0.72, 1.172, -0.276 ], [ -1.968, -1.52, -1.752 ] ],
    	[ [ 1.348, -1.304, 0.008 ], [ -0.76, -1.844, -0.036 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.2552077954663708, negative=9, min=-0.036, max=-0.036, mean=-0.6376666666666666, count=12.0, positive=3, stdDev=1.0738745529881764, zeros=0}
    Output: [
    	[ [ -0.72, -0.276 ], [ -1.968, -1.752 ] ],
    	[ [ 1.348, 0.008 ], [ -0.76, -0.036 ] ]
    ]
    Outputs Statistics: {meanExponent=-0.46178790666710073, negative=6, min=-0.036, max=-0.036, mean=-0.5194999999999999, count=8.0, positive=2, stdDev=0.9857817963423753, zeros=0}
    Feedback for input 0
    Inputs Values: [
    	[ [ -0.72, 1.172, -0.276 ], [ -1.968, -1.52, -1.752 ] ],
    	[ [ 1.348, -1.304, 0.008 ], [ -0.76, -1.844, -0.036 ] ]
    ]
    Value Statistics: {meanExponent=-0.2552077954663708, negative=9, min=-0.036, max=-0.036, mean=-0.6376666666666666, count=12.0, positive=3, stdDev=1.0738745529881764, zeros=0}
    Implemented Feedback: [ [ 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ]
```
...[skipping 906 bytes](etc/252.txt)...
```
    nt=-3.464346423592884E-14, negative=0, min=1.0000000000000286, max=1.0000000000000286, mean=0.08333333333332668, count=96.0, positive=8, stdDev=0.27638539919626126, zeros=88}
    Feedback Error: [ [ -1.1013412404281553E-13, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, -1.1013412404281553E-13, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, -1.1013412404281553E-13, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, -1.1013412404281553E-13, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], ... ]
    Error Statistics: {meanExponent=-13.189204332852094, negative=7, min=2.864375403532904E-14, max=2.864375403532904E-14, mean=-6.647460359943125E-15, count=96.0, positive=1, stdDev=2.688545160831E-14, zeros=88}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 7.2442e-15 +- 2.6731e-14 [0.0000e+00 - 1.1013e-13] (96#)
    relativeTol: 4.3465e-14 +- 2.0293e-14 [2.9976e-15 - 5.5067e-14] (8#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=7.2442e-15 +- 2.6731e-14 [0.0000e+00 - 1.1013e-13] (96#), relativeTol=4.3465e-14 +- 2.0293e-14 [2.9976e-15 - 5.5067e-14] (8#)}
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
      "class": "com.simiacryptus.mindseye.layers.java.ImgBandSelectLayer",
      "id": "3e4d518e-52f6-41c2-bec9-181f6ab80013",
      "isFrozen": false,
      "name": "ImgBandSelectLayer/3e4d518e-52f6-41c2-bec9-181f6ab80013",
      "bands": [
        0,
        2
      ]
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
    	[ [ -0.792, 0.268, -1.364 ], [ -1.192, 1.128, -0.224 ] ],
    	[ [ 1.048, -0.216, -0.424 ], [ -1.836, 0.524, 1.504 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ -0.792, -1.364 ], [ -1.192, -0.224 ] ],
    	[ [ 1.048, -0.424 ], [ -1.836, 1.504 ] ]
    ]
    --------------------
    Derivative: 
    [
    	[ [ 1.0, 0.0, 1.0 ], [ 1.0, 0.0, 1.0 ] ],
    	[ [ 1.0, 0.0, 1.0 ], [ 1.0, 0.0, 1.0 ] ]
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
    	[ [ -1.228, 1.004, -1.6 ], [ 1.0, 1.444, -1.78 ] ],
    	[ [ 0.688, -1.796, 0.58 ], [ 1.136, 1.392, 0.012 ] ]
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
    Constructing line search parameters: GD
    F(0.0) = LineSearchPoint{point=PointSample{avg=2.2228060000000003}, derivative=-1.1114030000000001}
    New Minimum: 2.2228060000000003 > 2.22280599988886
    F(1.0E-10) = LineSearchPoint{point=PointSample{avg=2.22280599988886}, derivative=-1.1114029999722148}, delta = -1.1114043019233577E-10
    New Minimum: 2.22280599988886 > 2.222805999222018
    F(7.000000000000001E-10) = LineSearchPoint{point=PointSample{avg=2.222805999222018}, derivative=-1.1114029998055046}, delta = -7.779821231679307E-10
    New Minimum: 2.222805999222018 > 2.2228059945541254
    F(4.900000000000001E-9) = LineSearchPoint{point=PointSample{avg=2.2228059945541254}, derivative=-1.1114029986385314}, delta = -5.445874862175515E-9
    New Minimum: 2.2228059945541254 > 2.2228059618788776
    F(3.430000000000001E-8) = LineSearchPoint{point=PointSample{avg=2.2228059618788776}, derivative=-1.1114029904697194}, delta = -3.8121122702960974E-8
    New Minimum: 2.2228059618788776 > 2.2228057331521476
    F(2.4010000000000004E-7) = LineSe
```
...[skipping 1503 bytes](etc/253.txt)...
```
    -0.21432895386147033
    New Minimum: 2.00847704613853 > 0.9506361322197692
    F(1.3841287201) = LineSearchPoint{point=PointSample{avg=0.9506361322197692}, derivative=-0.726821797023675}, delta = -1.2721698677802311
    Loops = 12
    New Minimum: 0.9506361322197692 > 6.176517516960627E-33
    F(4.0) = LineSearchPoint{point=PointSample{avg=6.176517516960627E-33}, derivative=3.1109663456430074E-17}, delta = -2.2228060000000003
    Right bracket at 4.0
    Converged to right
    Iteration 1 complete. Error: 6.176517516960627E-33 Total: 239661807940243.2000; Orientation: 0.0000; Line Search: 0.0022
    Zero gradient: 5.557210413939996E-17
    F(0.0) = LineSearchPoint{point=PointSample{avg=6.176517516960627E-33}, derivative=-3.088258758480314E-33}
    New Minimum: 6.176517516960627E-33 > 0.0
    F(4.0) = LineSearchPoint{point=PointSample{avg=0.0}, derivative=0.0}, delta = -6.176517516960627E-33
    0.0 <= 6.176517516960627E-33
    Converged to right
    Iteration 2 complete. Error: 0.0 Total: 239661808232346.2000; Orientation: 0.0000; Line Search: 0.0002
    
```

Returns: 

```
    0.0
```



Training Converged

Next, we run the same optimization using L-BFGS, which is nearly ideal for purely second-order or quadratic functions.

Code from [LearningTester.java:249](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L249) executed in 0.01 seconds: 
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
    th(0)=2.2228060000000003;dx=-1.1114030000000001
    New Minimum: 2.2228060000000003 > 0.47319529149996226
    END: th(2.154434690031884)=0.47319529149996226; dx=-0.5127917055486235 delta=1.749610708500038
    Iteration 1 complete. Error: 0.47319529149996226 Total: 239661838833326.1600; Orientation: 0.0001; Line Search: 0.0002
    LBFGS Accumulation History: 1 points
    th(0)=0.47319529149996226;dx=-0.2365976457499811
    New Minimum: 0.47319529149996226 > 0.01217402040732044
    WOLF (strong): th(4.641588833612779)=0.01217402040732044; dx=0.03794960189306498 delta=0.4610212710926418
    END: th(2.3207944168063896)=0.08339272608375568; dx=-0.09932402192845807 delta=0.38980256541620656
    Iteration 2 complete. Error: 0.01217402040732044 Total: 239661839192968.1600; Orientation: 0.0000; Line Search: 0.0003
    LBFGS Accumulation History: 1 points
    th(0)=0.08339272608375568;dx=-0.04169636304187784
    New Minimum: 0.08339272608375568 > 0.005212045380234739
    WOLF (st
```
...[skipping 10958 bytes](etc/254.txt)...
```
    8E-33 Total: 239661846863162.1600; Orientation: 0.0000; Line Search: 0.0002
    LBFGS Accumulation History: 1 points
    th(0)=1.604314690002258E-33;dx=-8.02157345001129E-34
    New Minimum: 1.604314690002258E-33 > 1.578359774736102E-33
    WOLF (strong): th(7.06425419560186)=1.578359774736102E-33; dx=7.94822260251998E-34 delta=2.595491526615611E-35
    New Minimum: 1.578359774736102E-33 > 1.504632769052528E-36
    END: th(3.53212709780093)=1.504632769052528E-36; dx=-4.890056499420716E-36 delta=1.6028100572332055E-33
    Iteration 24 complete. Error: 1.504632769052528E-36 Total: 239661847378403.1600; Orientation: 0.0000; Line Search: 0.0004
    LBFGS Accumulation History: 1 points
    th(0)=1.504632769052528E-36;dx=-7.52316384526264E-37
    Armijo: th(7.609737149103964)=1.504632769052528E-36; dx=7.52316384526264E-37 delta=0.0
    New Minimum: 1.504632769052528E-36 > 0.0
    END: th(3.804868574551982)=0.0; dx=0.0 delta=1.504632769052528E-36
    Iteration 25 complete. Error: 0.0 Total: 239661847762838.1600; Orientation: 0.0000; Line Search: 0.0002
    
```

Returns: 

```
    0.0
```



Training Converged

Code from [LearningTester.java:95](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L95) executed in 0.00 seconds: 
```java
    return TestUtil.compare(runs);
```

Returns: 

![Result](etc/test.162.png)



Code from [LearningTester.java:98](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L98) executed in 0.00 seconds: 
```java
    return TestUtil.compareTime(runs);
```

Returns: 

![Result](etc/test.163.png)



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
    	[2, 2, 3]
    Performance:
    	Evaluation performance: 0.000245s +- 0.000084s [0.000182s - 0.000411s]
    	Learning performance: 0.000055s +- 0.000008s [0.000050s - 0.000072s]
    
```

