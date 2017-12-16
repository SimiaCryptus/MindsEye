# SubsampleLayer
## Float
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
    	[ [ -0.704 ], [ 0.684 ] ],
    	[ [ -1.724 ], [ -0.856 ] ]
    ],
    [
    	[ [ 1.956 ], [ 1.988 ] ],
    	[ [ 1.004 ], [ 1.676 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.03709005324298109, negative=3, min=-0.856, max=-0.856, mean=-0.6499999999999999, count=4.0, positive=1, stdDev=0.8628997624289857, zeros=0},
    {meanExponent=0.20394823940403384, negative=0, min=1.676, max=1.676, mean=1.6560000000000001, count=4.0, positive=4, stdDev=0.3955148543354594, zeros=0}
    Output: [
    	[ [ -0.704, 1.956 ], [ 0.684, 1.988 ] ],
    	[ [ -1.724, 1.004 ], [ -0.856, 1.676 ] ]
    ]
    Outputs Statistics: {meanExponent=0.08342909308052637, negative=3, min=1.676, max=1.676, mean=0.503, count=8.0, positive=5, stdDev=1.3341375491305236, zeros=0}
    Feedback for input 0
    Inputs Values: [
    	[ [ -0.704 ], [ 0.684 ] ],
    	[ [ -1.724 ], [ -0.856 ] ]
    ]
    Value Statistics: {meanExponent=-0.03709005324298109, negative=3, min=-0.856, max=-0.856, mean=-0.6499999999999999, count=4.0, positive=1, stdDev=0.8628997624289857, zeros=0}
    Implemented Feedback: [ [ 1.0, 0.0
```
...[skipping 1944 bytes](etc/180.txt)...
```
    998899, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9999999999998899, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9999999999998899 ] ]
    Measured Statistics: {meanExponent=-4.7830642341045674E-14, negative=0, min=0.9999999999998899, max=0.9999999999998899, mean=0.12499999999998623, count=32.0, positive=4, stdDev=0.3307189138830374, zeros=28}
    Feedback Error: [ [ 0.0, 0.0, 0.0, 0.0, -1.1013412404281553E-13, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, -1.1013412404281553E-13, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.1013412404281553E-13, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.1013412404281553E-13 ] ]
    Error Statistics: {meanExponent=-12.958078098036825, negative=4, min=-1.1013412404281553E-13, max=-1.1013412404281553E-13, mean=-1.3766765505351941E-14, count=32.0, positive=0, stdDev=3.6423437884903677E-14, zeros=28}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 1.3767e-14 +- 3.6423e-14 [0.0000e+00 - 1.1013e-13] (64#)
    relativeTol: 5.5067e-14 +- 0.0000e+00 [5.5067e-14 - 5.5067e-14] (8#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=1.3767e-14 +- 3.6423e-14 [0.0000e+00 - 1.1013e-13] (64#), relativeTol=5.5067e-14 +- 0.0000e+00 [5.5067e-14 - 5.5067e-14] (8#)}
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
      "id": "085f7c6b-0d45-4b63-bcac-8fdf5db46cc1",
      "isFrozen": false,
      "name": "SubsampleLayer/085f7c6b-0d45-4b63-bcac-8fdf5db46cc1",
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
    	[ [ 0.376 ], [ -0.224 ] ],
    	[ [ -1.84 ], [ 0.5 ] ]
    ],
    [
    	[ [ -0.264 ], [ -0.324 ] ],
    	[ [ 0.136 ], [ -0.772 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 0.376, -0.264 ], [ -0.224, -0.324 ] ],
    	[ [ -1.84, 0.136 ], [ 0.5, -0.772 ] ]
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
    	[ [ 0.94 ], [ 0.872 ] ],
    	[ [ -1.464 ], [ 1.36 ] ]
    ]
    [
    	[ [ -1.588 ], [ -0.336 ] ],
    	[ [ 1.7 ], [ -1.276 ] ]
    ]
```



First, we use a conjugate gradient descent method, which converges the fastest for purely linear functions.

Code from [LearningTester.java:225](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L225) executed in 0.06 seconds: 
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
    F(0.0) = LineSearchPoint{point=PointSample{avg=2.569896}, derivative=-1.019306}
    New Minimum: 2.569896 > 2.569895999898069
    F(1.0E-10) = LineSearchPoint{point=PointSample{avg=2.569895999898069}, derivative=-1.0193059999490346}, delta = -1.0193090815846517E-10
    New Minimum: 2.569895999898069 > 2.5698959992864854
    F(7.000000000000001E-10) = LineSearchPoint{point=PointSample{avg=2.5698959992864854}, derivative=-1.0193059996432428}, delta = -7.135145807524168E-10
    New Minimum: 2.5698959992864854 > 2.5698959950054006
    F(4.900000000000001E-9) = LineSearchPoint{point=PointSample{avg=2.5698959950054006}, derivative=-1.0193059975027001}, delta = -4.9945994007316585E-9
    New Minimum: 2.5698959950054006 > 2.569895965037804
    F(3.430000000000001E-8) = LineSearchPoint{point=PointSample{avg=2.569895965037804}, derivative=-1.019305982518902}, delta = -3.496219580512161E-8
    New Minimum: 2.569895965037804 > 2.5698957552646444
    F(2.4010000000000004E-7) = LineSearchPoint{point=PointSample{
```
...[skipping 2300 bytes](etc/181.txt)...
```
    =PointSample{avg=1.55059}, derivative=-7.703719777548943E-34}, delta = 0.0
    Left bracket at 1.0
    F(1.5) = LineSearchPoint{point=PointSample{avg=1.55059}, derivative=7.703719777548943E-34}, delta = 0.0
    Right bracket at 1.5
    F(1.25) = LineSearchPoint{point=PointSample{avg=1.55059}, derivative=7.703719777548943E-34}, delta = 0.0
    Right bracket at 1.25
    F(1.125) = LineSearchPoint{point=PointSample{avg=1.55059}, derivative=7.703719777548943E-34}, delta = 0.0
    Right bracket at 1.125
    F(1.0625) = LineSearchPoint{point=PointSample{avg=1.55059}, derivative=7.703719777548943E-34}, delta = 0.0
    Right bracket at 1.0625
    F(1.03125) = LineSearchPoint{point=PointSample{avg=1.55059}, derivative=7.703719777548943E-34}, delta = 0.0
    Right bracket at 1.03125
    F(1.015625) = LineSearchPoint{point=PointSample{avg=1.55059}, derivative=7.703719777548943E-34}, delta = 0.0
    Right bracket at 1.015625
    Converged to left
    Iteration 2 failed, aborting. Error: 1.55059 Total: 239633095965650.9400; Orientation: 0.0002; Line Search: 0.0264
    
```

Returns: 

```
    1.55059
```



This training run resulted in the following regressed input:

Code from [LearningTester.java:144](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L144) executed in 0.00 seconds: 
```java
    return Arrays.stream(input_gd).map(x -> x.prettyPrint()).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [
    	[ [ -0.32400000000000007 ], [ 0.26799999999999996 ] ],
    	[ [ 0.118 ], [ 0.04200000000000004 ] ]
    ]
    [
    	[ [ -0.336 ], [ -1.588 ] ],
    	[ [ 1.7 ], [ -1.276 ] ]
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
    th(0)=2.569896;dx=-1.019306
    New Minimum: 2.569896 > 1.5566676307509872
    WOLF (strong): th(2.154434690031884)=1.5566676307509872; dx=0.07870810307881973 delta=1.0132283692490127
    END: th(1.077217345015942)=1.767581856148337; dx=-0.47029894846059006 delta=0.802314143851663
    Iteration 1 complete. Error: 1.5566676307509872 Total: 239633107285562.9000; Orientation: 0.0001; Line Search: 0.0042
    LBFGS Accumulation History: 1 points
    th(0)=1.767581856148337;dx=-0.216991856148337
    New Minimum: 1.767581856148337 > 1.556172606869562
    WOLF (strong): th(2.3207944168063896)=1.556172606869562; dx=0.034804887972420845 delta=0.2114092492787749
    END: th(1.1603972084031948)=1.5888311717682646; dx=-0.09109348408795807 delta=0.17875068438007236
    Iteration 2 complete. Error: 1.556172606869562 Total: 239633114551371.9000; Orientation: 0.0002; Line Search: 0.0046
    LBFGS Accumulation History: 1 points
    th(0)=1.5888311717682646;dx=-0.03824117176826439
    New
```
...[skipping 4879 bytes](etc/182.txt)...
```
    a=5.995204332975845E-15
    Iteration 12 complete. Error: 1.5505900000000004 Total: 239633164961198.8400; Orientation: 0.0000; Line Search: 0.0034
    LBFGS Accumulation History: 1 points
    th(0)=1.5505900000000004;dx=-2.1820949624185328E-16
    New Minimum: 1.5505900000000004 > 1.5505900000000001
    WOLF (strong): th(3.5065668783071047)=1.5505900000000001; dx=1.643735993878535E-16 delta=2.220446049250313E-16
    END: th(1.7532834391535523)=1.5505900000000001; dx=-2.6917948348216175E-17 delta=2.220446049250313E-16
    Iteration 13 complete. Error: 1.5505900000000001 Total: 239633169091960.8400; Orientation: 0.0000; Line Search: 0.0031
    LBFGS Accumulation History: 1 points
    th(0)=1.5505900000000001;dx=-3.320551835535835E-18
    WOLF (strong): th(3.777334662770819)=1.5505900000000001; dx=2.9508659293174688E-18 delta=0.0
    END: th(1.8886673313854094)=1.5505900000000001; dx=-1.8484307624515655E-19 delta=0.0
    Iteration 14 failed, aborting. Error: 1.5505900000000001 Total: 239633173130389.8400; Orientation: 0.0000; Line Search: 0.0030
    
```

Returns: 

```
    1.5505900000000001
```



This training run resulted in the following regressed input:

Code from [LearningTester.java:154](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L154) executed in 0.00 seconds: 
```java
    return Arrays.stream(input_lbgfs).map(x -> x.prettyPrint()).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [
    	[ [ -0.3240000001145384 ], [ 0.2680000000606851 ] ],
    	[ [ 0.118000000082588 ], [ 0.04200000013242227 ] ]
    ]
    [
    	[ [ -0.336 ], [ -1.588 ] ],
    	[ [ 1.7 ], [ -1.276 ] ]
    ]
```



Code from [LearningTester.java:95](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L95) executed in 0.01 seconds: 
```java
    return TestUtil.compare(runs);
```

Returns: 

![Result](etc/test.106.png)



Code from [LearningTester.java:98](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L98) executed in 0.00 seconds: 
```java
    return TestUtil.compareTime(runs);
```

Returns: 

![Result](etc/test.107.png)



### Performance
Now we execute larger-scale runs to benchmark performance:

Code from [PerformanceTester.java:66](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/PerformanceTester.java#L66) executed in 0.02 seconds: 
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
    	Evaluation performance: 0.001677s +- 0.001884s [0.000333s - 0.005346s]
    	Learning performance: 0.000294s +- 0.000063s [0.000237s - 0.000409s]
    
```

