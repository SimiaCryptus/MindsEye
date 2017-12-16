# ImgConcatLayer
## ImgConcatLayerTest
### Batch Execution
Code from [BatchingTester.java:66](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/BatchingTester.java#L66) executed in 0.00 seconds: 
```java
    return test(reference, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (160#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (160#)}
```



Code from [SingleDerivativeTester.java:77](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/SingleDerivativeTester.java#L77) executed in 0.00 seconds: 
```java
    return test(component, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ 1.268 ], [ 0.288 ] ],
    	[ [ -1.888 ], [ -0.024 ] ]
    ],
    [
    	[ [ 0.06 ], [ -0.604 ] ],
    	[ [ -1.256 ], [ 1.572 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.44531875675534976, negative=2, min=-0.024, max=-0.024, mean=-0.08899999999999998, count=4.0, positive=2, stdDev=1.1428258835010694, zeros=0},
    {meanExponent=-0.28634240747266454, negative=2, min=1.572, max=1.572, mean=-0.05699999999999994, count=4.0, positive=2, stdDev=1.049302149049548, zeros=0}
    Output: [
    	[ [ 1.268, 0.06 ], [ 0.288, -0.604 ] ],
    	[ [ -1.888, -1.256 ], [ -0.024, 1.572 ] ]
    ]
    Outputs Statistics: {meanExponent=-0.36583058211400715, negative=4, min=1.572, max=1.572, mean=-0.07300000000000001, count=8.0, positive=4, stdDev=1.0971777431209586, zeros=0}
    Feedback for input 0
    Inputs Values: [
    	[ [ 1.268 ], [ 0.288 ] ],
    	[ [ -1.888 ], [ -0.024 ] ]
    ]
    Value Statistics: {meanExponent=-0.44531875675534976, negative=2, min=-0.024, max=-0.024, mean=-0.08899999999999998, count=4.0, positive=2, stdDev=1.1428258835010694, zeros=0}
    Implemented Fee
```
...[skipping 1959 bytes](etc/255.txt)...
```
    999998899, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9999999999998899, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9999999999998899 ] ]
    Measured Statistics: {meanExponent=-3.276302567614995E-14, negative=0, min=0.9999999999998899, max=0.9999999999998899, mean=0.12499999999999056, count=32.0, positive=4, stdDev=0.33071891388304886, zeros=28}
    Feedback Error: [ [ 0.0, 0.0, 0.0, 0.0, 2.864375403532904E-14, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, -1.1013412404281553E-13, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.1013412404281553E-13, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.1013412404281553E-13 ] ]
    Error Statistics: {meanExponent=-13.104301089584563, negative=3, min=-1.1013412404281553E-13, max=-1.1013412404281553E-13, mean=-9.429956815409923E-15, count=32.0, positive=1, stdDev=3.2769779210413227E-14, zeros=28}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 1.0866e-14 +- 3.2132e-14 [0.0000e+00 - 1.1013e-13] (64#)
    relativeTol: 4.3465e-14 +- 2.0293e-14 [2.9976e-15 - 5.5067e-14] (8#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=1.0866e-14 +- 3.2132e-14 [0.0000e+00 - 1.1013e-13] (64#), relativeTol=4.3465e-14 +- 2.0293e-14 [2.9976e-15 - 5.5067e-14] (8#)}
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
      "class": "com.simiacryptus.mindseye.layers.java.ImgConcatLayer",
      "id": "d3ca71a6-c0d6-4d99-a970-eb7b3360257a",
      "isFrozen": false,
      "name": "ImgConcatLayer/d3ca71a6-c0d6-4d99-a970-eb7b3360257a"
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
    	[ [ 1.04 ], [ 1.124 ] ],
    	[ [ 0.612 ], [ -1.06 ] ]
    ],
    [
    	[ [ 1.948 ], [ 1.08 ] ],
    	[ [ 0.352 ], [ -0.168 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 1.04, 1.948 ], [ 1.124, 1.08 ] ],
    	[ [ 0.612, 0.352 ], [ -1.06, -0.168 ] ]
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



### Input Learning
In this test, we use a network to learn this target input, given it's pre-evaluated output:

Code from [LearningTester.java:127](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L127) executed in 0.00 seconds: 
```java
    return Arrays.stream(input_target).map(x -> x.prettyPrint()).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [
    	[ [ 0.532 ], [ -0.64 ] ],
    	[ [ -1.248 ], [ -1.416 ] ]
    ]
    [
    	[ [ 0.104 ], [ -0.192 ] ],
    	[ [ -0.688 ], [ 1.516 ] ]
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
    F(0.0) = LineSearchPoint{point=PointSample{avg=2.306304}, derivative=-1.725422}
    New Minimum: 2.306304 > 2.306303999827458
    F(1.0E-10) = LineSearchPoint{point=PointSample{avg=2.306303999827458}, derivative=-1.725421999913729}, delta = -1.7254198070304483E-10
    New Minimum: 2.306303999827458 > 2.3063039987922043
    F(7.000000000000001E-10) = LineSearchPoint{point=PointSample{avg=2.3063039987922043}, derivative=-1.7254219993961022}, delta = -1.2077956412781532E-9
    New Minimum: 2.3063039987922043 > 2.306303991545432
    F(4.900000000000001E-9) = LineSearchPoint{point=PointSample{avg=2.306303991545432}, derivative=-1.7254219957727162}, delta = -8.454567712590233E-9
    New Minimum: 2.306303991545432 > 2.306303940818026
    F(3.430000000000001E-8) = LineSearchPoint{point=PointSample{avg=2.306303940818026}, derivative=-1.7254219704090126}, delta = -5.918197398813163E-8
    New Minimum: 2.306303940818026 > 2.3063035857262024
    F(2.4010000000000004E-7) = LineSearchPoint{point=PointSample{avg=
```
...[skipping 3300 bytes](etc/256.txt)...
```
    565E-16
    F(0.1681313679493015) = LineSearchPoint{point=PointSample{avg=0.5808819999999999}, derivative=-3.177784408238939E-32}, delta = 0.0
    F(1.1769195756451105) = LineSearchPoint{point=PointSample{avg=0.580882}, derivative=-8.666684749742561E-33}, delta = 1.1102230246251565E-16
    F(0.09053227504962388) = LineSearchPoint{point=PointSample{avg=0.5808819999999999}, derivative=-3.177784408238939E-32}, delta = 0.0
    F(0.6337259253473672) = LineSearchPoint{point=PointSample{avg=0.5808819999999999}, derivative=-2.3303752327085554E-32}, delta = 0.0
    F(4.43608147743157) = LineSearchPoint{point=PointSample{avg=0.580882}, derivative=3.4088960015654075E-32}, delta = 1.1102230246251565E-16
    Loops = 12
    F(2.1402147478836526) = LineSearchPoint{point=PointSample{avg=0.580882}, derivative=1.9259299443872359E-34}, delta = 1.1102230246251565E-16
    Right bracket at 2.1402147478836526
    Converged to right
    Iteration 2 failed, aborting. Error: 0.5808819999999999 Total: 239662015717133.0000; Orientation: 0.0000; Line Search: 0.0017
    
```

Returns: 

```
    0.5808819999999999
```



This training run resulted in the following regressed input:

Code from [LearningTester.java:144](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L144) executed in 0.00 seconds: 
```java
    return Arrays.stream(input_gd).map(x -> x.prettyPrint()).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [
    	[ [ 0.318 ], [ -0.416 ] ],
    	[ [ -0.968 ], [ 0.05000000000000006 ] ]
    ]
    [
    	[ [ -0.192 ], [ -0.688 ] ],
    	[ [ 1.516 ], [ 0.104 ] ]
    ]
```



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
    th(0)=2.306304;dx=-1.725422
    New Minimum: 2.306304 > 0.5911698603732641
    WOLF (strong): th(2.154434690031884)=0.5911698603732641; dx=0.1332325058720965 delta=1.7151341396267359
    END: th(1.077217345015942)=0.9481932121572678; dx=-0.7960947470639519 delta=1.358110787842732
    Iteration 1 complete. Error: 0.5911698603732641 Total: 239662021979247.9700; Orientation: 0.0001; Line Search: 0.0004
    LBFGS Accumulation History: 1 points
    th(0)=0.9481932121572678;dx=-0.36731121215726786
    New Minimum: 0.9481932121572678 > 0.5903319126955926
    WOLF (strong): th(2.3207944168063896)=0.5903319126955926; dx=0.05891569304521938 delta=0.3578612994616752
    END: th(1.1603972084031948)=0.6456144346906054; dx=-0.1541977595560242 delta=0.3025787774666624
    Iteration 2 complete. Error: 0.5903319126955926 Total: 239662022674594.9700; Orientation: 0.0000; Line Search: 0.0006
    LBFGS Accumulation History: 1 points
    th(0)=0.6456144346906054;dx=-0.06473243469060541
```
...[skipping 4953 bytes](etc/257.txt)...
```
    5808820000000003 Total: 239662027666845.9700; Orientation: 0.0000; Line Search: 0.0004
    LBFGS Accumulation History: 1 points
    th(0)=0.5808820000000003;dx=-3.693723625002769E-16
    New Minimum: 0.5808820000000003 > 0.5808820000000001
    WOLF (strong): th(3.5065668783071047)=0.5808820000000001; dx=2.7824208317721666E-16 delta=2.220446049250313E-16
    New Minimum: 0.5808820000000001 > 0.5808819999999999
    END: th(1.7532834391535523)=0.5808819999999999; dx=-4.5565138884454556E-17 delta=4.440892098500626E-16
    Iteration 13 complete. Error: 0.5808819999999999 Total: 239662028272139.9700; Orientation: 0.0000; Line Search: 0.0005
    LBFGS Accumulation History: 1 points
    th(0)=0.5808819999999999;dx=-5.6208371073189785E-18
    Armijo: th(3.777334662770819)=0.580882; dx=4.995054151420767E-18 delta=-1.1102230246251565E-16
    END: th(1.8886673313854094)=0.5808819999999999; dx=-3.1289144037710334E-19 delta=0.0
    Iteration 14 failed, aborting. Error: 0.5808819999999999 Total: 239662028747768.9700; Orientation: 0.0000; Line Search: 0.0003
    
```

Returns: 

```
    0.5808819999999999
```



This training run resulted in the following regressed input:

Code from [LearningTester.java:154](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L154) executed in 0.00 seconds: 
```java
    return Arrays.stream(input_lbgfs).map(x -> x.prettyPrint()).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [
    	[ [ 0.3179999998426607 ], [ -0.41600000002250576 ] ],
    	[ [ -0.9679999998492919 ], [ 0.04999999985270789 ] ]
    ]
    [
    	[ [ -0.192 ], [ -0.688 ] ],
    	[ [ 1.516 ], [ 0.104 ] ]
    ]
```



Code from [LearningTester.java:95](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L95) executed in 0.00 seconds: 
```java
    return TestUtil.compare(runs);
```

Returns: 

![Result](etc/test.164.png)



Code from [LearningTester.java:98](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L98) executed in 0.00 seconds: 
```java
    return TestUtil.compareTime(runs);
```

Returns: 

![Result](etc/test.165.png)



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
    	[2, 2, 1]
    	[2, 2, 1]
    Performance:
    	Evaluation performance: 0.000105s +- 0.000017s [0.000089s - 0.000133s]
    	Learning performance: 0.000197s +- 0.000008s [0.000187s - 0.000208s]
    
```

