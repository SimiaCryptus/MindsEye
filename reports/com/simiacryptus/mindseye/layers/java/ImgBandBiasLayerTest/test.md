# ImgBandBiasLayer
## ImgBandBiasLayerTest
### Batch Execution
Code from [BatchingTester.java:66](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/BatchingTester.java#L66) executed in 0.00 seconds: 
```java
    return test(reference, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (240#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (240#)}
```



Code from [SingleDerivativeTester.java:77](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/SingleDerivativeTester.java#L77) executed in 0.00 seconds: 
```java
    return test(component, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ -1.896, 0.468, 0.988 ], [ -1.404, 0.324, 0.172 ] ],
    	[ [ -1.648, -1.868, -1.636 ], [ 1.612, 1.936, -1.628 ] ]
    ]
    Inputs Statistics: {meanExponent=0.020360322096404326, negative=6, min=-1.628, max=-1.628, mean=-0.3816666666666666, count=12.0, positive=6, stdDev=1.3852368830717086, zeros=0}
    Output: [
    	[ [ -2.112, 2.2880000000000003, 1.476 ], [ -1.6199999999999999, 2.144, 0.6599999999999999 ] ],
    	[ [ -1.8639999999999999, -0.04800000000000004, -1.148 ], [ 1.3960000000000001, 3.7560000000000002, -1.14 ] ]
    ]
    Outputs Statistics: {meanExponent=0.08347207577068409, negative=6, min=-1.14, max=-1.14, mean=0.31566666666666676, count=12.0, positive=6, stdDev=1.8367931898344523, zeros=0}
    Feedback for input 0
    Inputs Values: [
    	[ [ -1.896, 0.468, 0.988 ], [ -1.404, 0.324, 0.172 ] ],
    	[ [ -1.648, -1.868, -1.636 ], [ 1.612, 1.936, -1.628 ] ]
    ]
    Value Statistics: {meanExponent=0.020360322096404326, negative=6, min=-1.628, max=-1.628, mean=-0.3816666666666666, count=12.0, positive=6, stdDev=1.3852368830717086,
```
...[skipping 2753 bytes](etc/242.txt)...
```
    99976694, 0.9999999999976694, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ] ]
    Measured Statistics: {meanExponent=-2.0855188676696804E-13, negative=0, min=0.9999999999998899, max=0.9999999999998899, mean=0.3333333333331733, count=36.0, positive=12, stdDev=0.4714045207908053, zeros=24}
    Gradient Error: [ [ 2.1103119252074976E-12, -1.1013412404281553E-13, -1.1013412404281553E-13, -1.1013412404281553E-13, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, -2.3305801732931286E-12, -1.1013412404281553E-13, -2.3305801732931286E-12, -2.3305801732931286E-12, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ] ]
    Error Statistics: {meanExponent=-12.519823832389234, negative=11, min=-1.1013412404281553E-13, max=-1.1013412404281553E-13, mean=-1.6006948852817813E-13, count=36.0, positive=1, stdDev=7.439172552304769E-13, zeros=24}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 1.1092e-13 +- 4.6831e-13 [0.0000e+00 - 2.3306e-12] (180#)
    relativeTol: 4.1596e-13 +- 5.1113e-13 [5.5067e-14 - 1.1653e-12] (24#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=1.1092e-13 +- 4.6831e-13 [0.0000e+00 - 2.3306e-12] (180#), relativeTol=4.1596e-13 +- 5.1113e-13 [5.5067e-14 - 1.1653e-12] (24#)}
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
      "class": "com.simiacryptus.mindseye.layers.java.ImgBandBiasLayer",
      "id": "1cfa2dda-8b23-45b4-b388-288df381d207",
      "isFrozen": false,
      "name": "ImgBandBiasLayer/1cfa2dda-8b23-45b4-b388-288df381d207",
      "bias": [
        -0.216,
        1.82,
        0.488
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
    	[ [ -0.896, 0.012, -0.432 ], [ 0.776, 0.576, -0.7 ] ],
    	[ [ -1.9, 1.192, -1.476 ], [ -1.524, -1.968, 1.18 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ -1.112, 1.832, 0.055999999999999994 ], [ 0.56, 2.396, -0.21199999999999997 ] ],
    	[ [ -2.116, 3.012, -0.988 ], [ -1.74, -0.1479999999999999, 1.668 ] ]
    ]
    --------------------
    Derivative: 
    [
    	[ [ 1.0, 1.0, 1.0 ], [ 1.0, 1.0, 1.0 ] ],
    	[ [ 1.0, 1.0, 1.0 ], [ 1.0, 1.0, 1.0 ] ]
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
    	[ [ -1.824, -1.12, 0.948 ], [ 1.888, -0.16, -1.512 ] ],
    	[ [ 0.42, -0.276, -1.092 ], [ -1.168, 1.444, -1.22 ] ]
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
    F(0.0) = LineSearchPoint{point=PointSample{avg=4.392296000000001}, derivative=-1.4640986666666664}
    New Minimum: 4.392296000000001 > 4.39229599985359
    F(1.0E-10) = LineSearchPoint{point=PointSample{avg=4.39229599985359}, derivative=-1.4640986666422648}, delta = -1.4641088341704744E-10
    New Minimum: 4.39229599985359 > 4.392295998975132
    F(7.000000000000001E-10) = LineSearchPoint{point=PointSample{avg=4.392295998975132}, derivative=-1.464098666495855}, delta = -1.0248690784919745E-9
    New Minimum: 4.392295998975132 > 4.392295992825917
    F(4.900000000000001E-9) = LineSearchPoint{point=PointSample{avg=4.392295992825917}, derivative=-1.464098665470986}, delta = -7.1740835494438215E-9
    New Minimum: 4.392295992825917 > 4.392295949781416
    F(3.430000000000001E-8) = LineSearchPoint{point=PointSample{avg=4.392295949781416}, derivative=-1.4640986582969024}, delta = -5.021858484610675E-8
    New Minimum: 4.392295949781416 > 4.392295648469918
    F(2.4010000000000004E-7) = LineSearchPoint{p
```
...[skipping 1537 bytes](etc/243.txt)...
```
    54382 > 2.5995398409254213
    F(1.3841287201) = LineSearchPoint{point=PointSample{avg=2.5995398409254213}, derivative=-1.126348497734425}, delta = -1.7927561590745795
    Loops = 12
    New Minimum: 2.5995398409254213 > 2.9685000209488594E-31
    F(6.000000000000002) = LineSearchPoint{point=PointSample{avg=2.9685000209488594E-31}, derivative=3.6699038869553783E-16}, delta = -4.392296000000001
    Right bracket at 6.000000000000002
    Converged to right
    Iteration 1 complete. Error: 2.9685000209488594E-31 Total: 239660999806505.0000; Orientation: 0.0000; Line Search: 0.0013
    Zero gradient: 3.14563190310461E-16
    F(0.0) = LineSearchPoint{point=PointSample{avg=2.9685000209488594E-31}, derivative=-9.89500006982953E-32}
    New Minimum: 2.9685000209488594E-31 > 0.0
    F(6.000000000000002) = LineSearchPoint{point=PointSample{avg=0.0}, derivative=0.0}, delta = -2.9685000209488594E-31
    0.0 <= 2.9685000209488594E-31
    Converged to right
    Iteration 2 complete. Error: 0.0 Total: 239661000051016.0000; Orientation: 0.0000; Line Search: 0.0002
    
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
    th(0)=4.392296000000001;dx=-1.4640986666666664
    New Minimum: 4.392296000000001 > 1.804303044779519
    END: th(2.154434690031884)=1.804303044779519; dx=-0.9383811738173176 delta=2.587992955220482
    Iteration 1 complete. Error: 1.804303044779519 Total: 239661002335973.0000; Orientation: 0.0000; Line Search: 0.0002
    LBFGS Accumulation History: 1 points
    th(0)=1.804303044779519;dx=-0.6014343482598397
    New Minimum: 1.804303044779519 > 0.09248460946856431
    END: th(4.641588833612779)=0.09248460946856431; dx=-0.13616585575416443 delta=1.7118184353109547
    Iteration 2 complete. Error: 0.09248460946856431 Total: 239661002595019.0000; Orientation: 0.0000; Line Search: 0.0002
    LBFGS Accumulation History: 1 points
    th(0)=0.09248460946856431;dx=-0.030828203156188092
    New Minimum: 0.09248460946856431 > 0.04110427087491746
    WOLF (strong): th(10.000000000000002)=0.04110427087491746; dx=0.02055213543745873 delta=0.051380338593646845
    New Minimum: 0.0411
```
...[skipping 9283 bytes](etc/244.txt)...
```
    9661010863097.0000; Orientation: 0.0000; Line Search: 0.0003
    LBFGS Accumulation History: 1 points
    th(0)=8.511069610236073E-30;dx=-2.8370232034120238E-30
    New Minimum: 8.511069610236073E-30 > 7.052498665686806E-30
    WOLF (strong): th(11.30280671296297)=7.052498665686806E-30; dx=2.576808668703704E-30 delta=1.458570944549267E-30
    New Minimum: 7.052498665686806E-30 > 2.2597578014143566E-32
    END: th(5.651403356481485)=2.2597578014143566E-32; dx=-1.1778131571008162E-31 delta=8.488472032221929E-30
    Iteration 21 complete. Error: 2.2597578014143566E-32 Total: 239661011304243.0000; Orientation: 0.0000; Line Search: 0.0004
    LBFGS Accumulation History: 1 points
    th(0)=2.2597578014143566E-32;dx=-7.532526004714522E-33
    Armijo: th(12.175579438566336)=2.2597578014143566E-32; dx=7.532526004714522E-33 delta=0.0
    New Minimum: 2.2597578014143566E-32 > 0.0
    END: th(6.087789719283168)=0.0; dx=0.0 delta=2.2597578014143566E-32
    Iteration 22 complete. Error: 0.0 Total: 239661011678420.0000; Orientation: 0.0000; Line Search: 0.0003
    
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

![Result](etc/test.154.png)



Code from [LearningTester.java:98](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L98) executed in 0.08 seconds: 
```java
    return TestUtil.compareTime(runs);
```

Returns: 

![Result](etc/test.155.png)



### Model Learning
In this test, attempt to train a network to emulate a randomized network given an example input/output. The target state is:

Code from [LearningTester.java:176](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L176) executed in 0.00 seconds: 
```java
    return network_target.state().stream().map(Arrays::toString).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [1.82, -0.216, 0.488]
```



First, we use a conjugate gradient descent method, which converges the fastest for purely linear functions.

Code from [LearningTester.java:225](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L225) executed in 0.01 seconds: 
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
    F(0.0) = LineSearchPoint{point=PointSample{avg=2.1383786666666666}, derivative=-2.851171555555556}
    New Minimum: 2.1383786666666666 > 2.13837866638155
    F(1.0E-10) = LineSearchPoint{point=PointSample{avg=2.13837866638155}, derivative=-2.8511715553654775}, delta = -2.851168190431963E-10
    New Minimum: 2.13837866638155 > 2.138378664670847
    F(7.000000000000001E-10) = LineSearchPoint{point=PointSample{avg=2.138378664670847}, derivative=-2.8511715542250093}, delta = -1.9958195096592135E-9
    New Minimum: 2.138378664670847 > 2.1383786526959265
    F(4.900000000000001E-9) = LineSearchPoint{point=PointSample{avg=2.1383786526959265}, derivative=-2.851171546241728}, delta = -1.3970740120328173E-8
    New Minimum: 2.1383786526959265 > 2.138378568871483
    F(3.430000000000001E-8) = LineSearchPoint{point=PointSample{avg=2.138378568871483}, derivative=-2.8511714903587664}, delta = -9.779518350683247E-8
    New Minimum: 2.138378568871483 > 2.138377982100431
    F(2.4010000000000004E-7) = LineSearchPoi
```
...[skipping 4813 bytes](etc/245.txt)...
```
    rivative=-2.139922160430262E-35}, delta = -1.2325951644078308E-32
    Left bracket at 1.8051570983403196
    Converged to left
    Iteration 3 complete. Error: 6.419766481290786E-35 Total: 239661234192396.7800; Orientation: 0.0000; Line Search: 0.0007
    Zero gradient: 4.625929269271485E-18
    F(0.0) = LineSearchPoint{point=PointSample{avg=6.419766481290786E-35}, derivative=-2.139922160430262E-35}
    F(1.8051570983403196) = LineSearchPoint{point=PointSample{avg=6.419766481290786E-35}, derivative=-2.139922160430262E-35}, delta = 0.0
    F(12.636099688382236) = LineSearchPoint{point=PointSample{avg=6.419766481290786E-35}, derivative=2.139922160430262E-35}, delta = 0.0
    6.419766481290786E-35 <= 6.419766481290786E-35
    New Minimum: 6.419766481290786E-35 > 0.0
    F(6.318049844191118) = LineSearchPoint{point=PointSample{avg=0.0}, derivative=0.0}, delta = -6.419766481290786E-35
    Right bracket at 6.318049844191118
    Converged to right
    Iteration 4 complete. Error: 0.0 Total: 239661234998314.7500; Orientation: 0.0000; Line Search: 0.0006
    
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
    th(0)=2.7635306666666666;dx=-3.6847075555555553
    New Minimum: 2.7635306666666666 > 0.5260347013536278
    WOLF (strong): th(2.154434690031884)=0.5260347013536278; dx=1.6076002979854258 delta=2.237495965313039
    New Minimum: 0.5260347013536278 > 0.21954123026053893
    END: th(1.077217345015942)=0.21954123026053893; dx=-1.0385536287850647 delta=2.5439894364061275
    Iteration 1 complete. Error: 0.21954123026053893 Total: 239661239925591.7500; Orientation: 0.0001; Line Search: 0.0005
    LBFGS Accumulation History: 1 points
    th(0)=0.21954123026053893;dx=-0.2927216403473851
    New Minimum: 0.21954123026053893 > 0.06573586207012222
    WOLF (strong): th(2.3207944168063896)=0.06573586207012222; dx=0.1601761920503611 delta=0.1538053681904167
    New Minimum: 0.06573586207012222 > 0.01125320106377984
    END: th(1.1603972084031948)=0.01125320106377984; dx=-0.06627272414851199 delta=0.2082880291967591
    Iteration 2 complete. Error: 0.01125320106377984 Total: 2396
```
...[skipping 9922 bytes](etc/246.txt)...
```
    695392E-30 delta=2.9473147915606002E-31
    New Minimum: 7.0572492928829615E-31 > 5.7777898331617076E-34
    END: th(1.4128508391203713)=5.7777898331617076E-34; dx=-1.1298789007071783E-32 delta=9.9987862946104E-31
    Iteration 21 complete. Error: 5.7777898331617076E-34 Total: 239661252543294.7500; Orientation: 0.0000; Line Search: 0.0006
    LBFGS Accumulation History: 1 points
    th(0)=5.7777898331617076E-34;dx=-1.9259299443872359E-34
    New Minimum: 5.7777898331617076E-34 > 6.419766481290786E-35
    END: th(3.043894859641584)=6.419766481290786E-35; dx=-6.419766481290786E-35 delta=5.135813185032629E-34
    Iteration 22 complete. Error: 6.419766481290786E-35 Total: 239661252823143.7500; Orientation: 0.0000; Line Search: 0.0002
    LBFGS Accumulation History: 1 points
    th(0)=6.419766481290786E-35;dx=-2.139922160430262E-35
    New Minimum: 6.419766481290786E-35 > 0.0
    END: th(6.55787267842156)=0.0; dx=0.0 delta=6.419766481290786E-35
    Iteration 23 complete. Error: 0.0 Total: 239661253090168.7500; Orientation: 0.0000; Line Search: 0.0002
    
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

![Result](etc/test.156.png)



Code from [LearningTester.java:98](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L98) executed in 0.00 seconds: 
```java
    return TestUtil.compareTime(runs);
```

Returns: 

![Result](etc/test.157.png)



### Performance
Now we execute larger-scale runs to benchmark performance:

Code from [PerformanceTester.java:66](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/PerformanceTester.java#L66) executed in 0.01 seconds: 
```java
    test(component, inputPrototype);
```
Logging: 
```
    100 batches
    Input Dimensions:
    	[2, 2, 3]
    Performance:
    	Evaluation performance: 0.000448s +- 0.000136s [0.000318s - 0.000711s]
    	Learning performance: 0.000247s +- 0.000038s [0.000204s - 0.000312s]
    
```

