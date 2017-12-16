# BiasMetaLayer
## BiasMetaLayerTest
Code from [SingleDerivativeTester.java:77](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/SingleDerivativeTester.java#L77) executed in 0.01 seconds: 
```java
    return test(component, inputPrototype);
```
Logging: 
```
    Inputs: [ -1.508, -1.516, 0.54 ],
    [ 0.496, -0.784, 1.64 ]
    Inputs Statistics: {meanExponent=0.030498100884252848, negative=2, min=0.54, max=0.54, mean=-0.828, count=3.0, positive=1, stdDev=0.9673275901506513, zeros=0},
    {meanExponent=-0.06511947092588875, negative=1, min=1.64, max=1.64, mean=0.4506666666666666, count=3.0, positive=2, stdDev=0.9901129004086127, zeros=0}
    Output: [ -1.012, -2.3, 2.1799999999999997 ]
    Outputs Statistics: {meanExponent=0.23512161404199264, negative=2, min=2.1799999999999997, max=2.1799999999999997, mean=-0.37733333333333335, count=3.0, positive=1, stdDev=1.8832067214078103, zeros=0}
    Feedback for input 0
    Inputs Values: [ -1.508, -1.516, 0.54 ]
    Value Statistics: {meanExponent=0.030498100884252848, negative=2, min=0.54, max=0.54, mean=-0.828, count=3.0, positive=1, stdDev=0.9673275901506513, zeros=0}
    Implemented Feedback: [ [ 1.0, 0.0, 0.0 ], [ 0.0, 1.0, 0.0 ], [ 0.0, 0.0, 1.0 ] ]
    Implemented Statistics: {meanExponent=0.0, negative=0, min=1.0, max=1.0, mean=0.3333333333333333, c
```
...[skipping 1050 bytes](etc/202.txt)...
```
    ] ]
    Implemented Statistics: {meanExponent=0.0, negative=0, min=1.0, max=1.0, mean=0.3333333333333333, count=9.0, positive=3, stdDev=0.4714045207910317, zeros=6}
    Measured Feedback: [ [ 0.9999999999998899, 0.0, 0.0 ], [ 0.0, 0.9999999999976694, 0.0 ], [ 0.0, 0.0, 1.0000000000021103 ] ]
    Measured Statistics: {meanExponent=-4.7830642341759385E-14, negative=0, min=1.0000000000021103, max=1.0000000000021103, mean=0.3333333333332966, count=9.0, positive=3, stdDev=0.4714045207909798, zeros=6}
    Feedback Error: [ [ -1.1013412404281553E-13, 0.0, 0.0 ], [ 0.0, -2.3305801732931286E-12, 0.0 ], [ 0.0, 0.0, 2.1103119252074976E-12 ] ]
    Error Statistics: {meanExponent=-12.088755799140722, negative=2, min=2.1103119252074976E-12, max=2.1103119252074976E-12, mean=-3.671137468093851E-14, count=9.0, positive=1, stdDev=1.0480150744155462E-12, zeros=6}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 5.0567e-13 +- 9.1868e-13 [0.0000e+00 - 2.3306e-12] (18#)
    relativeTol: 7.5850e-13 +- 4.9943e-13 [5.5067e-14 - 1.1653e-12] (6#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=5.0567e-13 +- 9.1868e-13 [0.0000e+00 - 2.3306e-12] (18#), relativeTol=7.5850e-13 +- 4.9943e-13 [5.5067e-14 - 1.1653e-12] (6#)}
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
      "class": "com.simiacryptus.mindseye.layers.java.BiasMetaLayer",
      "id": "123a880a-1e9c-4b9b-a8fe-9739ca4cc1ca",
      "isFrozen": false,
      "name": "BiasMetaLayer/123a880a-1e9c-4b9b-a8fe-9739ca4cc1ca"
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
    [[ 1.864, 1.4, 0.976 ],
    [ 0.244, 0.044, 1.576 ]]
    --------------------
    Output: 
    [ 2.108, 1.444, 2.552 ]
    --------------------
    Derivative: 
    [ 1.0, 1.0, 1.0 ],
    [ 1.0, 1.0, 1.0 ]
```



### Input Learning
In this test, we use a network to learn this target input, given it's pre-evaluated output:

Code from [LearningTester.java:127](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L127) executed in 0.00 seconds: 
```java
    return Arrays.stream(input_target).map(x -> x.prettyPrint()).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [ 0.148, -1.388, -0.268 ]
    [ 1.64, 1.616, 0.44 ]
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
    F(0.0) = LineSearchPoint{point=PointSample{avg=3.9171146666666665}, derivative=-20.891278222222216}
    New Minimum: 3.9171146666666665 > 3.917114664577539
    F(1.0E-10) = LineSearchPoint{point=PointSample{avg=3.917114664577539}, derivative=-20.89127821665121}, delta = -2.08912753763002E-9
    New Minimum: 3.917114664577539 > 3.9171146520427715
    F(7.000000000000001E-10) = LineSearchPoint{point=PointSample{avg=3.9171146520427715}, derivative=-20.891278183225165}, delta = -1.462389498385619E-8
    New Minimum: 3.9171146520427715 > 3.9171145642994034
    F(4.900000000000001E-9) = LineSearchPoint{point=PointSample{avg=3.9171145642994034}, derivative=-20.89127794924285}, delta = -1.0236726311063649E-7
    New Minimum: 3.9171145642994034 > 3.9171139500958567
    F(3.430000000000001E-8) = LineSearchPoint{point=PointSample{avg=3.9171139500958567}, derivative=-20.89127631136664}, delta = -7.165708098000323E-7
    New Minimum: 3.9171139500958567 > 3.9171096506723706
    F(2.4010000000000004E-7) = LineSea
```
...[skipping 1453 bytes](etc/203.txt)...
```
    5752882}, derivative=-9.875576055754621}, delta = -3.0418061880913783
    F(1.3841287201) = LineSearchPoint{point=PointSample{avg=28.365920572053795}, derivative=56.21863694305095}, delta = 24.44880590538713
    Loops = 12
    New Minimum: 0.8753084785752882 > 6.676557140542418E-32
    F(0.375) = LineSearchPoint{point=PointSample{avg=6.676557140542418E-32}, derivative=-2.4415037892645663E-15}, delta = -3.9171146666666665
    Left bracket at 0.375
    Converged to left
    Iteration 1 complete. Error: 6.676557140542418E-32 Total: 239635278774156.7200; Orientation: 0.0001; Line Search: 0.0024
    Zero gradient: 5.967269455082413E-16
    F(0.0) = LineSearchPoint{point=PointSample{avg=6.676557140542418E-32}, derivative=-3.5608304749559557E-31}
    New Minimum: 6.676557140542418E-32 > 0.0
    F(0.375) = LineSearchPoint{point=PointSample{avg=0.0}, derivative=0.0}, delta = -6.676557140542418E-32
    0.0 <= 6.676557140542418E-32
    Converged to right
    Iteration 2 complete. Error: 0.0 Total: 239635279169706.7200; Orientation: 0.0001; Line Search: 0.0002
    
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
    th(0)=3.9171146666666665;dx=-20.891278222222216
    Armijo: th(2.154434690031884)=88.19985176715612; dx=99.13244050061289 delta=-84.28273710048946
    Armijo: th(1.077217345015942)=13.735575311523242; dx=39.12058113919534 delta=-9.818460644856575
    New Minimum: 3.9171146666666665 > 0.007066458199266725
    END: th(0.3590724483386473)=0.007066458199266725; dx=-0.8873251017497011 delta=3.9100482084674
    Iteration 1 complete. Error: 0.007066458199266725 Total: 239635282896083.7200; Orientation: 0.0000; Line Search: 0.0006
    LBFGS Accumulation History: 1 points
    th(0)=0.007066458199266725;dx=-0.03768777706275586
    Armijo: th(0.7735981389354633)=0.007983802630381627; dx=0.04005940746087767 delta=-9.173444311149021E-4
    New Minimum: 0.007066458199266725 > 6.995757954805832E-6
    WOLF (strong): th(0.3867990694677316)=6.995757954805832E-6; dx=0.001185815199060906 delta=0.007059462441311919
    END: th(0.12893302315591054)=0.0030426074617473354; dx=-0.0247299
```
...[skipping 9580 bytes](etc/204.txt)...
```
    3726117.7200; Orientation: 0.0000; Line Search: 0.0004
    LBFGS Accumulation History: 1 points
    th(0)=6.676557140542417E-33;dx=-3.5608304749559555E-32
    New Minimum: 6.676557140542417E-33 > 2.5679065925163146E-33
    WOLF (strong): th(0.6279337062757206)=2.5679065925163146E-33; dx=2.1912802922805882E-32 delta=4.108650548026102E-33
    New Minimum: 2.5679065925163146E-33 > 2.5679065925163143E-34
    END: th(0.3139668531378603)=2.5679065925163143E-34; dx=-6.847750913376838E-33 delta=6.419766481290785E-33
    Iteration 21 complete. Error: 2.5679065925163143E-34 Total: 239635294132781.7200; Orientation: 0.0000; Line Search: 0.0003
    LBFGS Accumulation History: 1 points
    th(0)=2.5679065925163143E-34;dx=-1.3695501826753676E-33
    Armijo: th(0.676421079920352)=2.5679065925163143E-34; dx=1.3695501826753676E-33 delta=0.0
    New Minimum: 2.5679065925163143E-34 > 0.0
    END: th(0.338210539960176)=0.0; dx=0.0 delta=2.5679065925163143E-34
    Iteration 22 complete. Error: 0.0 Total: 239635294535456.7200; Orientation: 0.0000; Line Search: 0.0003
    
```

Returns: 

```
    0.0
```



Training Converged

Code from [LearningTester.java:95](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L95) executed in 0.01 seconds: 
```java
    return TestUtil.compare(runs);
```

Returns: 

![Result](etc/test.122.png)



Code from [LearningTester.java:98](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L98) executed in 0.00 seconds: 
```java
    return TestUtil.compareTime(runs);
```

Returns: 

![Result](etc/test.123.png)



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
    	[3]
    	[3]
    Performance:
    	Evaluation performance: 0.000157s +- 0.000021s [0.000134s - 0.000197s]
    	Learning performance: 0.000043s +- 0.000004s [0.000038s - 0.000049s]
    
```

