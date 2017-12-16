# L1NormalizationLayer
## L1NormalizationLayerTest
### Batch Execution
Code from [BatchingTester.java:66](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/BatchingTester.java#L66) executed in 0.00 seconds: 
```java
    return test(reference, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (80#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (40#)}
```



Code from [SingleDerivativeTester.java:77](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/SingleDerivativeTester.java#L77) executed in 0.00 seconds: 
```java
    return test(component, inputPrototype);
```
Logging: 
```
    Inputs: [ 0.0, -105.2, -11.600000000000001, -165.6 ]
    Inputs Statistics: {meanExponent=1.7685113538311665, negative=3, min=-165.6, max=-165.6, mean=-70.6, count=4.0, positive=0, stdDev=68.3511521482996, zeros=1}
    Output: [ -0.0, 0.3725212464589236, 0.041076487252124656, 0.5864022662889519 ]
    Outputs Statistics: {meanExponent=-0.6823533385485994, negative=0, min=0.5864022662889519, max=0.5864022662889519, mean=0.25, count=4.0, positive=3, stdDev=0.24203665774893626, zeros=1}
    Feedback for input 0
    Inputs Values: [ 0.0, -105.2, -11.600000000000001, -165.6 ]
    Value Statistics: {meanExponent=1.7685113538311665, negative=3, min=-165.6, max=-165.6, mean=-70.6, count=4.0, positive=0, stdDev=68.3511521482996, zeros=1}
    Implemented Feedback: [ [ -0.003541076487252125, 0.001319126226837548, 1.4545498318741028E-4, 0.002076495277227167 ], [ 0.0, -0.0022219502604145765, 1.4545498318741028E-4, 0.002076495277227167 ], [ 0.0, 0.001319126226837548, -0.0033956215040647146, 0.002076495277227167 ], [ 0.0, 0.001319126226837548, 1
```
...[skipping 565 bytes](etc/263.txt)...
```
    , 1.4545503461416143E-4, -0.0014645817292269214 ] ]
    Measured Statistics: {meanExponent=-2.9697275313314426, negative=4, min=-0.0014645817292269214, max=-0.0014645817292269214, mean=-4.5130500898882264E-13, count=16.0, positive=9, stdDev=0.0017566092247263056, zeros=3}
    Feedback Error: [ [ -1.2539227121910468E-9, 4.660404885354491E-10, 5.1426751149653144E-11, 7.345775402886612E-10 ], [ 0.0, -7.877390155106023E-10, 5.1426751149653144E-11, 7.345775402886612E-10 ], [ 0.0, 4.660404885354491E-10, -1.2024915303950057E-9, 7.345775402886612E-10 ], [ 0.0, 4.660404885354491E-10, 5.1426751149653144E-11, -5.192019633237094E-10 ] ]
    Error Statistics: {meanExponent=-9.420998259444813, negative=4, min=-5.192019633237094E-10, max=-5.192019633237094E-10, mean=-4.5130509369211737E-13, count=16.0, positive=9, stdDev=6.218287573978167E-10, zeros=3}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 4.6997e-10 +- 4.0719e-10 [0.0000e+00 - 1.2539e-09] (16#)
    relativeTol: 1.7689e-07 +- 2.0419e-10 [1.7665e-07 - 1.7726e-07] (13#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=4.6997e-10 +- 4.0719e-10 [0.0000e+00 - 1.2539e-09] (16#), relativeTol=1.7689e-07 +- 2.0419e-10 [1.7665e-07 - 1.7726e-07] (13#)}
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
      "class": "com.simiacryptus.mindseye.layers.java.L1NormalizationLayer",
      "id": "f27f7438-3a7b-4c35-bac2-748e59e49366",
      "isFrozen": false,
      "name": "L1NormalizationLayer/f27f7438-3a7b-4c35-bac2-748e59e49366"
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
    [[ 177.6, -56.39999999999999, -99.6, 56.8 ]]
    --------------------
    Output: 
    [ 2.265306122448979, -0.7193877551020407, -1.270408163265306, 0.7244897959183673 ]
    --------------------
    Derivative: 
    [ 0.0, 0.0, 0.0, 0.0 ]
```



### Input Learning
In this test, we use a network to learn this target input, given it's pre-evaluated output:

Code from [LearningTester.java:127](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L127) executed in 0.00 seconds: 
```java
    return Arrays.stream(input_target).map(x -> x.prettyPrint()).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [ -84.39999999999999, -117.19999999999999, 16.0, 14.000000000000002 ]
```



First, we use a conjugate gradient descent method, which converges the fastest for purely linear functions.

Code from [LearningTester.java:225](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L225) executed in 0.05 seconds: 
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
    F(0.0) = LineSearchPoint{point=PointSample{avg=0.23747969202514663}, derivative=-1.5725643624334414E-5}
    New Minimum: 0.23747969202514663 > 0.23747969202514502
    F(1.0E-10) = LineSearchPoint{point=PointSample{avg=0.23747969202514502}, derivative=-1.5725643624334245E-5}, delta = -1.609823385706477E-15
    New Minimum: 0.23747969202514502 > 0.23747969202513564
    F(7.000000000000001E-10) = LineSearchPoint{point=PointSample{avg=0.23747969202513564}, derivative=-1.5725643624333255E-5}, delta = -1.099120794378905E-14
    New Minimum: 0.23747969202513564 > 0.2374796920250696
    F(4.900000000000001E-9) = LineSearchPoint{point=PointSample{avg=0.2374796920250696}, derivative=-1.5725643624326293E-5}, delta = -7.702172233337023E-14
    New Minimum: 0.2374796920250696 > 0.23747969202460723
    F(3.430000000000001E-8) = LineSearchPoint{point=PointSample{avg=0.23747969202460723}, derivative=-1.5725643624277538E-5}, delta = -5.394018565141323E-13
    New Minimum: 0.23747969202460723 > 0.2374796920213709
```
...[skipping 41630 bytes](etc/264.txt)...
```
    e=-9.03178037394403E-39}, delta = 0.0
    Right bracket at 151.3933196877868
    F(75.6966598438934) = LineSearchPoint{point=PointSample{avg=3.274080905458301E-33}, derivative=-9.03178037394403E-39}, delta = 0.0
    Right bracket at 75.6966598438934
    F(37.8483299219467) = LineSearchPoint{point=PointSample{avg=3.274080905458301E-33}, derivative=-9.03178037394403E-39}, delta = 0.0
    Right bracket at 37.8483299219467
    F(18.92416496097335) = LineSearchPoint{point=PointSample{avg=3.274080905458301E-33}, derivative=-9.03178037394403E-39}, delta = 0.0
    Right bracket at 18.92416496097335
    F(9.462082480486675) = LineSearchPoint{point=PointSample{avg=3.274080905458301E-33}, derivative=-9.03178037394403E-39}, delta = 0.0
    Right bracket at 9.462082480486675
    F(4.7310412402433375) = LineSearchPoint{point=PointSample{avg=3.274080905458301E-33}, derivative=-9.03178037394403E-39}, delta = 0.0
    Loops = 12
    Iteration 28 failed, aborting. Error: 3.274080905458301E-33 Total: 239662541840816.4700; Orientation: 0.0000; Line Search: 0.0026
    
```

Returns: 

```
    3.274080905458301E-33
```



Training Converged

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
    th(0)=0.23747969202514663;dx=-1.5725643624334414E-5
    New Minimum: 0.23747969202514663 > 0.23744581600092451
    WOLFE (weak): th(2.154434690031884)=0.23744581600092451; dx=-1.5722071699236205E-5 delta=3.3876024222112555E-5
    New Minimum: 0.23744581600092451 > 0.23741194767106832
    WOLFE (weak): th(4.308869380063768)=0.23741194767106832; dx=-1.5718500807726433E-5 delta=6.774435407830293E-5
    New Minimum: 0.23741194767106832 > 0.23727655125078614
    WOLFE (weak): th(12.926608140191302)=0.23727655125078614; dx=-1.5704227570222473E-5 delta=2.0314077436048295E-4
    New Minimum: 0.23727655125078614 > 0.2366687865909584
    WOLFE (weak): th(51.70643256076521)=0.2366687865909584; dx=-1.5640201950059522E-5 delta=8.109054341882138E-4
    New Minimum: 0.2366687865909584 > 0.23346888806853228
    WOLFE (weak): th(258.53216280382605)=0.23346888806853228; dx=-1.5304290467371392E-5 delta=0.0040108039566143505
    New Minimum: 0.23346888806853228 > 0.21495092927199538
```
...[skipping 216 bytes](etc/265.txt)...
```
    0.0009
    LBFGS Accumulation History: 1 points
    th(0)=0.21495092927199538;dx=-1.1470450980907502E-5
    New Minimum: 0.21495092927199538 > 0.18129299546632355
    END: th(3341.943960201201)=0.18129299546632355; dx=-8.818177364870774E-6 delta=0.033657933805671836
    Iteration 2 complete. Error: 0.18129299546632355 Total: 239662569459781.4400; Orientation: 0.0000; Line Search: 0.0001
    LBFGS Accumulation History: 1 points
    th(0)=0.18129299546632355;dx=-6.886557990257821E-6
    New Minimum: 0.18129299546632355 > 0.1397653727274794
    END: th(7200.000000000001)=0.1397653727274794; dx=-4.801156201273165E-6 delta=0.04152762273884414
    Iteration 3 complete. Error: 0.1397653727274794 Total: 239662569701728.4400; Orientation: 0.0000; Line Search: 0.0002
    LBFGS Accumulation History: 1 points
    th(0)=0.1397653727274794;dx=-3.4958530958087255E-6
    MAX ALPHA: th(0)=0.1397653727274794;th'(0)=-3.4958530958087255E-6;
    Iteration 4 failed, aborting. Error: 0.1397653727274794 Total: 239662569941395.4400; Orientation: 0.0000; Line Search: 0.0002
    
```

Returns: 

```
    0.1397653727274794
```



This training run resulted in the following regressed input:

Code from [LearningTester.java:154](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L154) executed in 0.00 seconds: 
```java
    return Arrays.stream(input_lbgfs).map(x -> x.prettyPrint()).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [ -13.826772764419346, -101.96894367396162, 5.834224927616071, -105.88325487286029 ]
```



Code from [LearningTester.java:95](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L95) executed in 0.02 seconds: 
```java
    return TestUtil.compare(runs);
```

Returns: 

![Result](etc/test.170.png)



Code from [LearningTester.java:98](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L98) executed in 0.00 seconds: 
```java
    return TestUtil.compareTime(runs);
```

Returns: 

![Result](etc/test.171.png)



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
    	[4]
    Performance:
    	Evaluation performance: 0.000073s +- 0.000016s [0.000060s - 0.000101s]
    	Learning performance: 0.000029s +- 0.000006s [0.000025s - 0.000040s]
    
```

