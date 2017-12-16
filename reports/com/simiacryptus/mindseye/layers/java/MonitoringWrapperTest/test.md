# MonitoringWrapperLayer
## MonitoringWrapperTest
### Batch Execution
Code from [BatchingTester.java:66](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/BatchingTester.java#L66) executed in 0.00 seconds: 
```java
    return test(reference, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (60#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (60#)}
```



Code from [SingleDerivativeTester.java:77](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/SingleDerivativeTester.java#L77) executed in 0.00 seconds: 
```java
    return test(component, inputPrototype);
```
Logging: 
```
    Inputs: [ -1.44, 1.24, -1.344 ]
    Inputs Statistics: {meanExponent=0.12672781532509705, negative=2, min=-1.344, max=-1.344, mean=-0.5146666666666667, count=3.0, positive=1, stdDev=1.2413555314878792, zeros=0}
    Output: [ -1.44, 1.24, -1.344 ]
    Outputs Statistics: {meanExponent=0.12672781532509705, negative=2, min=-1.344, max=-1.344, mean=-0.5146666666666667, count=3.0, positive=1, stdDev=1.2413555314878792, zeros=0}
    Feedback for input 0
    Inputs Values: [ -1.44, 1.24, -1.344 ]
    Value Statistics: {meanExponent=0.12672781532509705, negative=2, min=-1.344, max=-1.344, mean=-0.5146666666666667, count=3.0, positive=1, stdDev=1.2413555314878792, zeros=0}
    Implemented Feedback: [ [ 1.0, 0.0, 0.0 ], [ 0.0, 1.0, 0.0 ], [ 0.0, 0.0, 1.0 ] ]
    Implemented Statistics: {meanExponent=0.0, negative=0, min=1.0, max=1.0, mean=0.3333333333333333, count=9.0, positive=3, stdDev=0.4714045207910317, zeros=6}
    Measured Feedback: [ [ 0.9999999999998899, 0.0, 0.0 ], [ 0.0, 0.9999999999998899, 0.0 ], [ 0.0, 0.0, 0.9999999999998899 ] ]
    Measured Statistics: {meanExponent=-4.7830642341045674E-14, negative=0, min=0.9999999999998899, max=0.9999999999998899, mean=0.3333333333332966, count=9.0, positive=3, stdDev=0.4714045207909798, zeros=6}
    Feedback Error: [ [ -1.1013412404281553E-13, 0.0, 0.0 ], [ 0.0, -1.1013412404281553E-13, 0.0 ], [ 0.0, 0.0, -1.1013412404281553E-13 ] ]
    Error Statistics: {meanExponent=-12.958078098036827, negative=3, min=-1.1013412404281553E-13, max=-1.1013412404281553E-13, mean=-3.671137468093851E-14, count=9.0, positive=0, stdDev=5.1917723967143496E-14, zeros=6}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 3.6711e-14 +- 5.1918e-14 [0.0000e+00 - 1.1013e-13] (9#)
    relativeTol: 5.5067e-14 +- 0.0000e+00 [5.5067e-14 - 5.5067e-14] (3#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=3.6711e-14 +- 5.1918e-14 [0.0000e+00 - 1.1013e-13] (9#), relativeTol=5.5067e-14 +- 0.0000e+00 [5.5067e-14 - 5.5067e-14] (3#)}
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
      "class": "com.simiacryptus.mindseye.layers.java.MonitoringWrapperLayer",
      "id": "210d50e4-f53b-4541-b8a8-9376e5b10941",
      "isFrozen": false,
      "name": "MonitoringSynapse/adc883d0-1813-4e40-9d50-b11b16e72a58",
      "inner": {
        "class": "com.simiacryptus.mindseye.layers.java.MonitoringSynapse",
        "id": "adc883d0-1813-4e40-9d50-b11b16e72a58",
        "isFrozen": false,
        "name": "MonitoringSynapse/adc883d0-1813-4e40-9d50-b11b16e72a58",
        "totalBatches": 0,
        "totalItems": 0
      },
      "totalBatches": 0,
      "totalItems": 0,
      "recordSignalMetrics": true
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
    [[ -0.46, -1.328, -0.416 ]]
    --------------------
    Output: 
    [ -0.46, -1.328, -0.416 ]
    --------------------
    Derivative: 
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
    [ 1.044, -1.2, 1.692 ]
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
    F(0.0) = LineSearchPoint{point=PointSample{avg=5.575776}, derivative=-7.434367999999999}
    New Minimum: 5.575776 > 5.575775999256563
    F(1.0E-10) = LineSearchPoint{point=PointSample{avg=5.575775999256563}, derivative=-7.434367999504374}, delta = -7.43437311712114E-10
    New Minimum: 5.575775999256563 > 5.575775994795943
    F(7.000000000000001E-10) = LineSearchPoint{point=PointSample{avg=5.575775994795943}, derivative=-7.434367996530628}, delta = -5.204057629271119E-9
    New Minimum: 5.575775994795943 > 5.575775963571597
    F(4.900000000000001E-9) = LineSearchPoint{point=PointSample{avg=5.575775963571597}, derivative=-7.434367975714397}, delta = -3.6428403404897836E-8
    New Minimum: 5.575775963571597 > 5.575775745001178
    F(3.430000000000001E-8) = LineSearchPoint{point=PointSample{avg=5.575775745001178}, derivative=-7.434367830000784}, delta = -2.54998822057928E-7
    New Minimum: 5.575775745001178 > 5.575774215008386
    F(2.4010000000000004E-7) = LineSearchPoint{point=PointSample{avg=5
```
...[skipping 931 bytes](etc/293.txt)...
```
    
    F(0.004035360700000001) = LineSearchPoint{point=PointSample{avg=5.545815997629938}, derivative=-7.414367762362307}, delta = -0.029960002370062355
    New Minimum: 5.545815997629938 > 5.367750855041604
    F(0.028247524900000005) = LineSearchPoint{point=PointSample{avg=5.367750855041604}, derivative=-7.294366336536156}, delta = -0.20802514495839652
    New Minimum: 5.367750855041604 > 4.202648695260697
    F(0.19773267430000002) = LineSearchPoint{point=PointSample{avg=4.202648695260697}, derivative=-6.454356355753105}, delta = -1.3731273047393033
    New Minimum: 4.202648695260697 > 0.03327165532853238
    F(1.3841287201) = LineSearchPoint{point=PointSample{avg=0.03327165532853238}, derivative=-0.5742864902717348}, delta = -5.5425043446714675
    Loops = 12
    New Minimum: 0.03327165532853238 > 0.0
    F(1.5) = LineSearchPoint{point=PointSample{avg=0.0}, derivative=0.0}, delta = -5.575776
    Right bracket at 1.5
    Converged to right
    Iteration 1 complete. Error: 0.0 Total: 239674087186565.9000; Orientation: 0.0000; Line Search: 0.0014
    
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
    th(0)=5.575776;dx=-7.434367999999999
    New Minimum: 5.575776 > 1.0613421802597671
    WOLF (strong): th(2.154434690031884)=1.0613421802597671; dx=3.24353887844197 delta=4.514433819740233
    New Minimum: 1.0613421802597671 > 0.4429524656492028
    END: th(1.077217345015942)=0.4429524656492028; dx=-2.0954145607790142 delta=5.1328235343507975
    Iteration 1 complete. Error: 0.4429524656492028 Total: 239674091062271.9000; Orientation: 0.0001; Line Search: 0.0003
    LBFGS Accumulation History: 1 points
    th(0)=0.4429524656492028;dx=-0.5906032875322703
    New Minimum: 0.4429524656492028 > 0.13263049565214305
    WOLF (strong): th(2.3207944168063896)=0.13263049565214305; dx=0.32317592063599077 delta=0.31032196999705974
    New Minimum: 0.13263049565214305 > 0.022704770086839934
    END: th(1.1603972084031948)=0.022704770086839934; dx=-0.13371368344813986 delta=0.4202476955623629
    Iteration 2 complete. Error: 0.022704770086839934 Total: 239674091414504.9000; Orien
```
...[skipping 9149 bytes](etc/294.txt)...
```
    : 1 points
    th(0)=1.3045787220092483E-28;dx=-1.739438296012331E-28
    New Minimum: 1.3045787220092483E-28 > 7.26080724847173E-29
    WOLF (strong): th(2.623149071368624)=7.26080724847173E-29; dx=1.2976761890885644E-28 delta=5.784979971620753E-29
    New Minimum: 7.26080724847173E-29 > 2.1036290805893647E-30
    END: th(1.311574535684312)=2.1036290805893647E-30; dx=-2.208810534618833E-29 delta=1.2835424312033547E-28
    Iteration 20 complete. Error: 2.1036290805893647E-30 Total: 239674097936235.9000; Orientation: 0.0000; Line Search: 0.0004
    LBFGS Accumulation History: 1 points
    th(0)=2.1036290805893647E-30;dx=-2.804838774119153E-30
    New Minimum: 2.1036290805893647E-30 > 1.6105910148262323E-30
    WOLF (strong): th(2.8257016782407427)=1.6105910148262323E-30; dx=2.4542339273542586E-30 delta=4.930380657631324E-31
    New Minimum: 1.6105910148262323E-30 > 0.0
    END: th(1.4128508391203713)=0.0; dx=0.0 delta=2.1036290805893647E-30
    Iteration 21 complete. Error: 0.0 Total: 239674098328650.9000; Orientation: 0.0000; Line Search: 0.0003
    
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

![Result](etc/test.198.png)



Code from [LearningTester.java:98](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L98) executed in 0.00 seconds: 
```java
    return TestUtil.compareTime(runs);
```

Returns: 

![Result](etc/test.199.png)



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
    Performance:
    	Evaluation performance: 0.000326s +- 0.000046s [0.000292s - 0.000414s]
    	Learning performance: 0.000288s +- 0.000013s [0.000271s - 0.000303s]
    
```

