# AssertDimensionsLayer
## AssertDimensionsLayerTest
### Batch Execution
Code from [BatchingTester.java:66](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/BatchingTester.java#L66) executed in 0.00 seconds: 
```java
    return test(reference, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (80#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (80#)}
```



Code from [SingleDerivativeTester.java:77](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/SingleDerivativeTester.java#L77) executed in 0.00 seconds: 
```java
    return test(component, inputPrototype);
```
Logging: 
```
    Inputs: [ [ 1.804, 1.516 ], [ -1.604, 0.54 ] ]
    Inputs Statistics: {meanExponent=0.09363346456826775, negative=1, min=0.54, max=0.54, mean=0.5640000000000001, count=4.0, positive=3, stdDev=1.3364789560632822, zeros=0}
    Output: [ [ 1.804, 1.516 ], [ -1.604, 0.54 ] ]
    Outputs Statistics: {meanExponent=0.09363346456826775, negative=1, min=0.54, max=0.54, mean=0.5640000000000001, count=4.0, positive=3, stdDev=1.3364789560632822, zeros=0}
    Feedback for input 0
    Inputs Values: [ [ 1.804, 1.516 ], [ -1.604, 0.54 ] ]
    Value Statistics: {meanExponent=0.09363346456826775, negative=1, min=0.54, max=0.54, mean=0.5640000000000001, count=4.0, positive=3, stdDev=1.3364789560632822, zeros=0}
    Implemented Feedback: [ [ 1.0, 0.0, 0.0, 0.0 ], [ 0.0, 1.0, 0.0, 0.0 ], [ 0.0, 0.0, 1.0, 0.0 ], [ 0.0, 0.0, 0.0, 1.0 ] ]
    Implemented Statistics: {meanExponent=0.0, negative=0, min=1.0, max=1.0, mean=0.25, count=16.0, positive=4, stdDev=0.4330127018922193, zeros=12}
    Measured Feedback: [ [ 0.9999999999998899, 0.0, 0.0, 0.0 ], [ 0.0, 0.9999999999998899, 0.0, 0.0 ], [ 0.0, 0.0, 0.9999999999998899, 0.0 ], [ 0.0, 0.0, 0.0, 0.9999999999998899 ] ]
    Measured Statistics: {meanExponent=-4.7830642341045674E-14, negative=0, min=0.9999999999998899, max=0.9999999999998899, mean=0.24999999999997247, count=16.0, positive=4, stdDev=0.4330127018921716, zeros=12}
    Feedback Error: [ [ -1.1013412404281553E-13, 0.0, 0.0, 0.0 ], [ 0.0, -1.1013412404281553E-13, 0.0, 0.0 ], [ 0.0, 0.0, -1.1013412404281553E-13, 0.0 ], [ 0.0, 0.0, 0.0, -1.1013412404281553E-13 ] ]
    Error Statistics: {meanExponent=-12.958078098036825, negative=4, min=-1.1013412404281553E-13, max=-1.1013412404281553E-13, mean=-2.7533531010703882E-14, count=16.0, positive=0, stdDev=4.7689474622312385E-14, zeros=12}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 2.7534e-14 +- 4.7689e-14 [0.0000e+00 - 1.1013e-13] (16#)
    relativeTol: 5.5067e-14 +- 0.0000e+00 [5.5067e-14 - 5.5067e-14] (4#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=2.7534e-14 +- 4.7689e-14 [0.0000e+00 - 1.1013e-13] (16#), relativeTol=5.5067e-14 +- 0.0000e+00 [5.5067e-14 - 5.5067e-14] (4#)}
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
      "class": "com.simiacryptus.mindseye.layers.java.AssertDimensionsLayer",
      "id": "da6cee95-45bd-48b5-bb5b-a55995bb5b5c",
      "isFrozen": false,
      "name": "AssertDimensionsLayer/da6cee95-45bd-48b5-bb5b-a55995bb5b5c",
      "dims": [
        2,
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
    [[ [ -0.944, 1.004 ], [ 0.156, 1.836 ] ]]
    --------------------
    Output: 
    [ [ -0.944, 1.004 ], [ 0.156, 1.836 ] ]
    --------------------
    Derivative: 
    [ [ 1.0, 1.0 ], [ 1.0, 1.0 ] ]
```



### Input Learning
In this test, we use a network to learn this target input, given it's pre-evaluated output:

Code from [LearningTester.java:127](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L127) executed in 0.00 seconds: 
```java
    return Arrays.stream(input_target).map(x -> x.prettyPrint()).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [ [ 0.22, -1.492 ], [ -0.684, -1.848 ] ]
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
    F(0.0) = LineSearchPoint{point=PointSample{avg=0.6774480000000002}, derivative=-0.6774480000000002}
    New Minimum: 0.6774480000000002 > 0.6774479999322552
    F(1.0E-10) = LineSearchPoint{point=PointSample{avg=0.6774479999322552}, derivative=-0.6774479999661277}, delta = -6.774492078420735E-11
    New Minimum: 0.6774479999322552 > 0.6774479995257866
    F(7.000000000000001E-10) = LineSearchPoint{point=PointSample{avg=0.6774479995257866}, derivative=-0.6774479997628933}, delta = -4.742135573110318E-10
    New Minimum: 0.6774479995257866 > 0.6774479966805048
    F(4.900000000000001E-9) = LineSearchPoint{point=PointSample{avg=0.6774479966805048}, derivative=-0.6774479983402525}, delta = -3.319495345266432E-9
    New Minimum: 0.6774479966805048 > 0.677447976763534
    F(3.430000000000001E-8) = LineSearchPoint{point=PointSample{avg=0.677447976763534}, derivative=-0.677447988381767}, delta = -2.3236466195619698E-8
    New Minimum: 0.677447976763534 > 0.677447837344745
    F(2.4010000000000004E-7) = Lin
```
...[skipping 1577 bytes](etc/186.txt)...
```
    23857191645005
    F(1.3841287201) = LineSearchPoint{point=PointSample{avg=0.06423857191645005}, derivative=-0.20861038341284754}, delta = -0.6132094280835501
    Loops = 12
    New Minimum: 0.06423857191645005 > 1.5407439555097887E-32
    F(1.9999999999999998) = LineSearchPoint{point=PointSample{avg=1.5407439555097887E-32}, derivative=-9.692247004977618E-17}, delta = -0.6774480000000002
    Right bracket at 1.9999999999999998
    Converged to right
    Iteration 1 complete. Error: 1.5407439555097887E-32 Total: 239634219351603.7800; Orientation: 0.0001; Line Search: 0.0017
    Zero gradient: 1.2412670766236366E-16
    F(0.0) = LineSearchPoint{point=PointSample{avg=1.5407439555097887E-32}, derivative=-1.5407439555097887E-32}
    New Minimum: 1.5407439555097887E-32 > 0.0
    F(1.9999999999999998) = LineSearchPoint{point=PointSample{avg=0.0}, derivative=0.0}, delta = -1.5407439555097887E-32
    0.0 <= 1.5407439555097887E-32
    Converged to right
    Iteration 2 complete. Error: 0.0 Total: 239634219577876.7800; Orientation: 0.0000; Line Search: 0.0001
    
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
    th(0)=0.6774480000000002;dx=-0.6774480000000002
    New Minimum: 0.6774480000000002 > 0.004039296145607916
    WOLF (strong): th(2.154434690031884)=0.004039296145607916; dx=0.052310735946359925 delta=0.6734087038543922
    END: th(1.077217345015942)=0.144216456063222; dx=-0.31256863202682006 delta=0.5332315439367782
    Iteration 1 complete. Error: 0.004039296145607916 Total: 239634223429644.7800; Orientation: 0.0001; Line Search: 0.0005
    LBFGS Accumulation History: 1 points
    th(0)=0.144216456063222;dx=-0.144216456063222
    New Minimum: 0.144216456063222 > 0.0037102949051327086
    WOLF (strong): th(2.3207944168063896)=0.0037102949051327086; dx=0.02313191695834281 delta=0.14050616115808928
    END: th(1.1603972084031948)=0.025415729262917262; dx=-0.06054226955243959 delta=0.11880072680030473
    Iteration 2 complete. Error: 0.0037102949051327086 Total: 239634224060016.7800; Orientation: 0.0001; Line Search: 0.0004
    LBFGS Accumulation History: 1 points
    t
```
...[skipping 10072 bytes](etc/187.txt)...
```
    ND: th(0.7064254195601859)=2.2186712959340957E-31; dx=-3.143117669239969E-31 delta=2.2494861750442915E-31
    Iteration 21 complete. Error: 3.0814879110195774E-33 Total: 239634230582886.7800; Orientation: 0.0000; Line Search: 0.0002
    LBFGS Accumulation History: 1 points
    th(0)=2.2186712959340957E-31;dx=-2.2186712959340957E-31
    New Minimum: 2.2186712959340957E-31 > 1.5407439555097887E-32
    END: th(1.5219474298207927)=1.5407439555097887E-32; dx=-5.546678239835239E-32 delta=2.064596900383117E-31
    Iteration 22 complete. Error: 1.5407439555097887E-32 Total: 239634230745324.7800; Orientation: 0.0000; Line Search: 0.0001
    LBFGS Accumulation History: 1 points
    th(0)=1.5407439555097887E-32;dx=-1.5407439555097887E-32
    Armijo: th(3.2789363392107815)=1.5407439555097887E-32; dx=1.5407439555097887E-32 delta=0.0
    New Minimum: 1.5407439555097887E-32 > 0.0
    END: th(1.6394681696053908)=0.0; dx=0.0 delta=1.5407439555097887E-32
    Iteration 23 complete. Error: 0.0 Total: 239634231008644.7800; Orientation: 0.0000; Line Search: 0.0002
    
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

![Result](etc/test.112.png)



Code from [LearningTester.java:98](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L98) executed in 0.00 seconds: 
```java
    return TestUtil.compareTime(runs);
```

Returns: 

![Result](etc/test.113.png)



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
    	[2, 2]
    Performance:
    	Evaluation performance: 0.000034s +- 0.000019s [0.000019s - 0.000070s]
    	Learning performance: 0.000034s +- 0.000004s [0.000027s - 0.000037s]
    
```

