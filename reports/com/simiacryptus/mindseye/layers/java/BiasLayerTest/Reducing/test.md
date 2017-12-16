# BiasLayer
## Reducing
### Batch Execution
Code from [BatchingTester.java:66](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/BatchingTester.java#L66) executed in 0.00 seconds: 
```java
    return test(reference, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (60#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (60#)}
```



Code from [SingleDerivativeTester.java:77](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/SingleDerivativeTester.java#L77) executed in 0.00 seconds: 
```java
    return test(component, inputPrototype);
```
Logging: 
```
    Inputs: [ -1.324, 0.328, -0.744 ]
    Inputs Statistics: {meanExponent=-0.1635550785462537, negative=2, min=-0.744, max=-0.744, mean=-0.58, count=3.0, positive=1, stdDev=0.6843235102396137, zeros=0}
    Output: [ -0.788, 0.8640000000000001, -0.20799999999999996 ]
    Outputs Statistics: {meanExponent=-0.28296556835626324, negative=2, min=-0.20799999999999996, max=-0.20799999999999996, mean=-0.04399999999999996, count=3.0, positive=1, stdDev=0.6843235102396138, zeros=0}
    Feedback for input 0
    Inputs Values: [ -1.324, 0.328, -0.744 ]
    Value Statistics: {meanExponent=-0.1635550785462537, negative=2, min=-0.744, max=-0.744, mean=-0.58, count=3.0, positive=1, stdDev=0.6843235102396137, zeros=0}
    Implemented Feedback: [ [ 1.0, 0.0, 0.0 ], [ 0.0, 1.0, 0.0 ], [ 0.0, 0.0, 1.0 ] ]
    Implemented Statistics: {meanExponent=0.0, negative=0, min=1.0, max=1.0, mean=0.3333333333333333, count=9.0, positive=3, stdDev=0.4714045207910317, zeros=6}
    Measured Feedback: [ [ 0.9999999999998899, 0.0, 0.0 ], [ 0.0, 0.9999999999998899, 0.0 ], [ 0
```
...[skipping 551 bytes](etc/199.txt)...
```
    tdDev=5.1917723967143496E-14, zeros=6}
    Learning Gradient for weight set 0
    Weights: [ 0.536 ]
    Implemented Gradient: [ [ 1.0, 1.0, 1.0 ] ]
    Implemented Statistics: {meanExponent=0.0, negative=0, min=1.0, max=1.0, mean=1.0, count=3.0, positive=3, stdDev=0.0, zeros=0}
    Measured Gradient: [ [ 0.9999999999998899, 0.9999999999998899, 0.9999999999998899 ] ]
    Measured Statistics: {meanExponent=-4.7830642341045674E-14, negative=0, min=0.9999999999998899, max=0.9999999999998899, mean=0.9999999999998899, count=3.0, positive=3, stdDev=0.0, zeros=0}
    Gradient Error: [ [ -1.1013412404281553E-13, -1.1013412404281553E-13, -1.1013412404281553E-13 ] ]
    Error Statistics: {meanExponent=-12.958078098036827, negative=3, min=-1.1013412404281553E-13, max=-1.1013412404281553E-13, mean=-1.1013412404281553E-13, count=3.0, positive=0, stdDev=0.0, zeros=0}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 5.5067e-14 +- 5.5067e-14 [0.0000e+00 - 1.1013e-13] (12#)
    relativeTol: 5.5067e-14 +- 0.0000e+00 [5.5067e-14 - 5.5067e-14] (6#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=5.5067e-14 +- 5.5067e-14 [0.0000e+00 - 1.1013e-13] (12#), relativeTol=5.5067e-14 +- 0.0000e+00 [5.5067e-14 - 5.5067e-14] (6#)}
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
      "class": "com.simiacryptus.mindseye.layers.java.BiasLayer",
      "id": "b1d477f4-6e33-42be-8423-ef3f6f569ab7",
      "isFrozen": false,
      "name": "BiasLayer/b1d477f4-6e33-42be-8423-ef3f6f569ab7",
      "bias": [
        0.536
      ]
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
    [[ 1.568, 0.596, 1.048 ]]
    --------------------
    Output: 
    [ 2.104, 1.1320000000000001, 1.584 ]
    --------------------
    Derivative: 
    [ 1.0, 1.0, 1.0 ]
```



### Input Learning
In this test, we use a network to learn this target input, given it's pre-evaluated output:

Code from [LearningTester.java:127](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L127) executed in 0.00 seconds: 
```java
    return Arrays.stream(input_target).map(x -> x.prettyPrint()).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [ 0.276, 1.592, 1.268 ]
```



First, we use a conjugate gradient descent method, which converges the fastest for purely linear functions.

Code from [LearningTester.java:225](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L225) executed in 0.00 seconds: 
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
    F(0.0) = LineSearchPoint{point=PointSample{avg=0.9402986666666667}, derivative=-1.2537315555555555}
    New Minimum: 0.9402986666666667 > 0.9402986665412936
    F(1.0E-10) = LineSearchPoint{point=PointSample{avg=0.9402986665412936}, derivative=-1.2537315554719732}, delta = -1.253731563011229E-10
    New Minimum: 0.9402986665412936 > 0.9402986657890547
    F(7.000000000000001E-10) = LineSearchPoint{point=PointSample{avg=0.9402986657890547}, derivative=-1.2537315549704806}, delta = -8.776119830855578E-10
    New Minimum: 0.9402986657890547 > 0.940298660523382
    F(4.900000000000001E-9) = LineSearchPoint{point=PointSample{avg=0.940298660523382}, derivative=-1.2537315514600322}, delta = -6.143284769777324E-9
    New Minimum: 0.940298660523382 > 0.9402986236636749
    F(3.430000000000001E-8) = LineSearchPoint{point=PointSample{avg=0.9402986236636749}, derivative=-1.253731526886894}, delta = -4.3002991834129034E-8
    New Minimum: 0.9402986236636749 > 0.9402983656457443
    F(2.4010000000000004E-7) = Li
```
...[skipping 1013 bytes](etc/200.txt)...
```
     = LineSearchPoint{point=PointSample{avg=0.9352462129307387}, derivative=-1.2503587161904632}, delta = -0.0050524537359279975
    New Minimum: 0.9352462129307387 > 0.9052173136070175
    F(0.028247524900000005) = LineSearchPoint{point=PointSample{avg=0.9052173136070175}, derivative=-1.230121679999908}, delta = -0.03508135305964921
    New Minimum: 0.9052173136070175 > 0.7087345267496471
    F(0.19773267430000002) = LineSearchPoint{point=PointSample{avg=0.7087345267496471}, derivative=-1.088462426666023}, delta = -0.23156413991701963
    New Minimum: 0.7087345267496471 > 0.005610930773261317
    F(1.3841287201) = LineSearchPoint{point=PointSample{avg=0.005610930773261317}, derivative=-0.09684765332882671}, delta = -0.9346877358934054
    Loops = 12
    New Minimum: 0.005610930773261317 > 0.0
    F(1.5) = LineSearchPoint{point=PointSample{avg=0.0}, derivative=0.0}, delta = -0.9402986666666667
    Right bracket at 1.5
    Converged to right
    Iteration 1 complete. Error: 0.0 Total: 239635098310735.9000; Orientation: 0.0001; Line Search: 0.0019
    
```

Returns: 

```
    0.0
```



Training Converged

Next, we run the same optimization using L-BFGS, which is nearly ideal for purely second-order or quadratic functions.

Code from [LearningTester.java:249](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L249) executed in 0.02 seconds: 
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
    th(0)=0.9402986666666667;dx=-1.2537315555555555
    New Minimum: 0.9402986666666667 > 0.17898470759502386
    WOLF (strong): th(2.154434690031884)=0.17898470759502386; dx=0.5469902812954609 delta=0.7613139590716429
    New Minimum: 0.17898470759502386 > 0.07469948807962484
    END: th(1.077217345015942)=0.07469948807962484; dx=-0.35337063713004735 delta=0.8655991785870418
    Iteration 1 complete. Error: 0.07469948807962484 Total: 239635105948726.9000; Orientation: 0.0001; Line Search: 0.0006
    LBFGS Accumulation History: 1 points
    th(0)=0.07469948807962484;dx=-0.09959931743949978
    New Minimum: 0.07469948807962484 > 0.022366802077603082
    WOLF (strong): th(2.3207944168063896)=0.022366802077603082; dx=0.054500375781379125 delta=0.05233268600202176
    New Minimum: 0.022366802077603082 > 0.003828931621289819
    END: th(1.1603972084031948)=0.003828931621289819; dx=-0.02254947082906036 delta=0.07087055645833502
    Iteration 2 complete. Error: 0.00382893162128
```
...[skipping 9136 bytes](etc/201.txt)...
```
     239635120796100.8800; Orientation: 0.0000; Line Search: 0.0002
    LBFGS Accumulation History: 1 points
    th(0)=2.0937683192741023E-29;dx=-2.7916910923654693E-29
    New Minimum: 2.0937683192741023E-29 > 1.093311910829746E-29
    WOLF (strong): th(2.623149071368624)=1.093311910829746E-29; dx=2.0170735090442812E-29 delta=1.0004564084443562E-29
    New Minimum: 1.093311910829746E-29 > 4.272996569947147E-31
    END: th(1.311574535684312)=4.272996569947147E-31; dx=-3.9881301319506706E-30 delta=2.0510383535746307E-29
    Iteration 20 complete. Error: 4.272996569947147E-31 Total: 239635121220718.8800; Orientation: 0.0000; Line Search: 0.0003
    LBFGS Accumulation History: 1 points
    th(0)=4.272996569947147E-31;dx=-5.69732875992953E-31
    Armijo: th(2.8257016782407427)=4.272996569947147E-31; dx=5.69732875992953E-31 delta=0.0
    New Minimum: 4.272996569947147E-31 > 0.0
    END: th(1.4128508391203713)=0.0; dx=0.0 delta=4.272996569947147E-31
    Iteration 21 complete. Error: 0.0 Total: 239635121511396.8800; Orientation: 0.0000; Line Search: 0.0002
    
```

Returns: 

```
    0.0
```



Training Converged

Code from [LearningTester.java:95](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L95) executed in 0.01 seconds: 
```java
    return TestUtil.compare(runs);
```

Returns: 

![Result](etc/test.120.png)



Code from [LearningTester.java:98](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L98) executed in 0.00 seconds: 
```java
    return TestUtil.compareTime(runs);
```

Returns: 

![Result](etc/test.121.png)



### Model Learning
In this test, attempt to train a network to emulate a randomized network given an example input/output. The target state is:

Code from [LearningTester.java:176](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L176) executed in 0.00 seconds: 
```java
    return network_target.state().stream().map(Arrays::toString).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [0.536]
```



First, we use a conjugate gradient descent method, which converges the fastest for purely linear functions.

Code from [LearningTester.java:225](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L225) executed in 0.00 seconds: 
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

Returns: 

```
    0.0
```



This training run resulted in the following configuration:

Code from [LearningTester.java:189](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L189) executed in 0.00 seconds: 
```java
    return network_gd.state().stream().map(Arrays::toString).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [0.536]
    [1.804, 0.812, 2.128]
```



Next, we run the same optimization using L-BFGS, which is nearly ideal for purely second-order or quadratic functions.

Code from [LearningTester.java:249](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L249) executed in 0.00 seconds: 
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

Returns: 

```
    0.0
```



This training run resulted in the following configuration:

Code from [LearningTester.java:203](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L203) executed in 0.00 seconds: 
```java
    return network_lbfgs.state().stream().map(Arrays::toString).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [0.536]
    [1.804, 0.812, 2.128]
```



Code from [LearningTester.java:95](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L95) executed in 0.00 seconds: 
```java
    return TestUtil.compare(runs);
```

Code from [LearningTester.java:98](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L98) executed in 0.00 seconds: 
```java
    return TestUtil.compareTime(runs);
```

### Performance
Now we execute larger-scale runs to benchmark performance:

Code from [PerformanceTester.java:66](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/PerformanceTester.java#L66) executed in 0.00 seconds: 
```java
    test(component, inputPrototype);
```
Logging: 
```
    100 batches
    Input Dimensions:
    	[3]
    Performance:
    	Evaluation performance: 0.000177s +- 0.000046s [0.000130s - 0.000243s]
    	Learning performance: 0.000199s +- 0.000072s [0.000109s - 0.000326s]
    
```

