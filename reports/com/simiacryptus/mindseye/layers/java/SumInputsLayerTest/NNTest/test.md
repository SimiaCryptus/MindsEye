# SumInputsLayer
## NNTest
### Batch Execution
Code from [BatchingTester.java:66](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/BatchingTester.java#L66) executed in 0.00 seconds: 
```java
    return test(reference, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (90#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (90#)}
```



Code from [SingleDerivativeTester.java:77](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/SingleDerivativeTester.java#L77) executed in 0.00 seconds: 
```java
    return test(component, inputPrototype);
```
Logging: 
```
    Inputs: [ 0.152, 0.076, -0.832 ],
    [ -0.048, 1.4, -0.544 ]
    Inputs Statistics: {meanExponent=-0.672406497827904, negative=1, min=-0.832, max=-0.832, mean=-0.20133333333333334, count=3.0, positive=2, stdDev=0.44702672055358045, zeros=0},
    {meanExponent=-0.47901060908266496, negative=2, min=-0.544, max=-0.544, mean=0.26933333333333326, count=3.0, positive=1, stdDev=0.8247461562336754, zeros=0}
    Output: [ 0.104, 1.476, -1.376 ]
    Outputs Statistics: {meanExponent=-0.22508728977156814, negative=1, min=-1.376, max=-1.376, mean=0.06800000000000006, count=3.0, positive=2, stdDev=1.1646023641855905, zeros=0}
    Feedback for input 0
    Inputs Values: [ 0.152, 0.076, -0.832 ]
    Value Statistics: {meanExponent=-0.672406497827904, negative=1, min=-0.832, max=-0.832, mean=-0.20133333333333334, count=3.0, positive=2, stdDev=0.44702672055358045, zeros=0}
    Implemented Feedback: [ [ 1.0, 0.0, 0.0 ], [ 0.0, 1.0, 0.0 ], [ 0.0, 0.0, 1.0 ] ]
    Implemented Statistics: {meanExponent=0.0, negative=0, min=1.0, max=1.0, mean=0.3333333333333333
```
...[skipping 1061 bytes](etc/345.txt)...
```
    ] ]
    Implemented Statistics: {meanExponent=0.0, negative=0, min=1.0, max=1.0, mean=0.3333333333333333, count=9.0, positive=3, stdDev=0.4714045207910317, zeros=6}
    Measured Feedback: [ [ 1.0000000000000286, 0.0, 0.0 ], [ 0.0, 0.9999999999998899, 0.0 ], [ 0.0, 0.0, 0.9999999999976694 ] ]
    Measured Statistics: {meanExponent=-3.491829756393393E-13, negative=0, min=0.9999999999976694, max=0.9999999999976694, mean=0.3333333333330653, count=9.0, positive=3, stdDev=0.47140452079065265, zeros=6}
    Feedback Error: [ [ 2.864375403532904E-14, 0.0, 0.0 ], [ 0.0, -1.1013412404281553E-13, 0.0 ], [ 0.0, 0.0, -2.3305801732931286E-12 ] ]
    Error Statistics: {meanExponent=-12.711194704920013, negative=2, min=-2.3305801732931286E-12, max=-2.3305801732931286E-12, mean=-2.680078381445128E-13, count=9.0, positive=1, stdDev=7.301522004432353E-13, zeros=6}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 2.7890e-13 +- 7.2649e-13 [0.0000e+00 - 2.3306e-12] (18#)
    relativeTol: 4.1835e-13 +- 5.2836e-13 [1.4322e-14 - 1.1653e-12] (6#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=2.7890e-13 +- 7.2649e-13 [0.0000e+00 - 2.3306e-12] (18#), relativeTol=4.1835e-13 +- 5.2836e-13 [1.4322e-14 - 1.1653e-12] (6#)}
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
      "class": "com.simiacryptus.mindseye.layers.java.SumInputsLayer",
      "id": "e0835342-8797-49c6-8c28-984b49fa099b",
      "isFrozen": false,
      "name": "SumInputsLayer/e0835342-8797-49c6-8c28-984b49fa099b"
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
    [[ -1.14, 0.08, 1.548 ],
    [ 0.608, -0.332, 1.368 ]]
    --------------------
    Output: 
    [ -0.5319999999999999, -0.252, 2.9160000000000004 ]
    --------------------
    Derivative: 
    [ 1.0, 1.0, 1.0 ],
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
    [ -0.3, -1.268, -1.328, 0.696, 1.868, 1.18, 1.604, 1.312, ... ]
    [ -0.06, -0.904, -0.944, 0.508, -1.548, 0.72, 0.212, -0.94, ... ]
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
    F(0.0) = LineSearchPoint{point=PointSample{avg=8.608497759999997}, derivative=-1.3773596416}
    New Minimum: 8.608497759999997 > 8.608497759862267
    F(1.0E-10) = LineSearchPoint{point=PointSample{avg=8.608497759862267}, derivative=-1.377359641588981}, delta = -1.3772982754289842E-10
    New Minimum: 8.608497759862267 > 8.60849775903585
    F(7.000000000000001E-10) = LineSearchPoint{point=PointSample{avg=8.60849775903585}, derivative=-1.377359641522868}, delta = -9.641478726507557E-10
    New Minimum: 8.60849775903585 > 8.608497753250937
    F(4.900000000000001E-9) = LineSearchPoint{point=PointSample{avg=8.608497753250937}, derivative=-1.377359641060075}, delta = -6.749059977551042E-9
    New Minimum: 8.608497753250937 > 8.608497712756565
    F(3.430000000000001E-8) = LineSearchPoint{point=PointSample{avg=8.608497712756565}, derivative=-1.3773596378205253}, delta = -4.724343227735517E-8
    New Minimum: 8.608497712756565 > 8.608497429295955
    F(2.4010000000000004E-7) = LineSearchPoint{point=Poi
```
...[skipping 1510 bytes](etc/346.txt)...
```
    846480822 > 6.807605224631447
    F(1.3841287201) = LineSearchPoint{point=PointSample{avg=6.807605224631447}, derivative=-1.2248441985723837}, delta = -1.8008925353685497
    Loops = 12
    New Minimum: 6.807605224631447 > 4.259149775623651E-30
    F(12.499999999999991) = LineSearchPoint{point=PointSample{avg=4.259149775623651E-30}, derivative=-9.599402162052684E-16}, delta = -8.608497759999997
    Right bracket at 12.499999999999991
    Converged to right
    Iteration 1 complete. Error: 4.259149775623651E-30 Total: 239730240313662.7500; Orientation: 0.0000; Line Search: 0.0022
    Zero gradient: 8.255083064995677E-16
    F(0.0) = LineSearchPoint{point=PointSample{avg=4.259149775623651E-30}, derivative=-6.814639640997842E-31}
    New Minimum: 4.259149775623651E-30 > 0.0
    F(12.499999999999991) = LineSearchPoint{point=PointSample{avg=0.0}, derivative=0.0}, delta = -4.259149775623651E-30
    0.0 <= 4.259149775623651E-30
    Converged to right
    Iteration 2 complete. Error: 0.0 Total: 239730240683279.7500; Orientation: 0.0000; Line Search: 0.0002
    
```

Returns: 

```
    0.0
```



Training Converged

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
Logging: 
```
    LBFGS Accumulation History: 1 points
    Constructing line search parameters: GD
    th(0)=8.608497759999997;dx=-1.3773596416
    New Minimum: 8.608497759999997 > 5.896791852779854
    END: th(2.154434690031884)=5.896791852779854; dx=-1.1399651301989662 delta=2.711705907220143
    Iteration 1 complete. Error: 5.896791852779854 Total: 239730244486316.7500; Orientation: 0.0001; Line Search: 0.0003
    LBFGS Accumulation History: 1 points
    th(0)=5.896791852779854;dx=-0.9434866964447769
    New Minimum: 5.896791852779854 > 2.330586725261254
    END: th(4.641588833612779)=2.330586725261254; dx=-0.593144511254354 delta=3.5662051275186
    Iteration 2 complete. Error: 2.330586725261254 Total: 239730244888420.7500; Orientation: 0.0000; Line Search: 0.0002
    LBFGS Accumulation History: 1 points
    th(0)=2.330586725261254;dx=-0.3728938760418005
    New Minimum: 2.330586725261254 > 0.09322346901044998
    END: th(10.000000000000002)=0.09322346901044998; dx=-0.07457877520836004 delta=2.237363256250804
    Iteration 3 complete. Error: 0.09322346901044998 Total
```
...[skipping 518 bytes](etc/347.txt)...
```
    170607797647 Total: 239730245769004.7500; Orientation: 0.0000; Line Search: 0.0003
    LBFGS Accumulation History: 1 points
    th(0)=0.001781170607797647;dx=-2.849872972476235E-4
    New Minimum: 0.001781170607797647 > 0.0013070665187544538
    WOLF (strong): th(23.2079441680639)=0.0013070665187544538; dx=2.4413024540279868E-4 delta=4.741040890431931E-4
    New Minimum: 0.0013070665187544538 > 9.152264754279274E-6
    END: th(11.60397208403195)=9.152264754279274E-6; dx=-2.0428525922412457E-5 delta=0.0017720183430433678
    Iteration 5 complete. Error: 9.152264754279274E-6 Total: 239730246450957.7500; Orientation: 0.0000; Line Search: 0.0004
    LBFGS Accumulation History: 1 points
    th(0)=9.152264754279274E-6;dx=-1.4643623606846844E-6
    Armijo: th(25.000000000000007)=9.152264754279274E-6; dx=1.4643623606846844E-6 delta=0.0
    New Minimum: 9.152264754279274E-6 > 0.0
    END: th(12.500000000000004)=0.0; dx=0.0 delta=9.152264754279274E-6
    Iteration 6 complete. Error: 0.0 Total: 239730246867310.7500; Orientation: 0.0000; Line Search: 0.0003
    
```

Returns: 

```
    0.0
```



Training Converged

Code from [LearningTester.java:95](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L95) executed in 0.00 seconds: 
```java
    return TestUtil.compare(runs);
```

Returns: 

![Result](etc/test.247.png)



Code from [LearningTester.java:98](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L98) executed in 0.00 seconds: 
```java
    return TestUtil.compareTime(runs);
```

Returns: 

![Result](etc/test.248.png)



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
    	[100]
    	[100]
    Performance:
    	Evaluation performance: 0.000162s +- 0.000037s [0.000130s - 0.000222s]
    	Learning performance: 0.000257s +- 0.000016s [0.000241s - 0.000288s]
    
```

