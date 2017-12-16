# SumInputsLayer
## N1Test
### Batch Execution
Code from [BatchingTester.java:66](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/BatchingTester.java#L66) executed in 0.00 seconds: 
```java
    return test(reference, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (70#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (70#)}
```



Code from [SingleDerivativeTester.java:77](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/SingleDerivativeTester.java#L77) executed in 0.00 seconds: 
```java
    return test(component, inputPrototype);
```
Logging: 
```
    Inputs: [ 1.724, -1.76, -1.024 ],
    [ -0.248 ]
    Inputs Statistics: {meanExponent=0.16411662864755192, negative=2, min=-1.024, max=-1.024, mean=-0.35333333333333333, count=3.0, positive=1, stdDev=1.4993130278749516, zeros=0},
    {meanExponent=-0.6055483191737837, negative=1, min=-0.248, max=-0.248, mean=-0.248, count=1.0, positive=0, stdDev=0.0, zeros=0}
    Output: [ 1.476, -2.008, -1.272 ]
    Outputs Statistics: {meanExponent=0.19211239242413317, negative=2, min=-1.272, max=-1.272, mean=-0.6013333333333334, count=3.0, positive=1, stdDev=1.4993130278749516, zeros=0}
    Feedback for input 0
    Inputs Values: [ 1.724, -1.76, -1.024 ]
    Value Statistics: {meanExponent=0.16411662864755192, negative=2, min=-1.024, max=-1.024, mean=-0.35333333333333333, count=3.0, positive=1, stdDev=1.4993130278749516, zeros=0}
    Implemented Feedback: [ [ 1.0, 0.0, 0.0 ], [ 0.0, 1.0, 0.0 ], [ 0.0, 0.0, 1.0 ] ]
    Implemented Statistics: {meanExponent=0.0, negative=0, min=1.0, max=1.0, mean=0.3333333333333333, count=9.0, positive=3, stdDev=0.47140452
```
...[skipping 838 bytes](etc/342.txt)...
```
    8, mean=-0.248, count=1.0, positive=0, stdDev=0.0, zeros=0}
    Implemented Feedback: [ [ 1.0, 1.0, 1.0 ] ]
    Implemented Statistics: {meanExponent=0.0, negative=0, min=1.0, max=1.0, mean=1.0, count=3.0, positive=3, stdDev=0.0, zeros=0}
    Measured Feedback: [ [ 0.9999999999998899, 0.9999999999976694, 0.9999999999998899 ] ]
    Measured Statistics: {meanExponent=-3.692731311925336E-13, negative=0, min=0.9999999999998899, max=0.9999999999998899, mean=0.9999999999991497, count=3.0, positive=3, stdDev=1.0536712127723509E-8, zeros=0}
    Feedback Error: [ [ -1.1013412404281553E-13, -2.3305801732931286E-12, -1.1013412404281553E-13 ] ]
    Error Statistics: {meanExponent=-12.516230716189696, negative=3, min=-1.1013412404281553E-13, max=-1.1013412404281553E-13, mean=-8.502828071262533E-13, count=3.0, positive=0, stdDev=1.0467283057891834E-12, zeros=0}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 4.2514e-13 +- 8.5356e-13 [0.0000e+00 - 2.3306e-12] (12#)
    relativeTol: 4.2514e-13 +- 5.2336e-13 [5.5067e-14 - 1.1653e-12] (6#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=4.2514e-13 +- 8.5356e-13 [0.0000e+00 - 2.3306e-12] (12#), relativeTol=4.2514e-13 +- 5.2336e-13 [5.5067e-14 - 1.1653e-12] (6#)}
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
      "id": "9b6aaee6-e9ef-4c9e-a6d7-f4eb62de489c",
      "isFrozen": false,
      "name": "SumInputsLayer/9b6aaee6-e9ef-4c9e-a6d7-f4eb62de489c"
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
    [[ -1.072, 0.888, -1.892 ],
    [ -1.832 ]]
    --------------------
    Output: 
    [ -2.904, -0.9440000000000001, -3.724 ]
    --------------------
    Derivative: 
    [ 1.0, 1.0, 1.0 ],
    [ 3.0 ]
```



### Input Learning
In this test, we use a network to learn this target input, given it's pre-evaluated output:

Code from [LearningTester.java:127](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L127) executed in 0.00 seconds: 
```java
    return Arrays.stream(input_target).map(x -> x.prettyPrint()).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [ 2.0, -0.112, 0.804, 1.968, -0.656, 1.28, -1.624, 0.992, ... ]
    [ -0.912 ]
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
    F(0.0) = LineSearchPoint{point=PointSample{avg=7.480302400000002}, derivative=-1.196848384}
    New Minimum: 7.480302400000002 > 7.480302399880316
    F(1.0E-10) = LineSearchPoint{point=PointSample{avg=7.480302399880316}, derivative=-1.1968483839904251}, delta = -1.1968559476827068E-10
    New Minimum: 7.480302399880316 > 7.480302399162204
    F(7.000000000000001E-10) = LineSearchPoint{point=PointSample{avg=7.480302399162204}, derivative=-1.1968483839329764}, delta = -8.37798275199475E-10
    New Minimum: 7.480302399162204 > 7.480302394135441
    F(4.900000000000001E-9) = LineSearchPoint{point=PointSample{avg=7.480302394135441}, derivative=-1.1968483835308354}, delta = -5.864561281043734E-9
    New Minimum: 7.480302394135441 > 7.480302358948098
    F(3.430000000000001E-8) = LineSearchPoint{point=PointSample{avg=7.480302358948098}, derivative=-1.196848380715848}, delta = -4.105190409831039E-8
    New Minimum: 7.480302358948098 > 7.480302112636706
    F(2.4010000000000004E-7) = LineSearchPoint{point=
```
...[skipping 1512 bytes](etc/343.txt)...
```
    245518153501539 > 5.91542765297335
    F(1.3841287201) = LineSearchPoint{point=PointSample{avg=5.91542765297335}, derivative=-1.0643210062480262}, delta = -1.5648747470266517
    Loops = 12
    New Minimum: 5.91542765297335 > 5.640881251205052E-30
    F(12.50000000000001) = LineSearchPoint{point=PointSample{avg=5.640881251205052E-30}, derivative=1.0356851376513987E-15}, delta = -7.480302400000002
    Right bracket at 12.50000000000001
    Converged to right
    Iteration 1 complete. Error: 5.640881251205052E-30 Total: 239730084669295.9000; Orientation: 0.0000; Line Search: 0.0023
    Zero gradient: 9.500215788037703E-16
    F(0.0) = LineSearchPoint{point=PointSample{avg=5.640881251205052E-30}, derivative=-9.025410001928083E-31}
    New Minimum: 5.640881251205052E-30 > 0.0
    F(12.50000000000001) = LineSearchPoint{point=PointSample{avg=0.0}, derivative=0.0}, delta = -5.640881251205052E-30
    0.0 <= 5.640881251205052E-30
    Converged to right
    Iteration 2 complete. Error: 0.0 Total: 239730085125545.9000; Orientation: 0.0000; Line Search: 0.0003
    
```

Returns: 

```
    0.0
```



Training Converged

Next, we run the same optimization using L-BFGS, which is nearly ideal for purely second-order or quadratic functions.

Code from [LearningTester.java:249](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L249) executed in 0.01 seconds: 
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
    th(0)=7.480302400000002;dx=-1.196848384
    New Minimum: 7.480302400000002 > 5.123981846589876
    END: th(2.154434690031884)=5.123981846589876; dx=-0.990565849824144 delta=2.356320553410126
    Iteration 1 complete. Error: 5.123981846589876 Total: 239730089633048.9000; Orientation: 0.0001; Line Search: 0.0004
    LBFGS Accumulation History: 1 points
    th(0)=5.123981846589876;dx=-0.81983709545438
    New Minimum: 5.123981846589876 > 2.0251493303960486
    END: th(4.641588833612779)=2.0251493303960486; dx=-0.5154093588429732 delta=3.0988325161938275
    Iteration 2 complete. Error: 2.0251493303960486 Total: 239730090075050.9000; Orientation: 0.0000; Line Search: 0.0003
    LBFGS Accumulation History: 1 points
    th(0)=2.0251493303960486;dx=-0.3240238928633679
    New Minimum: 2.0251493303960486 > 0.08100597321584185
    END: th(10.000000000000002)=0.08100597321584185; dx=-0.06480477857267353 delta=1.9441433571802067
    Iteration 3 complete. Error: 0.08100597321584185
```
...[skipping 11730 bytes](etc/344.txt)...
```
    239730101377578.9000; Orientation: 0.0000; Line Search: 0.0003
    LBFGS Accumulation History: 1 points
    th(0)=1.5648180798146292E-36;dx=-2.5037089277034065E-37
    New Minimum: 1.5648180798146292E-36 > 7.297468929904761E-37
    WOLF (strong): th(21.859575594738537)=7.297468929904761E-37; dx=1.6851887013388312E-37 delta=8.350711868241531E-37
    New Minimum: 7.297468929904761E-37 > 3.009265538105056E-38
    END: th(10.929787797369269)=3.009265538105056E-38; dx=-2.8888949165808535E-38 delta=1.5347254244335786E-36
    Iteration 26 complete. Error: 3.009265538105056E-38 Total: 239730101773413.9000; Orientation: 0.0000; Line Search: 0.0003
    LBFGS Accumulation History: 1 points
    th(0)=3.009265538105056E-38;dx=-4.81482486096809E-39
    Armijo: th(23.547513985339528)=3.009265538105056E-38; dx=4.81482486096809E-39 delta=0.0
    New Minimum: 3.009265538105056E-38 > 0.0
    END: th(11.773756992669764)=0.0; dx=0.0 delta=3.009265538105056E-38
    Iteration 27 complete. Error: 0.0 Total: 239730102154144.9000; Orientation: 0.0000; Line Search: 0.0003
    
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

![Result](etc/test.245.png)



Code from [LearningTester.java:98](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L98) executed in 0.00 seconds: 
```java
    return TestUtil.compareTime(runs);
```

Returns: 

![Result](etc/test.246.png)



### Performance
Now we execute larger-scale runs to benchmark performance:

Code from [PerformanceTester.java:66](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/PerformanceTester.java#L66) executed in 0.01 seconds: 
```java
    test(component, inputPrototype);
```
Logging: 
```
    100 batches
    Input Dimensions:
    	[100]
    	[1]
    Performance:
    	Evaluation performance: 0.000530s +- 0.000432s [0.000160s - 0.001144s]
    	Learning performance: 0.000263s +- 0.000008s [0.000251s - 0.000274s]
    
```

