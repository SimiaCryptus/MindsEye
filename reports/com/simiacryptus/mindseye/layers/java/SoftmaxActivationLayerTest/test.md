# SoftmaxActivationLayer
## SoftmaxActivationLayerTest
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
    Inputs: [ 0.696, 1.924, -0.124, -0.248 ]
    Inputs Statistics: {meanExponent=-0.3463280816747981, negative=2, min=-0.248, max=-0.248, mean=0.562, count=4.0, positive=2, stdDev=0.8659815240523322, zeros=0}
    Output: [ 0.19069798078200495, 0.6511180252071395, 0.08398942718677169, 0.07419456682408379 ]
    Outputs Statistics: {meanExponent=-0.7778493660601038, negative=0, min=0.07419456682408379, max=0.07419456682408379, mean=0.24999999999999997, count=4.0, positive=4, stdDev=0.23605055615765583, zeros=0}
    Feedback for input 0
    Inputs Values: [ 0.696, 1.924, -0.124, -0.248 ]
    Value Statistics: {meanExponent=-0.3463280816747981, negative=2, min=-0.248, max=-0.248, mean=0.562, count=4.0, positive=2, stdDev=0.8659815240523322, zeros=0}
    Implemented Feedback: [ [ 0.15433226090767102, -0.12416689265776809, -0.01601661417155459, -0.01414875407834831 ], [ -0.12416689265776809, 0.22716334245749434, -0.05468702996812961, -0.048309419831596576 ], [ -0.01601661417155459, -0.05468702996812961, 0.07693520330760965, -0.006231559167
```
...[skipping 684 bytes](etc/334.txt)...
```
    0.06869265799111512 ] ]
    Measured Statistics: {meanExponent=-1.4000249478821076, negative=12, min=0.06869265799111512, max=0.06869265799111512, mean=1.1275702593849246E-13, count=16.0, positive=4, stdDev=0.08962960992544415, zeros=0}
    Feedback Error: [ [ 4.773546828179542E-6, -3.840521172915334E-6, -4.953990550560228E-7, -4.376256287942637E-7 ], [ 1.8764601598664221E-6, -3.43298474267395E-6, 8.264527046400461E-7, 7.300713229935196E-7 ], [ -6.663223305891641E-7, -2.2750864597451126E-6, 3.2006550855917526E-6, -2.5924504626177736E-7 ], [ -6.02475290493315E-7, -2.057088485510339E-6, -2.653493300069226E-7, 2.924913244772842E-6 ] ]
    Error Statistics: {meanExponent=-5.92052258734506, negative=10, min=2.924913244772842E-6, max=2.924913244772842E-6, mean=1.1274987020415406E-13, count=16.0, positive=6, stdDev=2.28647280061963E-6, zeros=0}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 1.7915e-06 +- 1.4207e-06 [2.5925e-07 - 4.7735e-06] (16#)
    relativeTol: 1.6278e-05 +- 5.5296e-06 [7.5563e-06 - 2.1290e-05] (16#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=1.7915e-06 +- 1.4207e-06 [2.5925e-07 - 4.7735e-06] (16#), relativeTol=1.6278e-05 +- 5.5296e-06 [7.5563e-06 - 2.1290e-05] (16#)}
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
      "class": "com.simiacryptus.mindseye.layers.java.SoftmaxActivationLayer",
      "id": "83bec788-ce6a-47c7-9baa-624a212f8ec0",
      "isFrozen": false,
      "name": "SoftmaxActivationLayer/83bec788-ce6a-47c7-9baa-624a212f8ec0"
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
    [[ -1.564, -0.548, -1.272, 0.028 ]]
    --------------------
    Output: 
    [ 0.09985228453360605, 0.27580440563582326, 0.13371249553712924, 0.4906308142934414 ]
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
    [ -0.548, 1.036, -0.528, 1.552 ]
```



First, we use a conjugate gradient descent method, which converges the fastest for purely linear functions.

Code from [LearningTester.java:225](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L225) executed in 0.14 seconds: 
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
    F(0.0) = LineSearchPoint{point=PointSample{avg=0.023885253320659473}, derivative=-0.004006722799932294}
    New Minimum: 0.023885253320659473 > 0.02388525332025881
    F(1.0E-10) = LineSearchPoint{point=PointSample{avg=0.02388525332025881}, derivative=-0.004006722799905396}, delta = -4.0066214235245923E-13
    New Minimum: 0.02388525332025881 > 0.023885253317854755
    F(7.000000000000001E-10) = LineSearchPoint{point=PointSample{avg=0.023885253317854755}, derivative=-0.004006722799743996}, delta = -2.8047182631940615E-12
    New Minimum: 0.023885253317854755 > 0.023885253301026525
    F(4.900000000000001E-9) = LineSearchPoint{point=PointSample{avg=0.023885253301026525}, derivative=-0.004006722798614207}, delta = -1.9632948045078535E-11
    New Minimum: 0.023885253301026525 > 0.023885253183228893
    F(3.430000000000001E-8) = LineSearchPoint{point=PointSample{avg=0.023885253183228893}, derivative=-0.004006722790705676}, delta = -1.3743058080439852E-10
    New Minimum: 0.023885253183228893 > 0.023
```
...[skipping 249607 bytes](etc/335.txt)...
```
    t
    Iteration 193 complete. Error: 7.703719777548943E-34 Total: 239724946145010.0600; Orientation: 0.0000; Line Search: 0.0009
    Zero gradient: 7.83354813827098E-18
    F(0.0) = LineSearchPoint{point=PointSample{avg=7.703719777548943E-34}, derivative=-6.136447643460875E-35}
    F(443.1412691965056) = LineSearchPoint{point=PointSample{avg=4.726232083526277E-31}, derivative=2.1582859635191875E-33}, delta = 4.718528363748728E-31
    F(34.08778993819274) = LineSearchPoint{point=PointSample{avg=3.177784408238939E-33}, derivative=1.240095419877433E-34}, delta = 2.4074124304840448E-33
    F(2.622137687553288) = LineSearchPoint{point=PointSample{avg=7.703719777548943E-34}, derivative=-6.136447643460875E-35}, delta = 0.0
    New Minimum: 7.703719777548943E-34 > 0.0
    F(18.354963812873017) = LineSearchPoint{point=PointSample{avg=0.0}, derivative=0.0}, delta = -7.703719777548943E-34
    0.0 <= 7.703719777548943E-34
    Converged to right
    Iteration 194 complete. Error: 0.0 Total: 239724946719812.0600; Orientation: 0.0000; Line Search: 0.0004
    
```

Returns: 

```
    0.0
```



Training Converged

Next, we run the same optimization using L-BFGS, which is nearly ideal for purely second-order or quadratic functions.

Code from [LearningTester.java:249](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L249) executed in 0.07 seconds: 
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
    th(0)=0.023885253320659473;dx=-0.004006722799932294
    New Minimum: 0.023885253320659473 > 0.0159208895957143
    END: th(2.154434690031884)=0.0159208895957143; dx=-0.0033677695523132763 delta=0.007964363724945172
    Iteration 1 complete. Error: 0.0159208895957143 Total: 239724952172303.0600; Orientation: 0.0001; Line Search: 0.0003
    LBFGS Accumulation History: 1 points
    th(0)=0.0159208895957143;dx=-0.0028321192867031652
    New Minimum: 0.0159208895957143 > 0.005387840871492733
    END: th(4.641588833612779)=0.005387840871492733; dx=-0.0016778036358732417 delta=0.010533048724221568
    Iteration 2 complete. Error: 0.005387840871492733 Total: 239724952429639.0600; Orientation: 0.0000; Line Search: 0.0001
    LBFGS Accumulation History: 1 points
    th(0)=0.005387840871492733;dx=-9.956692613679945E-4
    New Minimum: 0.005387840871492733 > 4.280603233209316E-5
    END: th(10.000000000000002)=4.280603233209316E-5; dx=-8.490326779670565E-5 delta=0.00534503483916
```
...[skipping 125809 bytes](etc/336.txt)...
```
    : 2.104078464243055E-31 > 1.5494106402595312E-31
    END: th(10.786678029047264)=1.5494106402595312E-31; dx=-7.826114893907593E-34 delta=5.546678239835239E-32
    Iteration 249 complete. Error: 1.5494106402595312E-31 Total: 239725021051294.9700; Orientation: 0.0000; Line Search: 0.0004
    LBFGS Accumulation History: 1 points
    th(0)=1.5494106402595312E-31;dx=-7.016789861362726E-34
    New Minimum: 1.5494106402595312E-31 > 1.3640398831122598E-31
    WOLFE (weak): th(23.239193335984176)=1.3640398831122598E-31; dx=-6.548958463269121E-34 delta=1.8537075714727145E-32
    New Minimum: 1.3640398831122598E-31 > 1.2889286152811576E-31
    WOLFE (weak): th(46.47838667196835)=1.2889286152811576E-31; dx=-6.397052522049628E-34 delta=2.6048202497837365E-32
    New Minimum: 1.2889286152811576E-31 > 7.626682579773454E-32
    END: th(139.43516001590507)=7.626682579773454E-32; dx=-4.863344646671094E-34 delta=7.867423822821858E-32
    Iteration 250 complete. Error: 7.626682579773454E-32 Total: 239725021494720.9700; Orientation: 0.0000; Line Search: 0.0004
    
```

Returns: 

```
    7.626682579773454E-32
```



Training Converged

Code from [LearningTester.java:95](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L95) executed in 0.00 seconds: 
```java
    return TestUtil.compare(runs);
```

Returns: 

![Result](etc/test.238.png)



Code from [LearningTester.java:98](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L98) executed in 0.00 seconds: 
```java
    return TestUtil.compareTime(runs);
```

Returns: 

![Result](etc/test.239.png)



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
    	Evaluation performance: 0.000234s +- 0.000059s [0.000186s - 0.000350s]
    	Learning performance: 0.000031s +- 0.000003s [0.000026s - 0.000035s]
    
```

