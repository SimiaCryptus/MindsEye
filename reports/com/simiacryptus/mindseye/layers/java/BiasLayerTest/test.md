# BiasLayer
## BiasLayerTest
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
    Inputs: [ -1.584, -1.26, -0.476 ]
    Inputs Statistics: {meanExponent=-0.007422441636156407, negative=3, min=-0.476, max=-0.476, mean=-1.1066666666666667, count=3.0, positive=0, stdDev=0.46515182706533814, zeros=0}
    Output: [ -1.992, -1.948, 1.068 ]
    Outputs Statistics: {meanExponent=0.20581651310760476, negative=2, min=1.068, max=1.068, mean=-0.9573333333333333, count=3.0, positive=1, stdDev=1.4322395826893706, zeros=0}
    Feedback for input 0
    Inputs Values: [ -1.584, -1.26, -0.476 ]
    Value Statistics: {meanExponent=-0.007422441636156407, negative=3, min=-0.476, max=-0.476, mean=-1.1066666666666667, count=3.0, positive=0, stdDev=0.46515182706533814, zeros=0}
    Implemented Feedback: [ [ 1.0, 0.0, 0.0 ], [ 0.0, 1.0, 0.0 ], [ 0.0, 0.0, 1.0 ] ]
    Implemented Statistics: {meanExponent=0.0, negative=0, min=1.0, max=1.0, mean=0.3333333333333333, count=9.0, positive=3, stdDev=0.4714045207910317, zeros=6}
    Measured Feedback: [ [ 0.9999999999998899, 0.0, 0.0 ], [ 0.0, 0.9999999999998899, 0.0 ], [ 0.0, 0.0, 0.99999999999988
```
...[skipping 719 bytes](etc/195.txt)...
```
    
    Implemented Statistics: {meanExponent=0.0, negative=0, min=1.0, max=1.0, mean=0.3333333333333333, count=9.0, positive=3, stdDev=0.4714045207910317, zeros=6}
    Measured Gradient: [ [ 0.9999999999998899, 0.0, 0.0 ], [ 0.0, 0.9999999999998899, 0.0 ], [ 0.0, 0.0, 0.9999999999998899 ] ]
    Measured Statistics: {meanExponent=-4.7830642341045674E-14, negative=0, min=0.9999999999998899, max=0.9999999999998899, mean=0.3333333333332966, count=9.0, positive=3, stdDev=0.4714045207909798, zeros=6}
    Gradient Error: [ [ -1.1013412404281553E-13, 0.0, 0.0 ], [ 0.0, -1.1013412404281553E-13, 0.0 ], [ 0.0, 0.0, -1.1013412404281553E-13 ] ]
    Error Statistics: {meanExponent=-12.958078098036827, negative=3, min=-1.1013412404281553E-13, max=-1.1013412404281553E-13, mean=-3.671137468093851E-14, count=9.0, positive=0, stdDev=5.1917723967143496E-14, zeros=6}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 3.6711e-14 +- 5.1918e-14 [0.0000e+00 - 1.1013e-13] (18#)
    relativeTol: 5.5067e-14 +- 0.0000e+00 [5.5067e-14 - 5.5067e-14] (6#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=3.6711e-14 +- 5.1918e-14 [0.0000e+00 - 1.1013e-13] (18#), relativeTol=5.5067e-14 +- 0.0000e+00 [5.5067e-14 - 5.5067e-14] (6#)}
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
      "class": "com.simiacryptus.mindseye.layers.java.BiasLayer",
      "id": "f8bcdad4-0df1-4a2b-a891-8393b5b570f7",
      "isFrozen": false,
      "name": "BiasLayer/f8bcdad4-0df1-4a2b-a891-8393b5b570f7",
      "bias": [
        -0.408,
        -0.688,
        1.544
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
    [[ -0.332, -0.02, -0.112 ]]
    --------------------
    Output: 
    [ -0.74, -0.708, 1.432 ]
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
    [ -0.812, 1.08, 1.332 ]
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
    F(0.0) = LineSearchPoint{point=PointSample{avg=0.04233600000000004}, derivative=-0.05644800000000005}
    New Minimum: 0.04233600000000004 > 0.04233599999435528
    F(1.0E-10) = LineSearchPoint{point=PointSample{avg=0.04233599999435528}, derivative=-0.05644799999623687}, delta = -5.644762435252915E-12
    New Minimum: 0.04233599999435528 > 0.0423359999604865
    F(7.000000000000001E-10) = LineSearchPoint{point=PointSample{avg=0.0423359999604865}, derivative=-0.05644799997365768}, delta = -3.9513538274693616E-11
    New Minimum: 0.0423359999604865 > 0.0423359997234049
    F(4.900000000000001E-9) = LineSearchPoint{point=PointSample{avg=0.0423359997234049}, derivative=-0.05644799981560328}, delta = -2.765951426231261E-10
    New Minimum: 0.0423359997234049 > 0.042335998063833695
    F(3.430000000000001E-8) = LineSearchPoint{point=PointSample{avg=0.042335998063833695}, derivative=-0.05644799870922247}, delta = -1.936166345306578E-9
    New Minimum: 0.042335998063833695 > 0.04233598644683632
    F(2.401
```
...[skipping 1111 bytes](etc/196.txt)...
```
    2108518361509036}, derivative=-0.05629614130613765}, delta = -2.2748163849100472E-4
    New Minimum: 0.042108518361509036 > 0.04075649742727142
    F(0.028247524900000005) = LineSearchPoint{point=PointSample{avg=0.04075649742727142}, derivative=-0.05538498914296326}, delta = -0.001579502572728618
    New Minimum: 0.04075649742727142 > 0.031910057929615
    F(0.19773267430000002) = LineSearchPoint{point=PointSample{avg=0.031910057929615}, derivative=-0.049006924000742434}, delta = -0.010425942070385039
    New Minimum: 0.031910057929615 > 2.526265043625774E-4
    F(1.3841287201) = LineSearchPoint{point=PointSample{avg=2.526265043625774E-4}, derivative=-0.004360468005196809}, delta = -0.04208337349563746
    Loops = 12
    New Minimum: 2.526265043625774E-4 > 0.0
    F(1.5000000000000002) = LineSearchPoint{point=PointSample{avg=0.0}, derivative=0.0}, delta = -0.04233600000000004
    Right bracket at 1.5000000000000002
    Converged to right
    Iteration 1 complete. Error: 0.0 Total: 239634781342952.2500; Orientation: 0.0001; Line Search: 0.0453
    
```

Returns: 

```
    0.0
```



Training Converged

Next, we run the same optimization using L-BFGS, which is nearly ideal for purely second-order or quadratic functions.

Code from [LearningTester.java:249](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L249) executed in 0.02 seconds: 
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
    th(0)=0.04233600000000004;dx=-0.05644800000000005
    New Minimum: 0.04233600000000004 > 0.008058606110338302
    WOLF (strong): th(2.154434690031884)=0.008058606110338302; dx=0.024627686255279897 delta=0.034277393889661736
    New Minimum: 0.008058606110338302 > 0.003363269181854638
    END: th(1.077217345015942)=0.003363269181854638; dx=-0.01591015687236011 delta=0.0389727308181454
    Iteration 1 complete. Error: 0.003363269181854638 Total: 239634786371396.2200; Orientation: 0.0001; Line Search: 0.0005
    LBFGS Accumulation History: 1 points
    th(0)=0.003363269181854638;dx=-0.004484358909139517
    New Minimum: 0.003363269181854638 > 0.0010070427262374201
    WOLF (strong): th(2.3207944168063896)=0.0010070427262374201; dx=0.002453824503718477 delta=0.002356226455617218
    New Minimum: 0.0010070427262374201 > 1.7239378812858322E-4
    END: th(1.1603972084031948)=1.7239378812858322E-4; dx=-0.0010152672027105131 delta=0.003190875393726055
    Iteration 2 complete
```
...[skipping 9146 bytes](etc/197.txt)...
```
    801786732.2000; Orientation: 0.0000; Line Search: 0.0003
    LBFGS Accumulation History: 1 points
    th(0)=1.0025107337183691E-30;dx=-1.3366809782911587E-30
    New Minimum: 1.0025107337183691E-30 > 5.259072701473412E-31
    WOLF (strong): th(2.623149071368624)=5.259072701473412E-31; dx=9.641633286034588E-31 delta=4.766034635710279E-31
    New Minimum: 5.259072701473412E-31 > 1.6434602192104412E-32
    END: th(1.311574535684312)=1.6434602192104412E-32; dx=-1.095640146140294E-31 delta=9.860761315262648E-31
    Iteration 20 complete. Error: 1.6434602192104412E-32 Total: 239634802412829.2000; Orientation: 0.0000; Line Search: 0.0005
    LBFGS Accumulation History: 1 points
    th(0)=1.6434602192104412E-32;dx=-2.1912802922805882E-32
    Armijo: th(2.8257016782407427)=1.6434602192104412E-32; dx=2.1912802922805882E-32 delta=0.0
    New Minimum: 1.6434602192104412E-32 > 0.0
    END: th(1.4128508391203713)=0.0; dx=0.0 delta=1.6434602192104412E-32
    Iteration 21 complete. Error: 0.0 Total: 239634803614583.2000; Orientation: 0.0000; Line Search: 0.0008
    
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

![Result](etc/test.116.png)



Code from [LearningTester.java:98](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L98) executed in 0.00 seconds: 
```java
    return TestUtil.compareTime(runs);
```

Returns: 

![Result](etc/test.117.png)



### Model Learning
In this test, attempt to train a network to emulate a randomized network given an example input/output. The target state is:

Code from [LearningTester.java:176](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L176) executed in 0.00 seconds: 
```java
    return network_target.state().stream().map(Arrays::toString).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [-0.408, 1.544, -0.688]
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

Returns: 

```
    0.0
```



This training run resulted in the following configuration:

Code from [LearningTester.java:189](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L189) executed in 0.00 seconds: 
```java
    return network_gd.state().stream().map(Arrays::toString).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [-0.408, 1.544, -0.688]
    [0.9240000000000002, 2.624, -1.5]
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
    th(0)=3.321216000000001;dx=-4.428288
    New Minimum: 3.321216000000001 > 0.6321894262885781
    WOLF (strong): th(2.154434690031884)=0.6321894262885781; dx=1.9320168564346065 delta=2.689026573711423
    New Minimum: 0.6321894262885781 > 0.263845035409167
    END: th(1.077217345015942)=0.263845035409167; dx=-1.2481355717826965 delta=3.057370964590834
    Iteration 1 complete. Error: 0.263845035409167 Total: 239634932068878.0600; Orientation: 0.0001; Line Search: 0.0005
    LBFGS Accumulation History: 1 points
    th(0)=0.263845035409167;dx=-0.351793380545556
    New Minimum: 0.263845035409167 > 0.07900147427870646
    WOLF (strong): th(2.3207944168063896)=0.07900147427870646; dx=0.1925000284141587 delta=0.18484356113046052
    New Minimum: 0.07900147427870646 > 0.013524116766658835
    END: th(1.1603972084031948)=0.013524116766658835; dx=-0.07964667606569861 delta=0.25032091864250816
    Iteration 2 complete. Error: 0.013524116766658835 Total: 239634932662203.0600; O
```
...[skipping 9088 bytes](etc/198.txt)...
```
    940234078.0600; Orientation: 0.0000; Line Search: 0.0003
    LBFGS Accumulation History: 1 points
    th(0)=7.573064690121713E-29;dx=-1.0097419586828951E-28
    New Minimum: 7.573064690121713E-29 > 4.259848888193464E-29
    WOLF (strong): th(2.623149071368624)=4.259848888193464E-29; dx=7.573064690121713E-29 delta=3.3132158019282496E-29
    New Minimum: 4.259848888193464E-29 > 1.1832913578315177E-30
    END: th(1.311574535684312)=1.1832913578315177E-30; dx=-1.2621774483536189E-29 delta=7.454735554338562E-29
    Iteration 20 complete. Error: 1.1832913578315177E-30 Total: 239634940940255.0600; Orientation: 0.0000; Line Search: 0.0006
    LBFGS Accumulation History: 1 points
    th(0)=1.1832913578315177E-30;dx=-1.5777218104420236E-30
    Armijo: th(2.8257016782407427)=1.1832913578315177E-30; dx=1.5777218104420236E-30 delta=0.0
    New Minimum: 1.1832913578315177E-30 > 0.0
    END: th(1.4128508391203713)=0.0; dx=0.0 delta=1.1832913578315177E-30
    Iteration 21 complete. Error: 0.0 Total: 239634941772392.0600; Orientation: 0.0001; Line Search: 0.0004
    
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

![Result](etc/test.118.png)



Code from [LearningTester.java:98](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L98) executed in 0.00 seconds: 
```java
    return TestUtil.compareTime(runs);
```

Returns: 

![Result](etc/test.119.png)



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
    	Evaluation performance: 0.000288s +- 0.000111s [0.000175s - 0.000497s]
    	Learning performance: 0.000181s +- 0.000097s [0.000086s - 0.000367s]
    
```

