# AvgMetaLayer
## AvgMetaLayerTest
### Differential Validation
Code from [BatchDerivativeTester.java:76](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/BatchDerivativeTester.java#L76) executed in 0.03 seconds: 
```java
    return test(component, inputPrototype);
```
Logging: 
```
    Inputs: [ 1.84, -1.724, -1.856 ],
    [ 1.84, -1.724, -1.856 ],
    [ 1.84, -1.724, -1.856 ],
    [ 1.84, -1.724, -1.856 ],
    [ 1.84, -1.724, -1.856 ],
    [ 1.84, -1.724, -1.856 ],
    [ 1.84, -1.724, -1.856 ],
    [ 1.84, -1.724, -1.856 ],
    [ 1.84, -1.724, -1.856 ],
    [ 1.84, -1.724, -1.856 ]
    Inputs Statistics: {meanExponent=0.2566443521270246, negative=2, min=-1.856, max=-1.856, mean=-0.58, count=3.0, positive=1, stdDev=1.7120467283342473, zeros=0},
    {meanExponent=0.2566443521270246, negative=2, min=-1.856, max=-1.856, mean=-0.58, count=3.0, positive=1, stdDev=1.7120467283342473, zeros=0},
    {meanExponent=0.2566443521270246, negative=2, min=-1.856, max=-1.856, mean=-0.58, count=3.0, positive=1, stdDev=1.7120467283342473, zeros=0},
    {meanExponent=0.2566443521270246, negative=2, min=-1.856, max=-1.856, mean=-0.58, count=3.0, positive=1, stdDev=1.7120467283342473, zeros=0},
    {meanExponent=0.2566443521270246, negative=2, min=-1.856, max=-1.856, mean=-0.58, count=3.0, positive=1, stdDev=1.7120467283342473, zeros=0},
    {meanExponent=0.25664435212
```
...[skipping 11507 bytes](etc/189.txt)...
```
    
    Implemented Statistics: {meanExponent=-1.0, negative=0, min=0.1, max=0.1, mean=0.03333333333333334, count=9.0, positive=3, stdDev=0.047140452079103175, zeros=6}
    Measured Feedback: [ [ 0.09999999999843467, 0.0, 0.0 ], [ 0.0, 0.09999999999843467, 0.0 ], [ 0.0, 0.0, 0.10000000000065512 ] ]
    Measured Statistics: {meanExponent=-1.0000000000035836, negative=0, min=0.10000000000065512, max=0.10000000000065512, mean=0.033333333333058275, count=9.0, positive=3, stdDev=0.04714045207871417, zeros=6}
    Feedback Error: [ [ -1.5653311979946238E-12, 0.0, 0.0 ], [ 0.0, -1.5653311979946238E-12, 0.0 ], [ 0.0, 0.0, 6.551148512556892E-13 ] ]
    Error Statistics: {meanExponent=-11.931490024324164, negative=2, min=6.551148512556892E-13, max=6.551148512556892E-13, mean=-2.750608383037287E-13, count=9.0, positive=1, stdDev=7.18700432435841E-13, zeros=6}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 4.2064e-13 +- 6.4440e-13 [0.0000e+00 - 1.5653e-12] (90#)
    relativeTol: 6.3096e-12 +- 2.1454e-12 [3.2756e-12 - 7.8267e-12] (30#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=4.2064e-13 +- 6.4440e-13 [0.0000e+00 - 1.5653e-12] (90#), relativeTol=6.3096e-12 +- 2.1454e-12 [3.2756e-12 - 7.8267e-12] (30#)}
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
      "class": "com.simiacryptus.mindseye.layers.java.AvgMetaLayer",
      "id": "487d4063-e83f-4f3d-84f8-03d1a0f4a634",
      "isFrozen": false,
      "name": "AvgMetaLayer/487d4063-e83f-4f3d-84f8-03d1a0f4a634",
      "minBatchCount": 0
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
    [[ 0.328, -1.316, -1.104 ]]
    --------------------
    Output: 
    [ 0.328, -1.316, -1.104 ]
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
    [ 0.056, -0.336, 1.932, 0.216, -0.34, 1.9, 1.976, -1.004, ... ]
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
    F(0.0) = LineSearchPoint{point=PointSample{avg=2.5223561599999993}, derivative=-0.10089424640000001}
    New Minimum: 2.5223561599999993 > 2.5223561599899105
    F(1.0E-10) = LineSearchPoint{point=PointSample{avg=2.5223561599899105}, derivative=-0.10089424639979822}, delta = -1.0088818669373723E-11
    New Minimum: 2.5223561599899105 > 2.522356159929375
    F(7.000000000000001E-10) = LineSearchPoint{point=PointSample{avg=2.522356159929375}, derivative=-0.10089424639858749}, delta = -7.062439522087516E-11
    New Minimum: 2.522356159929375 > 2.5223561595056188
    F(4.900000000000001E-9) = LineSearchPoint{point=PointSample{avg=2.5223561595056188}, derivative=-0.10089424639011237}, delta = -4.943805365087428E-10
    New Minimum: 2.5223561595056188 > 2.522356156539328
    F(3.430000000000001E-8) = LineSearchPoint{point=PointSample{avg=2.522356156539328}, derivative=-0.10089424633078656}, delta = -3.460671305077767E-9
    New Minimum: 2.522356156539328 > 2.5223561357752917
    F(2.4010000000000004E-7) 
```
...[skipping 1584 bytes](etc/190.txt)...
```
    2.384638480261366
    F(1.3841287201) = LineSearchPoint{point=PointSample{avg=2.384638480261366}, derivative=-0.09810123391729829}, delta = -0.13771767973863325
    Loops = 12
    New Minimum: 2.384638480261366 > 1.1538833705456969E-30
    F(49.999999999999964) = LineSearchPoint{point=PointSample{avg=1.1538833705456969E-30}, derivative=-6.739642177677752E-17}, delta = -2.5223561599999993
    Right bracket at 49.999999999999964
    Converged to right
    Iteration 1 complete. Error: 1.1538833705456969E-30 Total: 239634419225756.6000; Orientation: 0.0000; Line Search: 0.0033
    Zero gradient: 2.1483792687006613E-16
    F(0.0) = LineSearchPoint{point=PointSample{avg=1.1538833705456969E-30}, derivative=-4.6155334821827877E-32}
    New Minimum: 1.1538833705456969E-30 > 0.0
    F(49.999999999999964) = LineSearchPoint{point=PointSample{avg=0.0}, derivative=0.0}, delta = -1.1538833705456969E-30
    0.0 <= 1.1538833705456969E-30
    Converged to right
    Iteration 2 complete. Error: 0.0 Total: 239634420036235.6000; Orientation: 0.0001; Line Search: 0.0004
    
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
    th(0)=2.5223561599999993;dx=-0.10089424640000001
    New Minimum: 2.5223561599999993 > 2.3096691916058765
    WOLFE (weak): th(2.154434690031884)=2.3096691916058765; dx=-0.09654684511062432 delta=0.21268696839412282
    New Minimum: 2.3096691916058765 > 2.1063484153610714
    WOLFE (weak): th(4.308869380063768)=2.1063484153610714; dx=-0.09219944382124862 delta=0.41600774463892787
    New Minimum: 2.1063484153610714 > 1.3867272318750592
    END: th(12.926608140191302)=1.3867272318750592; dx=-0.07480983866374587 delta=1.13562892812494
    Iteration 1 complete. Error: 1.3867272318750592 Total: 239634424597029.5600; Orientation: 0.0001; Line Search: 0.0010
    LBFGS Accumulation History: 1 points
    th(0)=1.3867272318750592;dx=-0.05546908927500236
    New Minimum: 1.3867272318750592 > 0.2721553081086773
    END: th(27.849533001676672)=0.2721553081086773; dx=-0.024573324628259802 delta=1.1145719237663818
    Iteration 2 complete. Error: 0.2721553081086773 Total: 23963442
```
...[skipping 11677 bytes](etc/191.txt)...
```
    9634440277395.5600; Orientation: 0.0000; Line Search: 0.0005
    LBFGS Accumulation History: 1 points
    th(0)=2.1122155182580912E-32;dx=-8.448862073032365E-34
    New Minimum: 2.1122155182580912E-32 > 1.9515929608961957E-32
    WOLF (strong): th(91.31684578924755)=1.9515929608961957E-32; dx=8.072727954893538E-34 delta=1.6062255736189554E-33
    New Minimum: 1.9515929608961957E-32 > 8.185202263645753E-36
    END: th(45.658422894623776)=8.185202263645753E-36; dx=-2.6963019221421303E-36 delta=2.1113969980317266E-32
    Iteration 25 complete. Error: 8.185202263645753E-36 Total: 239634441169093.5600; Orientation: 0.0001; Line Search: 0.0007
    LBFGS Accumulation History: 1 points
    th(0)=8.185202263645753E-36;dx=-3.274080905458301E-37
    Armijo: th(98.36809017632345)=8.185202263645753E-36; dx=3.274080905458301E-37 delta=0.0
    New Minimum: 8.185202263645753E-36 > 0.0
    END: th(49.184045088161724)=0.0; dx=0.0 delta=8.185202263645753E-36
    Iteration 26 complete. Error: 0.0 Total: 239634441896927.5600; Orientation: 0.0000; Line Search: 0.0005
    
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

![Result](etc/test.114.png)



Code from [LearningTester.java:98](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L98) executed in 0.00 seconds: 
```java
    return TestUtil.compareTime(runs);
```

Returns: 

![Result](etc/test.115.png)



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
    	[100]
    Performance:
    	Evaluation performance: 0.000247s +- 0.000023s [0.000226s - 0.000290s]
    	Learning performance: 0.000005s +- 0.000002s [0.000003s - 0.000008s]
    
```

