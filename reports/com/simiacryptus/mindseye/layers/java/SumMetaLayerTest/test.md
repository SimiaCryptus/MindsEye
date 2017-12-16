# SumMetaLayer
## SumMetaLayerTest
### Differential Validation
Code from [BatchDerivativeTester.java:76](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/BatchDerivativeTester.java#L76) executed in 0.01 seconds: 
```java
    return test(component, inputPrototype);
```
Logging: 
```
    Inputs: [ -1.112, 0.348, -1.908 ],
    [ -1.112, 0.348, -1.908 ],
    [ -1.112, 0.348, -1.908 ],
    [ -1.112, 0.348, -1.908 ],
    [ -1.112, 0.348, -1.908 ],
    [ -1.112, 0.348, -1.908 ],
    [ -1.112, 0.348, -1.908 ],
    [ -1.112, 0.348, -1.908 ],
    [ -1.112, 0.348, -1.908 ],
    [ -1.112, 0.348, -1.908 ]
    Inputs Statistics: {meanExponent=-0.04391253281310135, negative=2, min=-1.908, max=-1.908, mean=-0.8906666666666667, count=3.0, positive=1, stdDev=0.9342110158964206, zeros=0},
    {meanExponent=-0.04391253281310135, negative=2, min=-1.908, max=-1.908, mean=-0.8906666666666667, count=3.0, positive=1, stdDev=0.9342110158964206, zeros=0},
    {meanExponent=-0.04391253281310135, negative=2, min=-1.908, max=-1.908, mean=-0.8906666666666667, count=3.0, positive=1, stdDev=0.9342110158964206, zeros=0},
    {meanExponent=-0.04391253281310135, negative=2, min=-1.908, max=-1.908, mean=-0.8906666666666667, count=3.0, positive=1, stdDev=0.9342110158964206, zeros=0},
    {meanExponent=-0.04391253281310135, negative=2, min=-1.908, max=-1.908, mean=-0.890666666666666
```
...[skipping 11874 bytes](etc/348.txt)...
```
    Implemented Statistics: {meanExponent=0.0, negative=0, min=1.0, max=1.0, mean=0.3333333333333333, count=9.0, positive=3, stdDev=0.4714045207910317, zeros=6}
    Measured Feedback: [ [ 0.9999999999976694, 0.0, 0.0 ], [ 0.0, 1.0000000000021103, 0.0 ], [ 0.0, 0.0, 0.9999999999976694 ] ]
    Measured Statistics: {meanExponent=-3.6927313119324734E-13, negative=0, min=0.9999999999976694, max=0.9999999999976694, mean=0.33333333333304993, count=9.0, positive=3, stdDev=0.47140452079063083, zeros=6}
    Feedback Error: [ [ -2.3305801732931286E-12, 0.0, 0.0 ], [ 0.0, 2.1103119252074976E-12, 0.0 ], [ 0.0, 0.0, -2.3305801732931286E-12 ] ]
    Error Statistics: {meanExponent=-11.646908417293593, negative=2, min=-2.3305801732931286E-12, max=-2.3305801732931286E-12, mean=-2.834276023754177E-13, count=9.0, positive=1, stdDev=1.2733875842663203E-12, zeros=6}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 7.5239e-13 +- 1.0657e-12 [0.0000e+00 - 2.3306e-12] (90#)
    relativeTol: 1.1286e-12 +- 5.1918e-14 [1.0552e-12 - 1.1653e-12] (30#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=7.5239e-13 +- 1.0657e-12 [0.0000e+00 - 2.3306e-12] (90#), relativeTol=1.1286e-12 +- 5.1918e-14 [1.0552e-12 - 1.1653e-12] (30#)}
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
      "class": "com.simiacryptus.mindseye.layers.java.SumMetaLayer",
      "id": "a6370bd3-3bfa-450e-b476-7bcca18e8998",
      "isFrozen": false,
      "name": "SumMetaLayer/a6370bd3-3bfa-450e-b476-7bcca18e8998",
      "minBatches": 0
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
    [[ -1.504, 1.704, -1.016 ]]
    --------------------
    Output: 
    [ -1.504, 1.704, -1.016 ]
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
    [ 1.892, 1.908, 1.44, -0.388, 0.7, -1.084, 1.492, -0.684, ... ]
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
    F(0.0) = LineSearchPoint{point=PointSample{avg=3.1538249599999997}, derivative=-0.1261529984}
    New Minimum: 3.1538249599999997 > 3.153824959987385
    F(1.0E-10) = LineSearchPoint{point=PointSample{avg=3.153824959987385}, derivative=-0.1261529983997477}, delta = -1.2614798095000879E-11
    New Minimum: 3.153824959987385 > 3.1538249599116916
    F(7.000000000000001E-10) = LineSearchPoint{point=PointSample{avg=3.1538249599116916}, derivative=-0.12615299839823385}, delta = -8.830802755710465E-11
    New Minimum: 3.1538249599116916 > 3.153824959381852
    F(4.900000000000001E-9) = LineSearchPoint{point=PointSample{avg=3.153824959381852}, derivative=-0.126152998387637}, delta = -6.181477552047454E-10
    New Minimum: 3.153824959381852 > 3.1538249556729534
    F(3.430000000000001E-8) = LineSearchPoint{point=PointSample{avg=3.1538249556729534}, derivative=-0.12615299831345905}, delta = -4.327046276841884E-9
    New Minimum: 3.1538249556729534 > 3.153824929710664
    F(2.4010000000000004E-7) = LineSearc
```
...[skipping 1546 bytes](etc/349.txt)...
```
    9713820254 > 2.981629826465412
    F(1.3841287201) = LineSearchPoint{point=PointSample{avg=2.981629826465412}, derivative=-0.12266075863575661}, delta = -0.1721951335345877
    Loops = 12
    New Minimum: 2.981629826465412 > 1.089785533101573E-30
    F(50.00000000000003) = LineSearchPoint{point=PointSample{avg=1.089785533101573E-30}, derivative=7.33515470585644E-17}, delta = -3.1538249599999997
    Right bracket at 50.00000000000003
    Converged to right
    Iteration 1 complete. Error: 1.089785533101573E-30 Total: 239730378778963.6200; Orientation: 0.0001; Line Search: 0.0026
    Zero gradient: 2.0878558696438538E-16
    F(0.0) = LineSearchPoint{point=PointSample{avg=1.089785533101573E-30}, derivative=-4.3591421324062924E-32}
    New Minimum: 1.089785533101573E-30 > 0.0
    F(50.00000000000003) = LineSearchPoint{point=PointSample{avg=0.0}, derivative=0.0}, delta = -1.089785533101573E-30
    0.0 <= 1.089785533101573E-30
    Converged to right
    Iteration 2 complete. Error: 0.0 Total: 239730379138320.6200; Orientation: 0.0000; Line Search: 0.0002
    
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
    th(0)=3.1538249599999997;dx=-0.1261529984
    New Minimum: 3.1538249599999997 > 2.8878920674825035
    WOLFE (weak): th(2.154434690031884)=2.8878920674825035; dx=-0.12071723047991006 delta=0.2659328925174962
    New Minimum: 2.8878920674825035 > 2.6336701819390163
    WOLFE (weak): th(4.308869380063768)=2.6336701819390163; dx=-0.11528146255982014 delta=0.5201547780609834
    New Minimum: 2.6336701819390163 > 1.7338927095050958
    END: th(12.926608140191302)=1.7338927095050958; dx=-0.09353839087946039 delta=1.419932250494904
    Iteration 1 complete. Error: 1.7338927095050958 Total: 239730382016317.6200; Orientation: 0.0001; Line Search: 0.0005
    LBFGS Accumulation History: 1 points
    th(0)=1.7338927095050958;dx=-0.06935570838020383
    New Minimum: 1.7338927095050958 > 0.3402890588257119
    END: th(27.849533001676672)=0.3402890588257119; dx=-0.03072522659242083 delta=1.3936036506793839
    Iteration 2 complete. Error: 0.3402890588257119 Total: 239730382381374.6
```
...[skipping 11674 bytes](etc/350.txt)...
```
    4449639.6000; Orientation: 0.0000; Line Search: 0.0004
    LBFGS Accumulation History: 1 points
    th(0)=2.4831014772984633E-32;dx=-9.932405909193854E-34
    New Minimum: 2.4831014772984633E-32 > 2.4107346596381128E-32
    WOLF (strong): th(91.31684578924755)=2.4107346596381128E-32; dx=9.767546305954306E-34 delta=7.236681766035053E-34
    New Minimum: 2.4107346596381128E-32 > 1.1555579666323415E-35
    END: th(45.658422894623776)=1.1555579666323415E-35; dx=-3.235562306570556E-36 delta=2.481945919331831E-32
    Iteration 25 complete. Error: 1.1555579666323415E-35 Total: 239730395011047.6000; Orientation: 0.0000; Line Search: 0.0004
    LBFGS Accumulation History: 1 points
    th(0)=1.1555579666323415E-35;dx=-4.6222318665293665E-37
    Armijo: th(98.36809017632345)=1.1555579666323415E-35; dx=4.6222318665293665E-37 delta=0.0
    New Minimum: 1.1555579666323415E-35 > 0.0
    END: th(49.184045088161724)=0.0; dx=0.0 delta=1.1555579666323415E-35
    Iteration 26 complete. Error: 0.0 Total: 239730395577014.6000; Orientation: 0.0000; Line Search: 0.0004
    
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

![Result](etc/test.249.png)



Code from [LearningTester.java:98](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L98) executed in 0.00 seconds: 
```java
    return TestUtil.compareTime(runs);
```

Returns: 

![Result](etc/test.250.png)



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
    	Evaluation performance: 0.000255s +- 0.000029s [0.000234s - 0.000312s]
    	Learning performance: 0.000013s +- 0.000009s [0.000004s - 0.000029s]
    
```

