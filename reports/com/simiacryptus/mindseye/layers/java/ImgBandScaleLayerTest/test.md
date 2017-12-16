# ImgBandScaleLayer
## ImgBandScaleLayerTest
### Batch Execution
Code from [BatchingTester.java:66](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/BatchingTester.java#L66) executed in 0.00 seconds: 
```java
    return test(reference, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (240#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (240#)}
```



Code from [SingleDerivativeTester.java:77](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/SingleDerivativeTester.java#L77) executed in 0.00 seconds: 
```java
    return test(component, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ -0.772, 1.02, 1.26 ], [ -0.604, -0.064, 0.728 ] ],
    	[ [ 1.856, -1.26, 1.964 ], [ -1.768, -0.468, 0.62 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.0984878239304965, negative=6, min=0.62, max=0.62, mean=0.20933333333333334, count=12.0, positive=6, stdDev=1.1624655789408227, zeros=0}
    Output: [
    	[ [ -0.975808, 0.75072, 1.46664 ], [ -0.763456, -0.047104, 0.8473919999999999 ] ],
    	[ [ 2.345984, -0.92736, 2.2860959999999997 ], [ -2.234752, -0.34444800000000003, 0.72168 ] ]
    ]
    Outputs Statistics: {meanExponent=-0.08696186773125159, negative=6, min=0.72168, max=0.72168, mean=0.2604653333333333, count=12.0, positive=6, stdDev=1.334769668099415, zeros=0}
    Feedback for input 0
    Inputs Values: [
    	[ [ -0.772, 1.02, 1.26 ], [ -0.604, -0.064, 0.728 ] ],
    	[ [ 1.856, -1.26, 1.964 ], [ -1.768, -0.468, 0.62 ] ]
    ]
    Value Statistics: {meanExponent=-0.0984878239304965, negative=6, min=0.62, max=0.62, mean=0.20933333333333334, count=12.0, positive=6, stdDev=1.1624655789408227, zeros=0}
    Implemented Feedback: [ [ 1.264,
```
...[skipping 2751 bytes](etc/247.txt)...
```
    9995948, -0.06399999999996686, -0.46799999999957986, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ] ]
    Measured Statistics: {meanExponent=-0.0984878239305938, negative=6, min=0.6200000000000649, max=0.6200000000000649, mean=0.06977777777772382, count=36.0, positive=6, stdDev=0.6783656466960872, zeros=24}
    Gradient Error: [ [ 5.10702591327572E-15, 7.971401316808624E-14, 3.951283744640932E-13, -1.9901857939430556E-12, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, -6.45261621912141E-13, 4.0523140398818214E-13, 3.314015728506092E-14, 4.201639036693905E-13, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ] ]
    Error Statistics: {meanExponent=-12.645953231305443, negative=4, min=6.494804694057166E-14, max=6.494804694057166E-14, mean=-5.395992294962879E-14, count=36.0, positive=8, stdDev=4.724176569047245E-13, zeros=24}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 7.9565e-14 +- 3.5760e-13 [0.0000e+00 - 3.6209e-12] (180#)
    relativeTol: 2.7150e-13 +- 3.1314e-13 [3.7711e-16 - 1.4323e-12] (24#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=7.9565e-14 +- 3.5760e-13 [0.0000e+00 - 3.6209e-12] (180#), relativeTol=2.7150e-13 +- 3.1314e-13 [3.7711e-16 - 1.4323e-12] (24#)}
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
      "class": "com.simiacryptus.mindseye.layers.java.ImgBandScaleLayer",
      "id": "5b60cfd3-f9dc-4dd8-8e91-676ff8aef6af",
      "isFrozen": false,
      "name": "ImgBandScaleLayer/5b60cfd3-f9dc-4dd8-8e91-676ff8aef6af",
      "bias": [
        1.264,
        0.736,
        1.164
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
    [[
    	[ [ 1.212, -1.368, 0.648 ], [ -1.58, -1.576, 1.276 ] ],
    	[ [ -0.472, -0.808, 0.236 ], [ -0.348, 1.288, -1.464 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 1.531968, -1.006848, 0.7542719999999999 ], [ -1.99712, -1.159936, 1.485264 ] ],
    	[ [ -0.596608, -0.594688, 0.27470399999999995 ], [ -0.439872, 0.947968, -1.7040959999999998 ] ]
    ]
    --------------------
    Derivative: 
    [
    	[ [ 1.264, 0.736, 1.164 ], [ 1.264, 0.736, 1.164 ] ],
    	[ [ 1.264, 0.736, 1.164 ], [ 1.264, 0.736, 1.164 ] ]
    ]
```



### Input Learning
In this test, we use a network to learn this target input, given it's pre-evaluated output:

Code from [LearningTester.java:127](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L127) executed in 0.00 seconds: 
```java
    return Arrays.stream(input_target).map(x -> x.prettyPrint()).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [
    	[ [ -1.564, 1.26, -0.028 ], [ -0.004, 0.844, -0.36 ] ],
    	[ [ -1.548, -0.436, -1.468 ], [ -1.248, -0.62, -0.14 ] ]
    ]
```



First, we use a conjugate gradient descent method, which converges the fastest for purely linear functions.

Code from [LearningTester.java:225](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L225) executed in 0.04 seconds: 
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
    F(0.0) = LineSearchPoint{point=PointSample{avg=1.018449389248}, derivative=-0.4170368019267812}
    New Minimum: 1.018449389248 > 1.0184493892062962
    F(1.0E-10) = LineSearchPoint{point=PointSample{avg=1.0184493892062962}, derivative=-0.4170368019172298}, delta = -4.1703751563204605E-11
    New Minimum: 1.0184493892062962 > 1.0184493889560742
    F(7.000000000000001E-10) = LineSearchPoint{point=PointSample{avg=1.0184493889560742}, derivative=-0.4170368018599212}, delta = -2.919258168532224E-10
    New Minimum: 1.0184493889560742 > 1.0184493872045197
    F(4.900000000000001E-9) = LineSearchPoint{point=PointSample{avg=1.0184493872045197}, derivative=-0.41703680145876076}, delta = -2.043480273883347E-9
    New Minimum: 1.0184493872045197 > 1.018449374943638
    F(3.430000000000001E-8) = LineSearchPoint{point=PointSample{avg=1.018449374943638}, derivative=-0.41703679865063803}, delta = -1.4304361917183428E-8
    New Minimum: 1.018449374943638 > 1.018449289117467
    F(2.4010000000000004E-7) = LineSea
```
...[skipping 60347 bytes](etc/248.txt)...
```
     = LineSearchPoint{point=PointSample{avg=2.6346746716430206E-35}, derivative=-8.23826698341902E-36}, delta = -5.266214691683848E-35
    F(31.83688170243799) = LineSearchPoint{point=PointSample{avg=2.814290208448666E-34}, derivative=2.6925067701906065E-35}, delta = 2.024201272115979E-34
    F(2.4489909001875376) = LineSearchPoint{point=PointSample{avg=4.570322035997054E-35}, derivative=-1.0850400417186026E-35}, delta = -3.330567327329814E-35
    New Minimum: 2.6346746716430206E-35 > 2.5077212817542135E-35
    F(17.142936301312762) = LineSearchPoint{point=PointSample{avg=2.5077212817542135E-35}, derivative=8.03733364236002E-36}, delta = -5.393168081572655E-35
    2.5077212817542135E-35 <= 7.900889363326868E-35
    New Minimum: 2.5077212817542135E-35 > 0.0
    F(10.96530159813699) = LineSearchPoint{point=PointSample{avg=0.0}, derivative=0.0}, delta = -7.900889363326868E-35
    Right bracket at 10.96530159813699
    Converged to right
    Iteration 47 complete. Error: 0.0 Total: 239661455468996.5300; Orientation: 0.0000; Line Search: 0.0006
    
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
    th(0)=1.018449389248;dx=-0.4170368019267812
    New Minimum: 1.018449389248 > 0.34164007330576734
    END: th(2.154434690031884)=0.34164007330576734; dx=-0.21125731074573273 delta=0.6768093159422326
    Iteration 1 complete. Error: 0.34164007330576734 Total: 239661458799254.5300; Orientation: 0.0001; Line Search: 0.0003
    LBFGS Accumulation History: 1 points
    th(0)=0.34164007330576734;dx=-0.11259995661076097
    New Minimum: 0.34164007330576734 > 0.06256177476417274
    END: th(4.641588833612779)=0.06256177476417274; dx=-0.0076512369118562126 delta=0.2790782985415946
    Iteration 2 complete. Error: 0.06256177476417274 Total: 239661459083662.5300; Orientation: 0.0000; Line Search: 0.0002
    LBFGS Accumulation History: 1 points
    th(0)=0.06256177476417274;dx=-0.012664738055887824
    New Minimum: 0.06256177476417274 > 0.01117287875582549
    WOLF (strong): th(10.000000000000002)=0.01117287875582549; dx=0.0023869588542183755 delta=0.05138889600834725
    END: th(5
```
...[skipping 28138 bytes](etc/249.txt)...
```
    ta=2.4088230237050316E-34
    Iteration 57 complete. Error: 4.012354050806741E-36 Total: 239661478500665.5300; Orientation: 0.0000; Line Search: 0.0002
    LBFGS Accumulation History: 1 points
    th(0)=4.012354050806741E-36;dx=-7.244920466352695E-37
    New Minimum: 4.012354050806741E-36 > 2.648780603852888E-36
    WOLF (strong): th(20.739628697139278)=2.648780603852888E-36; dx=5.886497878911563E-37 delta=1.3635734469538532E-36
    New Minimum: 2.648780603852888E-36 > 1.5673258010963833E-38
    END: th(10.369814348569639)=1.5673258010963833E-38; dx=-4.528075291470434E-38 delta=3.9966807927957775E-36
    Iteration 58 complete. Error: 1.5673258010963833E-38 Total: 239661478872276.5300; Orientation: 0.0000; Line Search: 0.0003
    LBFGS Accumulation History: 1 points
    th(0)=1.5673258010963833E-38;dx=-2.8300470571690214E-39
    New Minimum: 1.5673258010963833E-38 > 0.0
    END: th(22.34108776174881)=0.0; dx=0.0 delta=1.5673258010963833E-38
    Iteration 59 complete. Error: 0.0 Total: 239661479167514.5300; Orientation: 0.0000; Line Search: 0.0002
    
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

![Result](etc/test.158.png)



Code from [LearningTester.java:98](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L98) executed in 0.00 seconds: 
```java
    return TestUtil.compareTime(runs);
```

Returns: 

![Result](etc/test.159.png)



### Model Learning
In this test, attempt to train a network to emulate a randomized network given an example input/output. The target state is:

Code from [LearningTester.java:176](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L176) executed in 0.00 seconds: 
```java
    return network_target.state().stream().map(Arrays::toString).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [0.736, 1.164, 1.264]
```



First, we use a conjugate gradient descent method, which converges the fastest for purely linear functions.

Code from [LearningTester.java:225](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L225) executed in 0.02 seconds: 
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
    F(0.0) = LineSearchPoint{point=PointSample{avg=0.146270902272}, derivative=-0.16565959018341986}
    New Minimum: 0.146270902272 > 0.14627090225543404
    F(1.0E-10) = LineSearchPoint{point=PointSample{avg=0.14627090225543404}, derivative=-0.16565959017345105}, delta = -1.6565943061763733E-11
    New Minimum: 0.14627090225543404 > 0.1462709021560383
    F(7.000000000000001E-10) = LineSearchPoint{point=PointSample{avg=0.1462709021560383}, derivative=-0.16565959011363804}, delta = -1.1596168469907298E-10
    New Minimum: 0.1462709021560383 > 0.14627090146026803
    F(4.900000000000001E-9) = LineSearchPoint{point=PointSample{avg=0.14627090146026803}, derivative=-0.1656595896949472}, delta = -8.117319594269645E-10
    New Minimum: 0.14627090146026803 > 0.1462708965898761
    F(3.430000000000001E-8) = LineSearchPoint{point=PointSample{avg=0.1462708965898761}, derivative=-0.16565958676411102}, delta = -5.682123882522205E-9
    New Minimum: 0.1462708965898761 > 0.14627086249713525
    F(2.4010000000000004
```
...[skipping 33221 bytes](etc/250.txt)...
```
    
    Right bracket at 2.2631594316266157
    F(2.2301780686704435) = LineSearchPoint{point=PointSample{avg=4.109669309796816E-33}, derivative=3.788146096556814E-33}, delta = -3.3125995043460457E-32
    Right bracket at 2.2301780686704435
    F(2.2069398054971674) = LineSearchPoint{point=PointSample{avg=4.109669309796816E-33}, derivative=3.788146096556814E-33}, delta = -3.3125995043460457E-32
    Right bracket at 2.2069398054971674
    Converged to right
    Iteration 26 complete. Error: 4.109669309796816E-33 Total: 239661621915185.3800; Orientation: 0.0000; Line Search: 0.0017
    Zero gradient: 4.664584690727835E-17
    F(0.0) = LineSearchPoint{point=PointSample{avg=4.109669309796816E-33}, derivative=-2.1758350336972492E-33}
    New Minimum: 4.109669309796816E-33 > 0.0
    F(2.2069398054971674) = LineSearchPoint{point=PointSample{avg=0.0}, derivative=0.0}, delta = -4.109669309796816E-33
    0.0 <= 4.109669309796816E-33
    Converged to right
    Iteration 27 complete. Error: 0.0 Total: 239661622173375.3800; Orientation: 0.0000; Line Search: 0.0002
    
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
    th(0)=0.14223871231999993;dx=-0.22485854556392643
    New Minimum: 0.14223871231999993 > 0.0773014282863647
    WOLF (strong): th(2.154434690031884)=0.0773014282863647; dx=0.1645761110727939 delta=0.06493728403363523
    New Minimum: 0.0773014282863647 > 0.004893628583331605
    END: th(1.077217345015942)=0.004893628583331605; dx=-0.03014121724556628 delta=0.13734508373666832
    Iteration 1 complete. Error: 0.004893628583331605 Total: 239661625820528.3800; Orientation: 0.0001; Line Search: 0.0003
    LBFGS Accumulation History: 1 points
    th(0)=0.004893628583331605;dx=-0.006670019163629186
    New Minimum: 0.004893628583331605 > 0.0017030973074462004
    WOLF (strong): th(2.3207944168063896)=0.0017030973074462004; dx=0.003920502659469505 delta=0.0031905312758854047
    New Minimum: 0.0017030973074462004 > 2.2605995562469344E-4
    END: th(1.1603972084031948)=2.2605995562469344E-4; dx=-0.0013747582520798433 delta=0.004667568627706911
    Iteration 2 complete. Error
```
...[skipping 9071 bytes](etc/251.txt)...
```
    -31 delta=2.6752571251456487E-30
    Iteration 19 complete. Error: 8.732526539320819E-32 Total: 239661632621823.3800; Orientation: 0.0000; Line Search: 0.0003
    LBFGS Accumulation History: 1 points
    th(0)=8.732526539320819E-32;dx=-1.1350691575452239E-31
    New Minimum: 8.732526539320819E-32 > 5.342162598027575E-32
    WOLF (strong): th(2.623149071368624)=5.342162598027575E-32; dx=8.993656475845892E-32 delta=3.390363941293244E-32
    New Minimum: 5.342162598027575E-32 > 4.109669309796816E-33
    END: th(1.311574535684312)=4.109669309796816E-33; dx=-1.5715353125705194E-32 delta=8.321559608341138E-32
    Iteration 20 complete. Error: 4.109669309796816E-33 Total: 239661633011958.3800; Orientation: 0.0000; Line Search: 0.0003
    LBFGS Accumulation History: 1 points
    th(0)=4.109669309796816E-33;dx=-2.1758350336972492E-33
    New Minimum: 4.109669309796816E-33 > 0.0
    END: th(2.8257016782407427)=0.0; dx=0.0 delta=4.109669309796816E-33
    Iteration 21 complete. Error: 0.0 Total: 239661633270148.3800; Orientation: 0.0000; Line Search: 0.0002
    
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

![Result](etc/test.160.png)



Code from [LearningTester.java:98](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L98) executed in 0.00 seconds: 
```java
    return TestUtil.compareTime(runs);
```

Returns: 

![Result](etc/test.161.png)



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
    	[2, 2, 3]
    Performance:
    	Evaluation performance: 0.000478s +- 0.000216s [0.000263s - 0.000862s]
    	Learning performance: 0.000101s +- 0.000009s [0.000090s - 0.000116s]
    
```

