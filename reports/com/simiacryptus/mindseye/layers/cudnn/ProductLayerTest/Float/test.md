# ProductLayer
## Float
### Batch Execution
Code from [BatchingTester.java:66](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/BatchingTester.java#L66) executed in 0.02 seconds: 
```java
    return test(reference, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (120#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (120#)}
```



Code from [SingleDerivativeTester.java:77](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/SingleDerivativeTester.java#L77) executed in 0.02 seconds: 
```java
    return test(component, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ 1.812 ], [ -1.86 ] ],
    	[ [ 0.3 ], [ -1.424 ] ]
    ],
    [
    	[ [ 1.82 ], [ -0.516 ] ],
    	[ [ -1.2 ], [ -0.552 ] ]
    ]
    Inputs Statistics: {meanExponent=0.039575595394802626, negative=2, min=-1.424, max=-1.424, mean=-0.293, count=4.0, positive=2, stdDev=1.4592227383096796, zeros=0},
    {meanExponent=-0.05153964665272252, negative=3, min=-0.552, max=-0.552, mean=-0.11199999999999999, count=4.0, positive=1, stdDev=1.1481707190135098, zeros=0}
    Output: [
    	[ [ 3.2978400000000003 ], [ 0.9597600000000001 ] ],
    	[ [ -0.36 ], [ 0.7860480000000001 ] ]
    ]
    Outputs Statistics: {meanExponent=-0.01196405125791987, negative=1, min=0.7860480000000001, max=0.7860480000000001, mean=1.1709120000000002, count=4.0, positive=3, stdDev=1.3285536999429117, zeros=0}
    Feedback for input 0
    Inputs Values: [
    	[ [ 1.812 ], [ -1.86 ] ],
    	[ [ 0.3 ], [ -1.424 ] ]
    ]
    Value Statistics: {meanExponent=0.039575595394802626, negative=2, min=-1.424, max=-1.424, mean=-0.293, count=4.0, positive=2, stdDev=1.4592227383096796, zeros=0}
    Implemented Fee
```
...[skipping 1623 bytes](etc/140.txt)...
```
    positive=2, stdDev=0.7405602186858271, zeros=12}
    Measured Feedback: [ [ 1.8119999999997027, 0.0, 0.0, 0.0 ], [ 0.0, 0.29999999999996696, 0.0, 0.0 ], [ 0.0, 0.0, -1.8599999999999728, 0.0 ], [ 0.0, 0.0, 0.0, -1.4240000000000919 ] ]
    Measured Statistics: {meanExponent=0.0395755953947783, negative=2, min=-1.4240000000000919, max=-1.4240000000000919, mean=-0.07325000000002468, count=16.0, positive=2, stdDev=0.7405602186857851, zeros=12}
    Feedback Error: [ [ -2.973177259946169E-13, 0.0, 0.0, 0.0 ], [ 0.0, -3.302913498259841E-14, 0.0, 0.0 ], [ 0.0, 0.0, 2.731148640577885E-14, 0.0 ], [ 0.0, 0.0, 0.0, -9.192646643896296E-14 ] ]
    Error Statistics: {meanExponent=-13.1520240235338, negative=3, min=-9.192646643896296E-14, max=-9.192646643896296E-14, mean=-2.4685115063149965E-14, count=16.0, positive=1, stdDev=7.455512556954937E-14, zeros=12}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 3.2007e-14 +- 1.0146e-13 [0.0000e+00 - 5.1137e-13] (32#)
    relativeTol: 4.5834e-14 +- 4.3742e-14 [2.8158e-15 - 1.4049e-13] (8#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=3.2007e-14 +- 1.0146e-13 [0.0000e+00 - 5.1137e-13] (32#), relativeTol=4.5834e-14 +- 4.3742e-14 [2.8158e-15 - 1.4049e-13] (8#)}
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
      "class": "com.simiacryptus.mindseye.layers.cudnn.ProductLayer",
      "id": "a474d527-b620-43a6-bbeb-aa19c5ff6762",
      "isFrozen": false,
      "name": "ProductLayer/a474d527-b620-43a6-bbeb-aa19c5ff6762"
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
    [[
    	[ [ 0.364 ], [ -0.604 ] ],
    	[ [ -1.588 ], [ -0.04 ] ]
    ],
    [
    	[ [ 1.14 ], [ 0.248 ] ],
    	[ [ -1.32 ], [ -1.076 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 0.41495999999999994 ], [ -0.14979199999999998 ] ],
    	[ [ 2.0961600000000002 ], [ 0.04304 ] ]
    ]
    --------------------
    Derivative: 
    [
    	[ [ 1.14 ], [ 0.248 ] ],
    	[ [ -1.32 ], [ -1.076 ] ]
    ],
    [
    	[ [ 0.364 ], [ -0.604 ] ],
    	[ [ -1.588 ], [ -0.04 ] ]
    ]
```



[GPU Log](etc/cuda.log)

### Input Learning
In this test, we use a network to learn this target input, given it's pre-evaluated output:

Code from [LearningTester.java:127](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L127) executed in 0.00 seconds: 
```java
    return Arrays.stream(input_target).map(x -> x.prettyPrint()).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [
    	[ [ 0.216 ], [ -0.384 ] ],
    	[ [ 1.884 ], [ -1.6 ] ]
    ]
    [
    	[ [ -0.676 ], [ -0.944 ] ],
    	[ [ 1.448 ], [ 1.956 ] ]
    ]
```



First, we use a conjugate gradient descent method, which converges the fastest for purely linear functions.

Code from [LearningTester.java:225](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L225) executed in 2.80 seconds: 
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
    F(0.0) = LineSearchPoint{point=PointSample{avg=14.792082774591998}, derivative=-177.4284061800769}
    New Minimum: 14.792082774591998 > 14.79208275684916
    F(1.0E-10) = LineSearchPoint{point=PointSample{avg=14.79208275684916}, derivative=-177.42840594733235}, delta = -1.7742838309686704E-8
    New Minimum: 14.79208275684916 > 14.792082650392116
    F(7.000000000000001E-10) = LineSearchPoint{point=PointSample{avg=14.792082650392116}, derivative=-177.42840455086454}, delta = -1.2419988237866164E-7
    New Minimum: 14.792082650392116 > 14.792081905192838
    F(4.900000000000001E-9) = LineSearchPoint{point=PointSample{avg=14.792081905192838}, derivative=-177.42839477559028}, delta = -8.693991606634199E-7
    New Minimum: 14.792081905192838 > 14.792076688799037
    F(3.430000000000001E-8) = LineSearchPoint{point=PointSample{avg=14.792076688799037}, derivative=-177.4283263486819}, delta = -6.08579296113021E-6
    New Minimum: 14.792076688799037 > 14.792040174098764
    F(2.4010000000000004E-7) = LineS
```
...[skipping 271998 bytes](etc/141.txt)...
```
    8582077E-8
    F(0.0) = LineSearchPoint{point=PointSample{avg=2.4539292080640025}, derivative=-1.1279037882805869E-15}
    F(0.25992757355884605) = LineSearchPoint{point=PointSample{avg=2.4539292080640025}, derivative=-6.134272788956362E-16}, delta = 0.0
    F(1.8194930149119224) = LineSearchPoint{point=PointSample{avg=2.453929208064004}, derivative=2.4734316097236235E-15}, delta = 1.3322676295501878E-15
    F(0.13996100114707094) = LineSearchPoint{point=PointSample{avg=2.4539292080640025}, derivative=-8.508779637111816E-16}, delta = 0.0
    F(0.9797270080294966) = LineSearchPoint{point=PointSample{avg=2.4539292080640025}, derivative=8.112768309429277E-16}, delta = 0.0
    2.4539292080640025 <= 2.4539292080640025
    F(0.5698477969936359) = LineSearchPoint{point=PointSample{avg=2.4539292080640025}, derivative=1.2604517164041321E-23}, delta = 0.0
    Right bracket at 0.5698477969936359
    Converged to right
    Iteration 237 failed, aborting. Error: 2.4539292080640025 Total: 239594399278734.6200; Orientation: 0.0000; Line Search: 0.0083
    
```

Returns: 

```
    2.4539292080640025
```



This training run resulted in the following regressed input:

Code from [LearningTester.java:144](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L144) executed in 0.00 seconds: 
```java
    return Arrays.stream(input_gd).map(x -> x.prettyPrint()).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [
    	[ [ 1.7604736339828607E-7 ], [ -0.602076407111257 ] ],
    	[ [ 1.6516755057801409 ], [ 3.8567662145191775E-112 ] ]
    ]
    [
    	[ [ -0.944 ], [ -0.676 ] ],
    	[ [ 1.956 ], [ 1.448 ] ]
    ]
```



Next, we run the same optimization using L-BFGS, which is nearly ideal for purely second-order or quadratic functions.

Code from [LearningTester.java:249](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L249) executed in 0.86 seconds: 
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
    th(0)=14.792082774591998;dx=-177.4284061800769
    Armijo: th(2.154434690031884)=103129.76358472338; dx=204990.93371681543 delta=-103114.97150194878
    Armijo: th(1.077217345015942)=4877.7778040096255; dx=20589.218717602973 delta=-4862.985721235033
    Armijo: th(0.3590724483386473)=26.985783233597004; dx=333.1227895368123 delta=-12.193700459005006
    New Minimum: 14.792082774591998 > 5.845724750272009
    END: th(0.08976811208466183)=5.845724750272009; dx=-43.993030311837984 delta=8.94635802431999
    Iteration 1 complete. Error: 5.845724750272009 Total: 239594411476664.6000; Orientation: 0.0001; Line Search: 0.0063
    LBFGS Accumulation History: 1 points
    th(0)=5.845724750272009;dx=-12.119104144840792
    New Minimum: 5.845724750272009 > 4.350433057971288
    END: th(0.1933995347338658)=4.350433057971288; dx=-4.018816271865152 delta=1.4952916923007207
    Iteration 2 complete. Error: 4.350433057971288 Total: 239594415380013.6000; Orientation: 0.0000; Line 
```
...[skipping 77978 bytes](etc/142.txt)...
```
    =4.440892098500626E-16
    Iteration 163 complete. Error: 2.4539292080640047 Total: 239595249139863.7500; Orientation: 0.0000; Line Search: 0.0049
    LBFGS Accumulation History: 1 points
    th(0)=2.4539292080640047;dx=-1.6149349635380276E-15
    New Minimum: 2.4539292080640047 > 2.453929208064004
    END: th(0.671657809906608)=2.453929208064004; dx=-7.694334026995306E-16 delta=8.881784197001252E-16
    Iteration 164 complete. Error: 2.453929208064004 Total: 239595252944610.7500; Orientation: 0.0000; Line Search: 0.0025
    LBFGS Accumulation History: 1 points
    th(0)=2.453929208064004;dx=-2.5248794306968993E-15
    Armijo: th(1.4470428854936368)=2.4539292080640083; dx=8.51933508180521E-15 delta=-4.440892098500626E-15
    Armijo: th(0.7235214427468184)=2.4539292080640043; dx=2.997227683812021E-15 delta=-4.440892098500626E-16
    END: th(0.24117381424893947)=2.453929208064004; dx=-6.841770967206725E-16 delta=0.0
    Iteration 165 failed, aborting. Error: 2.453929208064004 Total: 239595259034028.7500; Orientation: 0.0000; Line Search: 0.0048
    
```

Returns: 

```
    2.453929208064004
```



This training run resulted in the following regressed input:

Code from [LearningTester.java:154](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L154) executed in 0.00 seconds: 
```java
    return Arrays.stream(input_lbgfs).map(x -> x.prettyPrint()).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [
    	[ [ -2.2474685224265955E-7 ], [ -0.6020764071112569 ] ],
    	[ [ 1.6516755150673137 ], [ -1.9793319357862805E-30 ] ]
    ]
    [
    	[ [ -0.944 ], [ -0.676 ] ],
    	[ [ 1.956 ], [ 1.448 ] ]
    ]
```



Code from [LearningTester.java:95](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L95) executed in 0.00 seconds: 
```java
    return TestUtil.compare(runs);
```

Returns: 

![Result](etc/test.84.png)



Code from [LearningTester.java:98](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L98) executed in 0.00 seconds: 
```java
    return TestUtil.compareTime(runs);
```

Returns: 

![Result](etc/test.85.png)



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
    	[2, 2, 1]
    	[2, 2, 1]
    Performance:
    	Evaluation performance: 0.000469s +- 0.000262s [0.000302s - 0.000989s]
    	Learning performance: 0.000284s +- 0.000047s [0.000241s - 0.000352s]
    
```

