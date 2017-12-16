# ProductLayer
## Double
### Batch Execution
Code from [BatchingTester.java:66](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/BatchingTester.java#L66) executed in 0.05 seconds: 
```java
    return test(reference, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (120#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (120#)}
```



Code from [SingleDerivativeTester.java:77](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/SingleDerivativeTester.java#L77) executed in 0.04 seconds: 
```java
    return test(component, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ -1.98 ], [ 1.608 ] ],
    	[ [ -0.42 ], [ -1.044 ] ]
    ],
    [
    	[ [ 0.648 ], [ 0.124 ] ],
    	[ [ -0.592 ], [ 0.192 ] ]
    ]
    Inputs Statistics: {meanExponent=0.03622525593452685, negative=3, min=-1.044, max=-1.044, mean=-0.45899999999999996, count=4.0, positive=1, stdDev=1.316213888393524, zeros=0},
    {meanExponent=-0.5098450933851756, negative=1, min=0.192, max=0.192, mean=0.09300000000000001, count=4.0, positive=3, stdDev=0.44385019995489466, zeros=0}
    Output: [
    	[ [ -1.28304 ], [ 0.199392 ] ],
    	[ [ 0.24863999999999997 ], [ -0.20044800000000002 ] ]
    ]
    Outputs Statistics: {meanExponent=-0.47361983745064873, negative=2, min=-0.20044800000000002, max=-0.20044800000000002, mean=-0.258864, count=4.0, positive=2, stdDev=0.6164226998545722, zeros=0}
    Feedback for input 0
    Inputs Values: [
    	[ [ -1.98 ], [ 1.608 ] ],
    	[ [ -0.42 ], [ -1.044 ] ]
    ]
    Value Statistics: {meanExponent=0.03622525593452685, negative=3, min=-1.044, max=-1.044, mean=-0.45899999999999996, count=4.0, positive=1, stdDev=1.316213888393524, zeros=0}
```
...[skipping 1649 bytes](etc/134.txt)...
```
    , positive=1, stdDev=0.6874644990834072, zeros=12}
    Measured Feedback: [ [ -1.980000000001425, 0.0, 0.0, 0.0 ], [ 0.0, -0.4199999999998649, 0.0, 0.0 ], [ 0.0, 0.0, 1.6079999999998873, 0.0 ], [ 0.0, 0.0, 0.0, -1.0439999999997673 ] ]
    Measured Statistics: {meanExponent=0.036225255934538264, negative=3, min=-1.0439999999997673, max=-1.0439999999997673, mean=-0.11475000000007313, count=16.0, positive=1, stdDev=0.6874644990836078, zeros=12}
    Feedback Error: [ [ -1.425082274408851E-12, 0.0, 0.0, 0.0 ], [ 0.0, 1.350586309456503E-13, 0.0, 0.0 ], [ 0.0, 0.0, -1.127986593019159E-13, 0.0 ], [ 0.0, 0.0, 0.0, 2.327027459614328E-13 ] ]
    Error Statistics: {meanExponent=-12.57413306820353, negative=2, min=2.327027459614328E-13, max=2.327027459614328E-13, mean=-7.313247230023023E-14, count=16.0, positive=2, stdDev=3.562303233223569E-13, zeros=12}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 1.0980e-13 +- 3.3497e-13 [0.0000e+00 - 1.4251e-12] (32#)
    relativeTol: 2.6439e-13 +- 3.0729e-13 [3.5074e-14 - 1.0419e-12] (8#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=1.0980e-13 +- 3.3497e-13 [0.0000e+00 - 1.4251e-12] (32#), relativeTol=2.6439e-13 +- 3.0729e-13 [3.5074e-14 - 1.0419e-12] (8#)}
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
      "id": "1ea764a5-19b0-41d3-841a-426eab4d66c4",
      "isFrozen": false,
      "name": "ProductLayer/1ea764a5-19b0-41d3-841a-426eab4d66c4"
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
    	[ [ 0.32 ], [ -1.832 ] ],
    	[ [ -1.004 ], [ 0.68 ] ]
    ],
    [
    	[ [ -1.788 ], [ -1.428 ] ],
    	[ [ 0.268 ], [ 0.448 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ -0.57216 ], [ 2.616096 ] ],
    	[ [ -0.26907200000000003 ], [ 0.30464 ] ]
    ]
    --------------------
    Derivative: 
    [
    	[ [ -1.788 ], [ -1.428 ] ],
    	[ [ 0.268 ], [ 0.448 ] ]
    ],
    [
    	[ [ 0.32 ], [ -1.832 ] ],
    	[ [ -1.004 ], [ 0.68 ] ]
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
    	[ [ -0.752 ], [ 0.12 ] ],
    	[ [ -1.892 ], [ 1.348 ] ]
    ]
    [
    	[ [ 0.676 ], [ 0.424 ] ],
    	[ [ -0.876 ], [ 0.284 ] ]
    ]
```



First, we use a conjugate gradient descent method, which converges the fastest for purely linear functions.

Code from [LearningTester.java:225](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L225) executed in 2.10 seconds: 
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
    F(0.0) = LineSearchPoint{point=PointSample{avg=5.148213474560001}, derivative=-54.459287357325856}
    New Minimum: 5.148213474560001 > 5.148213469114071
    F(1.0E-10) = LineSearchPoint{point=PointSample{avg=5.148213469114071}, derivative=-54.459287303831054}, delta = -5.445929929237536E-9
    New Minimum: 5.148213469114071 > 5.148213436438499
    F(7.000000000000001E-10) = LineSearchPoint{point=PointSample{avg=5.148213436438499}, derivative=-54.45928698286221}, delta = -3.8121502399235396E-8
    New Minimum: 5.148213436438499 > 5.148213207709499
    F(4.900000000000001E-9) = LineSearchPoint{point=PointSample{avg=5.148213207709499}, derivative=-54.45928473608039}, delta = -2.6685050258379306E-7
    New Minimum: 5.148213207709499 > 5.1482116066067585
    F(3.430000000000001E-8) = LineSearchPoint{point=PointSample{avg=5.1482116066067585}, derivative=-54.459269008609525}, delta = -1.8679532427512413E-6
    New Minimum: 5.1482116066067585 > 5.148200398900526
    F(2.4010000000000004E-7) = LineSearchPo
```
...[skipping 245218 bytes](etc/135.txt)...
```
    lete. Error: 0.06460543897600007 Total: 239581363617121.6600; Orientation: 0.0000; Line Search: 0.0087
    Low gradient: 6.915386989752228E-9
    F(0.0) = LineSearchPoint{point=PointSample{avg=0.06460543897600007}, derivative=-4.7822577218034375E-17}
    F(0.9388546324415533) = LineSearchPoint{point=PointSample{avg=0.06460543897600009}, derivative=5.775236537107897E-17}, delta = 1.3877787807814457E-17
    F(0.07221958711088872) = LineSearchPoint{point=PointSample{avg=0.06460543897600007}, derivative=-3.970142935413424E-17}, delta = 0.0
    F(0.505537109776221) = LineSearchPoint{point=PointSample{avg=0.06460543897600007}, derivative=9.025471218467657E-18}, delta = 0.0
    0.06460543897600007 <= 0.06460543897600007
    F(0.4252755922810505) = LineSearchPoint{point=PointSample{avg=0.06460543897600007}, derivative=-2.442000938208674E-24}, delta = 0.0
    Left bracket at 0.4252755922810505
    Converged to left
    Iteration 210 failed, aborting. Error: 0.06460543897600007 Total: 239581372314092.6200; Orientation: 0.0000; Line Search: 0.0074
    
```

Returns: 

```
    0.06460543897600007
```



This training run resulted in the following regressed input:

Code from [LearningTester.java:144](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L144) executed in 0.00 seconds: 
```java
    return Arrays.stream(input_gd).map(x -> x.prettyPrint()).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [
    	[ [ -1.9946100178774418E-42 ], [ -0.22556591927670075 ] ],
    	[ [ 1.2873973752658818 ], [ -0.6187341917172511 ] ]
    ]
    [
    	[ [ 0.676 ], [ -0.876 ] ],
    	[ [ 0.424 ], [ 0.284 ] ]
    ]
```



Next, we run the same optimization using L-BFGS, which is nearly ideal for purely second-order or quadratic functions.

Code from [LearningTester.java:249](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L249) executed in 0.89 seconds: 
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
    th(0)=5.148213474560001;dx=-54.459287357325856
    Armijo: th(2.154434690031884)=6305.339587119806; dx=13514.285303127945 delta=-6300.191373645246
    Armijo: th(1.077217345015942)=202.4919859839217; dx=1021.4781854876446 delta=-197.34377250936168
    New Minimum: 5.148213474560001 > 0.7482209729189331
    WOLF (strong): th(0.3590724483386473)=0.7482209729189331; dx=0.2292946734119835 delta=4.399992501641068
    END: th(0.08976811208466183)=2.0113713427461652; dx=-19.54677960758819 delta=3.136842131813836
    Iteration 1 complete. Error: 0.7482209729189331 Total: 239581384997055.6200; Orientation: 0.0001; Line Search: 0.0064
    LBFGS Accumulation History: 1 points
    th(0)=2.0113713427461652;dx=-7.533790132522183
    New Minimum: 2.0113713427461652 > 1.0686188983926297
    END: th(0.1933995347338658)=1.0686188983926297; dx=-2.7715118763475592 delta=0.9427524443535356
    Iteration 2 complete. Error: 1.0686188983926297 Total: 239581388766464.6200; Orientation: 0.
```
...[skipping 61141 bytes](etc/136.txt)...
```
    -17
    New Minimum: 0.06460543897600009 > 0.06460543897600003
    WOLF (strong): th(6.06342560623273)=0.06460543897600003; dx=2.491829315733623E-19 delta=5.551115123125783E-17
    END: th(3.031712803116365)=0.06460543897600005; dx=-8.62257269727958E-18 delta=4.163336342344337E-17
    Iteration 128 complete. Error: 0.06460543897600003 Total: 239582248235355.7500; Orientation: 0.0000; Line Search: 0.0071
    LBFGS Accumulation History: 1 points
    th(0)=0.06460543897600005;dx=-3.740601150779161E-17
    Armijo: th(6.5316272332477)=0.06460543897600188; dx=6.004956663620901E-16 delta=-1.8318679906315083E-15
    Armijo: th(3.26581361662385)=0.06460543897600045; dx=2.815448217176041E-16 delta=-4.0245584642661925E-16
    Armijo: th(1.0886045388746166)=0.06460543897600006; dx=6.891093145922313E-17 delta=-1.3877787807814457E-17
    END: th(0.27215113471865415)=0.06460543897600005; dx=-1.0826778169277161E-17 delta=0.0
    Iteration 129 failed, aborting. Error: 0.06460543897600005 Total: 239582262048257.7500; Orientation: 0.0001; Line Search: 0.0113
    
```

Returns: 

```
    0.06460543897600005
```



This training run resulted in the following regressed input:

Code from [LearningTester.java:154](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L154) executed in 0.00 seconds: 
```java
    return Arrays.stream(input_lbgfs).map(x -> x.prettyPrint()).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [
    	[ [ -5.98100115567852E-29 ], [ -0.2255659822299235 ] ],
    	[ [ 1.2873973743910962 ], [ -0.6187341917172511 ] ]
    ]
    [
    	[ [ 0.676 ], [ -0.876 ] ],
    	[ [ 0.424 ], [ 0.284 ] ]
    ]
```



Code from [LearningTester.java:95](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L95) executed in 0.00 seconds: 
```java
    return TestUtil.compare(runs);
```

Returns: 

![Result](etc/test.80.png)



Code from [LearningTester.java:98](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L98) executed in 0.00 seconds: 
```java
    return TestUtil.compareTime(runs);
```

Returns: 

![Result](etc/test.81.png)



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
    	Evaluation performance: 0.000574s +- 0.000374s [0.000262s - 0.001292s]
    	Learning performance: 0.000215s +- 0.000024s [0.000194s - 0.000260s]
    
```

