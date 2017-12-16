# ActivationLayer
## Sigmoid_Float
### Batch Execution
Code from [BatchingTester.java:66](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/BatchingTester.java#L66) executed in 0.01 seconds: 
```java
    return test(reference, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (20#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (20#)}
```



Code from [SingleDerivativeTester.java:77](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/SingleDerivativeTester.java#L77) executed in 0.00 seconds: 
```java
    return test(component, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ 1.104 ] ]
    ]
    Inputs Statistics: {meanExponent=0.04296907339318013, negative=0, min=1.104, max=1.104, mean=1.104, count=1.0, positive=1, stdDev=0.0, zeros=0}
    Output: [
    	[ [ 0.7510088346069963 ] ]
    ]
    Outputs Statistics: {meanExponent=-0.12435495407638997, negative=0, min=0.7510088346069963, max=0.7510088346069963, mean=0.7510088346069963, count=1.0, positive=1, stdDev=0.0, zeros=0}
    Feedback for input 0
    Inputs Values: [
    	[ [ 1.104 ] ]
    ]
    Value Statistics: {meanExponent=0.04296907339318013, negative=0, min=1.104, max=1.104, mean=1.104, count=1.0, positive=1, stdDev=0.0, zeros=0}
    Implemented Feedback: [ [ 0.18699456494923758 ] ]
    Implemented Statistics: {meanExponent=-0.72817101617397, negative=0, min=0.18699456494923758, max=0.18699456494923758, mean=0.18699456494923758, count=1.0, positive=1, stdDev=0.0, zeros=0}
    Measured Feedback: [ [ 0.1869898711825968 ] ]
    Measured Statistics: {meanExponent=-0.7281819175738925, negative=0, min=0.1869898711825968, max=0.1869898711825968, mean=0.1869898711825968, count=1.0, positive=1, stdDev=0.0, zeros=0}
    Feedback Error: [ [ -4.693766640778696E-6 ] ]
    Error Statistics: {meanExponent=-5.3284785059477775, negative=1, min=-4.693766640778696E-6, max=-4.693766640778696E-6, mean=-4.693766640778696E-6, count=1.0, positive=0, stdDev=0.0, zeros=0}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 4.6938e-06 +- 0.0000e+00 [4.6938e-06 - 4.6938e-06] (1#)
    relativeTol: 1.2551e-05 +- 0.0000e+00 [1.2551e-05 - 1.2551e-05] (1#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=4.6938e-06 +- 0.0000e+00 [4.6938e-06 - 4.6938e-06] (1#), relativeTol=1.2551e-05 +- 0.0000e+00 [1.2551e-05 - 1.2551e-05] (1#)}
```



### Reference Implementation
Code from [EquivalencyTester.java:61](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/EquivalencyTester.java#L61) executed in 0.00 seconds: 
```java
    System.out.println(new GsonBuilder().setPrettyPrinting().create().toJson(this.reference.getJson()));
```
Logging: 
```
    {
      "class": "com.simiacryptus.mindseye.layers.java.SigmoidActivationLayer",
      "id": "4b6747b1-b75c-4bbf-b874-7c64a0822501",
      "isFrozen": true,
      "name": "SigmoidActivationLayer/4b6747b1-b75c-4bbf-b874-7c64a0822501",
      "balanced": false
    }
    
```

Code from [EquivalencyTester.java:64](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/EquivalencyTester.java#L64) executed in 0.00 seconds: 
```java
    return test(subject, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ -1.992 ] ]
    ]
    Error: [
    	[ [ -2.7755575615628914E-17 ] ]
    ]
    Accuracy:
    absoluteTol: 2.7756e-17 +- 0.0000e+00 [2.7756e-17 - 2.7756e-17] (1#)
    relativeTol: 1.1560e-16 +- 0.0000e+00 [1.1560e-16 - 1.1560e-16] (1#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=2.7756e-17 +- 0.0000e+00 [2.7756e-17 - 2.7756e-17] (1#), relativeTol=1.1560e-16 +- 0.0000e+00 [1.1560e-16 - 1.1560e-16] (1#)}
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
      "class": "com.simiacryptus.mindseye.layers.cudnn.ActivationLayer",
      "id": "212497bf-3ba5-4f93-a145-90b3fd1af23d",
      "isFrozen": false,
      "name": "ActivationLayer/212497bf-3ba5-4f93-a145-90b3fd1af23d",
      "mode": 0
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
    	[ [ -0.004 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 0.49900000133333117 ] ]
    ]
    --------------------
    Derivative: 
    [
    	[ [ 0.2499990000026667 ] ]
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
    	[ [ -1.108 ], [ -0.284 ], [ 0.3 ], [ -0.164 ], [ -1.648 ], [ -0.336 ], [ -1.052 ], [ -1.796 ], ... ],
    	[ [ -0.984 ], [ 0.34 ], [ 0.984 ], [ 0.06 ], [ 0.504 ], [ 1.7 ], [ -1.216 ], [ 1.568 ], ... ],
    	[ [ 1.792 ], [ -0.976 ], [ 1.188 ], [ -0.688 ], [ 1.932 ], [ 1.588 ], [ 0.5 ], [ 0.092 ], ... ],
    	[ [ 0.904 ], [ -1.512 ], [ 1.804 ], [ 1.536 ], [ -0.716 ], [ -0.776 ], [ -0.224 ], [ -0.212 ], ... ],
    	[ [ -0.128 ], [ 0.316 ], [ 0.896 ], [ -0.004 ], [ 1.84 ], [ -1.768 ], [ 1.3 ], [ 0.972 ], ... ],
    	[ [ 0.108 ], [ -0.124 ], [ -1.924 ], [ 1.708 ], [ 1.06 ], [ 0.256 ], [ -0.836 ], [ 0.404 ], ... ],
    	[ [ 1.068 ], [ 0.84 ], [ 0.892 ], [ -0.816 ], [ 0.94 ], [ 0.508 ], [ -1.976 ], [ -1.536 ], ... ],
    	[ [ -0.628 ], [ 0.092 ], [ 1.552 ], [ 0.532 ], [ -1.744 ], [ -1.08 ], [ 1.516 ], [ -1.576 ], ... ],
    	...
    ]
```



First, we use a conjugate gradient descent method, which converges the fastest for purely linear functions.

Code from [LearningTester.java:225](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L225) executed in 3.62 seconds: 
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
    F(0.0) = LineSearchPoint{point=PointSample{avg=0.11973717178115599}, derivative=-1.5115577685826839E-6}
    New Minimum: 0.11973717178115599 > 0.11973717178115587
    F(1.0E-10) = LineSearchPoint{point=PointSample{avg=0.11973717178115587}, derivative=-1.5115577685826837E-6}, delta = -1.249000902703301E-16
    New Minimum: 0.11973717178115587 > 0.11973717178115482
    F(7.000000000000001E-10) = LineSearchPoint{point=PointSample{avg=0.11973717178115482}, derivative=-1.5115577685826826E-6}, delta = -1.1657341758564144E-15
    New Minimum: 0.11973717178115482 > 0.11973717178114873
    F(4.900000000000001E-9) = LineSearchPoint{point=PointSample{avg=0.11973717178114873}, derivative=-1.5115577685826748E-6}, delta = -7.258083023486961E-15
    New Minimum: 0.11973717178114873 > 0.11973717178110496
    F(3.430000000000001E-8) = LineSearchPoint{point=PointSample{avg=0.11973717178110496}, derivative=-1.5115577685826201E-6}, delta = -5.102862576933376E-14
    New Minimum: 0.11973717178110496 > 0.119737171780
```
...[skipping 308599 bytes](etc/27.txt)...
```
    rivative=-1.621452092745335E-14}, delta = -6.949915141308237E-6
    Left bracket at 165000.7798034052
    Converged to left
    Iteration 249 complete. Error: 0.002811144696688906 Total: 239450562129209.4700; Orientation: 0.0004; Line Search: 0.0163
    Low gradient: 9.470029119294098E-6
    F(0.0) = LineSearchPoint{point=PointSample{avg=0.002811144696688906}, derivative=-8.968145152027813E-11}
    New Minimum: 0.002811144696688906 > 0.00280395770655076
    F(165000.7798034052) = LineSearchPoint{point=PointSample{avg=0.00280395770655076}, derivative=2.918804156842609E-12}, delta = -7.186990138146124E-6
    0.00280395770655076 <= 0.002811144696688906
    New Minimum: 0.00280395770655076 > 0.0028039502029780046
    F(159799.8766476762) = LineSearchPoint{point=PointSample{avg=0.0028039502029780046}, derivative=-3.2939369955505225E-14}, delta = -7.194493710901452E-6
    Left bracket at 159799.8766476762
    Converged to left
    Iteration 250 complete. Error: 0.0028039502029780046 Total: 239450569350561.4400; Orientation: 0.0003; Line Search: 0.0057
    
```

Returns: 

```
    0.0028039502029780046
```



This training run resulted in the following regressed input:

Code from [LearningTester.java:144](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L144) executed in 0.00 seconds: 
```java
    return Arrays.stream(input_gd).map(x -> x.prettyPrint()).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [
    	[ [ -1.1079999999999999 ], [ -0.28400000014490323 ], [ 0.29999999999056204 ], [ -0.1641031633861363 ], [ -1.648 ], [ -0.3360000000000855 ], [ -1.0519999999999998 ], [ -1.796 ], ... ],
    	[ [ -0.9840000000000002 ], [ 0.33999999999999525 ], [ 0.9839999999999999 ], [ 0.06366812701430173 ], [ 0.5040000000000002 ], [ 7.978479076696267 ], [ -1.216 ], [ 7.699762095773366 ], ... ],
    	[ [ 7.973749333579785 ], [ -0.976 ], [ 1.188 ], [ -0.6879999999999997 ], [ 1.9319999999999997 ], [ 1.5880000000000003 ], [ 0.49999999999999994 ], [ 0.09993879176386801 ], ... ],
    	[ [ 0.904 ], [ -5.62375768799327 ], [ 8.071649422957396 ], [ 7.816613126229469 ], [ -0.716 ], [ -0.776 ], [ -0.2240002929638279 ], [ -0.21199884461686647 ], ... ],
    	[ [ -0.12783939442205333 ], [ 0.31599999999867395 ], [ 0.8959999999999998 ], [ 0.03344462019106095 ], [ 1.8400000000000007 ], [ -6.637831660577932 ], [ 1.3000000000000007 ], [ 0.9719999999999999 ], ... ],
    	[ [ 0.10407617273676782 ], [ -0.12238426614540752 ], [ -8.34196712558127 ], [ 1.7080000000000006 ], [ 1.0599999999999998 ], [ 0.2559999946227428 ], [ -0.8360000000000003 ], [ 0.4040000000000003 ], ... ],
    	[ [ 1.068 ], [ 0.8400000000000002 ], [ 0.8919999999999998 ], [ -0.8159999999999998 ], [ 0.94 ], [ 0.508 ], [ -1.9760000000000002 ], [ -7.791331481867607 ], ... ],
    	[ [ -0.6279999999999999 ], [ 0.08386440089358159 ], [ 1.5520000000000005 ], [ 0.5320000000000001 ], [ -7.703320528321431 ], [ -1.0800000000000003 ], [ 6.152683728888603 ], [ -1.576 ], ... ],
    	...
    ]
```



Next, we run the same optimization using L-BFGS, which is nearly ideal for purely second-order or quadratic functions.

Code from [LearningTester.java:249](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L249) executed in 4.31 seconds: 
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
    th(0)=0.11973717178115599;dx=-1.5115577685826839E-6
    New Minimum: 0.11973717178115599 > 0.1197339152329695
    WOLFE (weak): th(2.154434690031884)=0.1197339152329695; dx=-1.5115537719933812E-6 delta=3.2565481864893497E-6
    New Minimum: 0.1197339152329695 > 0.11973065869339221
    WOLFE (weak): th(4.308869380063768)=0.11973065869339221; dx=-1.511549774092595E-6 delta=6.513087763782011E-6
    New Minimum: 0.11973065869339221 > 0.11971763262126364
    WOLFE (weak): th(12.926608140191302)=0.11971763262126364; dx=-1.5115337693742518E-6 delta=1.9539159892348335E-5
    New Minimum: 0.11971763262126364 > 0.11965901700722922
    WOLFE (weak): th(51.70643256076521)=0.11965901700722922; dx=-1.5114614884329352E-6 delta=7.815477392676684E-5
    New Minimum: 0.11965901700722922 > 0.11934644828075686
    WOLFE (weak): th(258.53216280382605)=0.11934644828075686; dx=-1.5110688087565027E-6 delta=3.907235003991344E-4
    New Minimum: 0.11934644828075686 > 0.11739486162772719
    W
```
...[skipping 323961 bytes](etc/28.txt)...
```
    -4 > 2.5608610634347297E-4
    WOLFE (weak): th(4.308869380063768)=2.5608610634347297E-4; dx=-2.548021285482624E-9 delta=1.0979232758947903E-8
    New Minimum: 2.5608610634347297E-4 > 2.5606414872911926E-4
    WOLFE (weak): th(12.926608140191302)=2.5606414872911926E-4; dx=-2.5478895945257318E-9 delta=3.29368471126525E-8
    New Minimum: 2.5606414872911926E-4 > 2.5596535350843935E-4
    WOLFE (weak): th(51.70643256076521)=2.5596535350843935E-4; dx=-2.5472969996361034E-9 delta=1.3173206779256482E-7
    New Minimum: 2.5596535350843935E-4 > 2.5543883375369995E-4
    WOLFE (weak): th(258.53216280382605)=2.5543883375369995E-4; dx=-2.544136891951324E-9 delta=6.58251822531963E-7
    New Minimum: 2.5543883375369995E-4 > 2.521628861484077E-4
    WOLFE (weak): th(1551.1929768229563)=2.521628861484077E-4; dx=-2.5244014154049375E-9 delta=3.934199427824223E-6
    MAX ALPHA: th(0)=2.560970855762319E-4;th'(0)=-2.548087131397935E-9;
    Iteration 250 complete. Error: 2.521628861484077E-4 Total: 239454887799945.1200; Orientation: 0.0005; Line Search: 0.0139
    
```

Returns: 

```
    2.521628861484077E-4
```



This training run resulted in the following regressed input:

Code from [LearningTester.java:154](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L154) executed in 0.00 seconds: 
```java
    return Arrays.stream(input_lbgfs).map(x -> x.prettyPrint()).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [
    	[ [ -1.1046834072685656 ], [ -0.28717472440912106 ], [ 0.29898283818086846 ], [ -0.20432078730663178 ], [ -1.4945968293528133 ], [ -0.3245636151078753 ], [ -0.9671913794104534 ], [ -1.513665123079821 ], ... ],
    	[ [ -1.0086875421939323 ], [ 0.35205048547119333 ], [ 0.9433970002417964 ], [ 0.06302740057428334 ], [ 0.5031102411369284 ], [ 1.4824846383043133 ], [ -1.1571107881895133 ], [ 1.399307324567051 ], ... ],
    	[ [ 1.5695335618179176 ], [ -0.953689314972496 ], [ 1.2099907920791615 ], [ -0.6898234048092508 ], [ 1.5730939705355298 ], [ 1.482345259085258 ], [ 0.5002139640927905 ], [ 0.09375380694526023 ], ... ],
    	[ [ 0.8566343421801178 ], [ -1.351150515051582 ], [ 1.542089458042439 ], [ 1.3906191023451475 ], [ -0.6883576739110567 ], [ -0.7583813005572358 ], [ -0.25746149998211537 ], [ -0.21834493021900986 ], ... ],
    	[ [ -0.10551770171075174 ], [ 0.3089426474239431 ], [ 0.8290207374847715 ], [ 0.011612591070651867 ], [ 1.8040129939758611 ], [ -1.510700293128607 ], [ 1.3410108501256972 ], [ 0.9211457327843584 ], ... ],
    	[ [ 0.13095190715053365 ], [ -0.13384923780208183 ], [ -1.6459019323690445 ], [ 1.5405173864121278 ], [ 1.0193976700670344 ], [ 0.2547334602893842 ], [ -0.8430189704696202 ], [ 0.39575275210139227 ], ... ],
    	[ [ 1.0287270108537003 ], [ 0.8223029520021321 ], [ 0.8914268330592472 ], [ -0.8969183456061088 ], [ 0.9228271848224753 ], [ 0.5295150533392625 ], [ -1.83939624550827 ], [ -1.3911509751433837 ], ... ],
    	[ [ -0.583249340885853 ], [ 0.07454971650312951 ], [ 1.5329834561146822 ], [ 0.5382297981047977 ], [ -1.5048013770380786 ], [ -1.066191830745498 ], [ 1.387129797584355 ], [ -1.3650410773970327 ], ... ],
    	...
    ]
```



Code from [LearningTester.java:95](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L95) executed in 0.00 seconds: 
```java
    return TestUtil.compare(runs);
```

Returns: 

![Result](etc/test.25.png)



Code from [LearningTester.java:98](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L98) executed in 0.00 seconds: 
```java
    return TestUtil.compareTime(runs);
```

Returns: 

![Result](etc/test.26.png)



### Performance
Now we execute larger-scale runs to benchmark performance:

Code from [PerformanceTester.java:66](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/PerformanceTester.java#L66) executed in 0.22 seconds: 
```java
    test(component, inputPrototype);
```
Logging: 
```
    100 batches
    Input Dimensions:
    	[100, 100, 1]
    Performance:
    	Evaluation performance: 0.006133s +- 0.000605s [0.005442s - 0.006948s]
    	Learning performance: 0.028841s +- 0.003785s [0.024522s - 0.033322s]
    
```

### Function Plots
Code from [ActivationLayerTest.java:90](../../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/cudnn/ActivationLayerTest.java#L90) executed in 0.00 seconds: 
```java
    return ActivationLayerTestBase.plot("Value Plot", plotData, x -> new double[]{x[0], x[1]});
```

Returns: 

![Result](etc/test.27.png)



Code from [ActivationLayerTest.java:94](../../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/cudnn/ActivationLayerTest.java#L94) executed in 0.00 seconds: 
```java
    return ActivationLayerTestBase.plot("Derivative Plot", plotData, x -> new double[]{x[0], x[2]});
```

Returns: 

![Result](etc/test.28.png)



