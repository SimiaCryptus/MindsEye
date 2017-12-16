# EntropyLayer
## EntropyLayerTest
### Batch Execution
Code from [BatchingTester.java:66](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/BatchingTester.java#L66) executed in 0.00 seconds: 
```java
    return test(reference, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (120#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (120#)}
```



Code from [SingleDerivativeTester.java:77](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/SingleDerivativeTester.java#L77) executed in 0.00 seconds: 
```java
    return test(component, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ 1.336 ], [ -1.748 ], [ 1.556 ] ],
    	[ [ 0.064 ], [ -0.484 ], [ 1.728 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.11851224118954083, negative=2, min=1.728, max=1.728, mean=0.4086666666666667, count=6.0, positive=4, stdDev=1.2572338242701273, zeros=0}
    Output: [
    	[ [ -0.38701258035291064 ], [ 0.9762095406038848 ], [ -0.6879362704766471 ] ],
    	[ [ 0.1759278205198378 ], [ -0.3512244601765046 ], [ -0.9451549504198609 ] ]
    ]
    Outputs Statistics: {meanExponent=-0.30312689542417043, negative=4, min=-0.9451549504198609, max=-0.9451549504198609, mean=-0.20319848338370008, count=6.0, positive=2, stdDev=0.6292733209319668, zeros=0}
    Feedback for input 0
    Inputs Values: [
    	[ [ 1.336 ], [ -1.748 ], [ 1.556 ] ],
    	[ [ 0.064 ], [ -0.484 ], [ 1.728 ] ]
    ]
    Value Statistics: {meanExponent=-0.11851224118954083, negative=2, min=1.728, max=1.728, mean=0.4086666666666667, count=6.0, positive=4, stdDev=1.2572338242701273, zeros=0}
    Implemented Feedback: [ [ -1.289680075114454, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 1.7488721956
```
...[skipping 671 bytes](etc/220.txt)...
```
    .0, 0.0, 0.0, -1.4421505587436378, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, -1.5469936050083177 ] ]
    Measured Statistics: {meanExponent=0.05539301256034962, negative=5, min=-1.5469936050083177, max=-1.5469936050083177, mean=-0.12120667495218543, count=36.0, positive=1, stdDev=0.5569572772904621, zeros=30}
    Feedback Error: [ [ -3.742421592445311E-5, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, -7.808434166380884E-4, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 2.860466537746298E-5, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 1.0331290029186313E-4, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, -3.213298743798276E-5, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, -2.893462645392475E-5 ] ]
    Error Statistics: {meanExponent=-4.182553821198959, negative=4, min=-2.893462645392475E-5, max=-2.893462645392475E-5, mean=-2.076160224403119E-5, count=36.0, positive=2, stdDev=1.3005991657520532E-4, zeros=30}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 2.8090e-05 +- 1.2868e-04 [0.0000e+00 - 7.8084e-04] (36#)
    relativeTol: 7.5968e-05 +- 9.2385e-05 [9.1772e-06 - 2.2329e-04] (6#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=2.8090e-05 +- 1.2868e-04 [0.0000e+00 - 7.8084e-04] (36#), relativeTol=7.5968e-05 +- 9.2385e-05 [9.1772e-06 - 2.2329e-04] (6#)}
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
      "class": "com.simiacryptus.mindseye.layers.java.EntropyLayer",
      "id": "b268f374-4606-42f6-aee3-1f685a817bc7",
      "isFrozen": true,
      "name": "EntropyLayer/b268f374-4606-42f6-aee3-1f685a817bc7"
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
    	[ [ 0.94 ], [ 0.772 ], [ 0.16 ] ],
    	[ [ 0.004 ], [ -1.748 ], [ 0.2 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 0.058162879495002276 ], [ 0.19977100275508258 ], [ 0.29321303419972966 ] ],
    	[ [ 0.022085843671448984 ], [ 0.9762095406038848 ], [ 0.3218875824868201 ] ]
    ]
    --------------------
    Derivative: 
    [
    	[ [ -0.9381245962819125 ], [ -0.7412292710426391 ], [ 0.8325814637483102 ] ],
    	[ [ 4.521460917862246 ], [ -1.5584722772333437 ], [ 0.6094379124341003 ] ]
    ]
```



### Input Learning
In this test, we use a network to learn this target input, given it's pre-evaluated output:

Code from [LearningTester.java:127](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L127) executed in 0.01 seconds: 
```java
    return Arrays.stream(input_target).map(x -> x.prettyPrint()).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [
    	[ [ -0.008 ], [ 0.732 ], [ -1.28 ], [ 0.92 ], [ -0.54 ], [ -1.224 ], [ 0.676 ], [ 1.712 ], ... ],
    	[ [ 0.208 ], [ 0.652 ], [ 1.588 ], [ -0.532 ], [ 0.44 ], [ -0.26 ], [ -1.056 ], [ 0.292 ], ... ],
    	[ [ 1.616 ], [ 1.624 ], [ 1.74 ], [ -1.3 ], [ -0.592 ], [ 0.932 ], [ 1.644 ], [ 1.572 ], ... ],
    	[ [ -0.656 ], [ 0.476 ], [ -0.316 ], [ -1.704 ], [ 0.972 ], [ 1.1 ], [ -1.104 ], [ 0.9 ], ... ],
    	[ [ 1.644 ], [ 1.756 ], [ -1.236 ], [ 1.556 ], [ -0.94 ], [ -1.116 ], [ 1.912 ], [ 1.188 ], ... ],
    	[ [ 1.836 ], [ -1.02 ], [ -1.364 ], [ 1.448 ], [ -1.516 ], [ 1.728 ], [ -1.936 ], [ -1.772 ], ... ],
    	[ [ 0.592 ], [ -1.808 ], [ -1.18 ], [ -1.988 ], [ 1.752 ], [ 0.232 ], [ 0.804 ], [ 0.696 ], ... ],
    	[ [ 1.328 ], [ 0.016 ], [ 0.524 ], [ 1.956 ], [ -0.832 ], [ 1.336 ], [ -0.628 ], [ -0.984 ], ... ],
    	...
    ]
```



First, we use a conjugate gradient descent method, which converges the fastest for purely linear functions.

Code from [LearningTester.java:225](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L225) executed in 4.96 seconds: 
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
    F(0.0) = LineSearchPoint{point=PointSample{avg=0.6472152154973349}, derivative=-4.7520632540981426E-4}
    New Minimum: 0.6472152154973349 > 0.647215215497288
    F(1.0E-10) = LineSearchPoint{point=PointSample{avg=0.647215215497288}, derivative=-4.752063254097986E-4}, delta = -4.6851411639181606E-14
    New Minimum: 0.647215215497288 > 0.6472152154970008
    F(7.000000000000001E-10) = LineSearchPoint{point=PointSample{avg=0.6472152154970008}, derivative=-4.7520632540970444E-4}, delta = -3.340661081097096E-13
    New Minimum: 0.6472152154970008 > 0.6472152154950083
    F(4.900000000000001E-9) = LineSearchPoint{point=PointSample{avg=0.6472152154950083}, derivative=-4.7520632540904524E-4}, delta = -2.326583370404478E-12
    New Minimum: 0.6472152154950083 > 0.6472152154810381
    F(3.430000000000001E-8) = LineSearchPoint{point=PointSample{avg=0.6472152154810381}, derivative=-4.75206325404431E-4}, delta = -1.6296741733867748E-11
    New Minimum: 0.6472152154810381 > 0.6472152153832385
    F(2.401000000
```
...[skipping 413414 bytes](etc/221.txt)...
```
    rchPoint{point=PointSample{avg=0.06572232163286751}, derivative=7.95362014946523E-11}, delta = -1.0918545818860981E-7
    Right bracket at 300.35283302528865
    New Minimum: 0.06572232163286751 > 0.06572232024635093
    F(274.4241433781931) = LineSearchPoint{point=PointSample{avg=0.06572232024635093}, derivative=2.716493270620855E-11}, delta = -1.1057197477071234E-7
    Right bracket at 274.4241433781931
    New Minimum: 0.06572232024635093 > 0.06572232008817117
    F(265.8452562303685) = LineSearchPoint{point=PointSample{avg=0.06572232008817117}, derivative=9.74891361558202E-12}, delta = -1.1073015453166857E-7
    Right bracket at 265.8452562303685
    New Minimum: 0.06572232008817117 > 0.06572232006778314
    F(262.8017244635688) = LineSearchPoint{point=PointSample{avg=0.06572232006778314}, derivative=3.6625029794451086E-12}, delta = -1.1075054255627048E-7
    Right bracket at 262.8017244635688
    Converged to right
    Iteration 250 complete. Error: 0.06572232006778314 Total: 239642413434418.6200; Orientation: 0.0004; Line Search: 0.0217
    
```

Returns: 

```
    0.06572232006778314
```



This training run resulted in the following regressed input:

Code from [LearningTester.java:144](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L144) executed in 0.01 seconds: 
```java
    return Arrays.stream(input_gd).map(x -> x.prettyPrint()).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [
    	[ [ -0.960586362729829 ], [ 0.09855480634830678 ], [ -1.2799999999999991 ], [ 0.9200000014889913 ], [ 1.2933855188928343 ], [ 0.7035472971008184 ], [ -1.2383185062890145 ], [ -0.36787944117144294 ], ... ],
    	[ [ -1.288498250184635 ], [ 0.1437909193893739 ], [ 1.5880000000000007 ], [ -0.53842233231384 ], [ 0.47651703036123444 ], [ 1.3072447038430561 ], [ 0.94066422126185 ], [ 0.4823159464639724 ], ... ],
    	[ [ 1.6160000000000008 ], [ -0.3678794411769365 ], [ -0.36787944117144256 ], [ 0.525594215076454 ], [ -0.5932955260421634 ], [ 0.015831575584576162 ], [ -0.36787944117166455 ], [ 1.5720000000000007 ], ... ],
    	[ [ 1.2480664985577492 ], [ -1.3096984550447204 ], [ -0.2915488090930464 ], [ 0.367879441171444 ], [ -1.0272366929360923 ], [ -0.8887242837472287 ], [ 0.8837329691690164 ], [ 0.8999995704943622 ], ... ],
    	[ [ -0.3678794411730114 ], [ -0.36787944117144256 ], [ 0.6806642652620731 ], [ 1.5559999999999994 ], [ -0.013513464390702608 ], [ 0.8684602452112286 ], [ -0.36787944117144245 ], [ -0.7653682470486667 ], ... ],
    	[ [ -0.36787944117144217 ], [ -1.0200000000238476 ], [ -1.363999999999999 ], [ 1.4479999999999993 ], [ -1.5159999999999996 ], [ 1.7279999999999998 ], [ 0.3678794411714422 ], [ -1.7719999999999994 ], ... ],
    	[ [ -1.2754817011428976 ], [ -1.8080000000000005 ], [ 0.777992587779858 ], [ 0.36787944117144245 ], [ -0.36787944117144256 ], [ -1.2983218211868797 ], [ 0.8039999996831095 ], [ -1.228015997828989 ], ... ],
    	[ [ 1.3279999999999992 ], [ -1.0641478281189278 ], [ -1.2980727833440189 ], [ -0.36787944117144245 ], [ -0.051635311143198824 ], [ -0.4235363711672181 ], [ -0.6284023730276189 ], [ -0.0026801049754506144 ], ... ],
    	...
    ]
```



Next, we run the same optimization using L-BFGS, which is nearly ideal for purely second-order or quadratic functions.

Code from [LearningTester.java:249](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L249) executed in 2.03 seconds: 
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
    th(0)=0.6472152154973349;dx=-4.7520632540981426E-4
    New Minimum: 0.6472152154973349 > 0.6461916335353495
    WOLFE (weak): th(2.154434690031884)=0.6461916335353495; dx=-4.7509497628242487E-4 delta=0.0010235819619853759
    New Minimum: 0.6461916335353495 > 0.6451676420568672
    WOLFE (weak): th(4.308869380063768)=0.6451676420568672; dx=-4.7490609556407156E-4 delta=0.002047573440467687
    New Minimum: 0.6451676420568672 > 0.6410844821160717
    WOLFE (weak): th(12.926608140191302)=0.6410844821160717; dx=-4.717179250332473E-4 delta=0.0061307333812631315
    New Minimum: 0.6410844821160717 > 0.6230481213937202
    WOLFE (weak): th(51.70643256076521)=0.6230481213937202; dx=-4.574877843725177E-4 delta=0.024167094103614617
    New Minimum: 0.6230481213937202 > 0.5352342333894712
    END: th(258.53216280382605)=0.5352342333894712; dx=-3.940455520788116E-4 delta=0.11198098210786367
    Iteration 1 complete. Error: 0.5352342333894712 Total: 239642450938144.5600; Orien
```
...[skipping 122304 bytes](etc/222.txt)...
```
    8438
    WOLFE (weak): th(222.52144129512064)=0.06625410792558438; dx=-2.3724754185025335E-10 delta=5.321745073783024E-8
    New Minimum: 0.06625410792558438 > 0.06625405549487047
    WOLFE (weak): th(445.0428825902413)=0.06625405549487047; dx=-2.3405422274018955E-10 delta=1.056481646471985E-7
    New Minimum: 0.06625405549487047 > 0.0662538522273182
    WOLFE (weak): th(1335.1286477707238)=0.0662538522273182; dx=-2.229386963372659E-10 delta=3.089157169233747E-7
    New Minimum: 0.0662538522273182 > 0.06625305058344545
    END: th(5340.514591082895)=0.06625305058344545; dx=-1.7735933467691662E-10 delta=1.1105595896648834E-6
    Iteration 248 complete. Error: 0.06625305058344545 Total: 239644457182873.5600; Orientation: 0.0005; Line Search: 0.0075
    LBFGS Accumulation History: 1 points
    th(0)=0.06625305058344545;dx=-1.1668988019027818E-9
    MAX ALPHA: th(0)=0.06625305058344545;th'(0)=-1.1668988019027818E-9;
    Iteration 249 failed, aborting. Error: 0.06625305058344545 Total: 239644461528794.5300; Orientation: 0.0006; Line Search: 0.0028
    
```

Returns: 

```
    0.06625305058344545
```



This training run resulted in the following regressed input:

Code from [LearningTester.java:154](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L154) executed in 0.01 seconds: 
```java
    return Arrays.stream(input_lbgfs).map(x -> x.prettyPrint()).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [
    	[ [ -0.9605863622834919 ], [ 0.09855480634830674 ], [ -1.28 ], [ 0.920000000033005 ], [ 1.2933855188928352 ], [ 0.7035267985628056 ], [ -1.238318506289016 ], [ -0.3678794411714422 ], ... ],
    	[ [ -1.2884982501846363 ], [ 0.1437909192045119 ], [ 1.588 ], [ -0.5365917938318159 ], [ 0.4736920839734284 ], [ 1.307244703843057 ], [ 0.9406642201176207 ], [ 0.47808102430276395 ], ... ],
    	[ [ 1.616 ], [ -0.3678794411713162 ], [ -0.36787944117144233 ], [ 0.5238275691079132 ], [ -0.59271897532838 ], [ 0.015831575584576162 ], [ -0.36787944117138155 ], [ 1.572 ], ... ],
    	[ [ 1.2480664985577508 ], [ -1.309698455044722 ], [ -0.29305732450495137 ], [ 0.36787944117144217 ], [ -1.0272366929229955 ], [ -0.8887243230888183 ], [ 0.8837329550741857 ], [ 0.8999999875084531 ], ... ],
    	[ [ -0.367879441171393 ], [ -0.36787944117144233 ], [ 0.6806197542507044 ], [ 1.556 ], [ -0.013513464390702608 ], [ 0.8684602432254271 ], [ -0.36787944117144233 ], [ -0.7653681055019194 ], ... ],
    	[ [ -0.36787944117144233 ], [ -1.0199999999995537 ], [ -1.3640000000000003 ], [ 1.448 ], [ -1.516 ], [ 1.728 ], [ 0.36787944117144233 ], [ -1.772 ], ... ],
    	[ [ -1.2754817011428985 ], [ -1.808 ], [ 0.7779913374519857 ], [ 0.36787944117144233 ], [ -0.36787944117144233 ], [ -1.2983218211868806 ], [ 0.8040000750296478 ], [ -1.2280159978289875 ], ... ],
    	[ [ 1.328 ], [ -1.0641478281253143 ], [ -1.2980727833440198 ], [ -0.36787944117144233 ], [ -0.051635311143198824 ], [ -0.4164800177932609 ], [ -0.6281990881299961 ], [ -0.002680104975450615 ], ... ],
    	...
    ]
```



Code from [LearningTester.java:95](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L95) executed in 0.00 seconds: 
```java
    return TestUtil.compare(runs);
```

Returns: 

![Result](etc/test.132.png)



Code from [LearningTester.java:98](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L98) executed in 0.00 seconds: 
```java
    return TestUtil.compareTime(runs);
```

Returns: 

![Result](etc/test.133.png)



### Performance
Now we execute larger-scale runs to benchmark performance:

Code from [PerformanceTester.java:66](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/PerformanceTester.java#L66) executed in 0.24 seconds: 
```java
    test(component, inputPrototype);
```
Logging: 
```
    100 batches
    Input Dimensions:
    	[100, 100, 1]
    Performance:
    	Evaluation performance: 0.006477s +- 0.000330s [0.006009s - 0.006843s]
    	Learning performance: 0.032973s +- 0.028137s [0.011178s - 0.087452s]
    
```

### Function Plots
Code from [ActivationLayerTestBase.java:110](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/java/ActivationLayerTestBase.java#L110) executed in 0.00 seconds: 
```java
    return plot("Value Plot", plotData, x -> new double[]{x[0], x[1]});
```

Returns: 

![Result](etc/test.134.png)



Code from [ActivationLayerTestBase.java:114](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/java/ActivationLayerTestBase.java#L114) executed in 0.00 seconds: 
```java
    return plot("Derivative Plot", plotData, x -> new double[]{x[0], x[2]});
```

Returns: 

![Result](etc/test.135.png)



