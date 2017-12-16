# LogActivationLayer
## LogActivationLayerTest
### Batch Execution
Code from [BatchingTester.java:66](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/BatchingTester.java#L66) executed in 0.00 seconds: 
```java
    return test(reference, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (120#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (118#)}
```



Code from [SingleDerivativeTester.java:77](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/SingleDerivativeTester.java#L77) executed in 0.00 seconds: 
```java
    return test(component, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ -0.888 ], [ -0.388 ], [ 0.44 ] ],
    	[ [ -0.012 ], [ -1.868 ], [ -0.312 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.49576498669681035, negative=5, min=-0.312, max=-0.312, mean=-0.5046666666666667, count=6.0, positive=1, stdDev=0.729322669026238, zeros=0}
    Output: [
    	[ [ -0.11878353598996698 ], [ -0.9467499393588635 ], [ -0.8209805520698302 ] ],
    	[ [ -4.422848629194137 ], [ 0.6248683398066509 ], [ -1.1647520911726548 ] ]
    ]
    Outputs Statistics: {meanExponent=-0.08782524993061942, negative=5, min=-1.1647520911726548, max=-1.1647520911726548, mean=-1.141541067996467, count=6.0, positive=1, stdDev=1.58505814068908, zeros=0}
    Feedback for input 0
    Inputs Values: [
    	[ [ -0.888 ], [ -0.388 ], [ 0.44 ] ],
    	[ [ -0.012 ], [ -1.868 ], [ -0.312 ] ]
    ]
    Value Statistics: {meanExponent=-0.49576498669681035, negative=5, min=-0.312, max=-0.312, mean=-0.5046666666666667, count=6.0, positive=1, stdDev=0.729322669026238, zeros=0}
    Implemented Feedback: [ [ -1.1261261261261262, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, -83.333
```
...[skipping 667 bytes](etc/271.txt)...
```
    0, 0.0, 0.0, 0.0, 2.2727272508404894, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, -3.205128251693168 ] ]
    Measured Statistics: {meanExponent=0.49576501835128745, negative=5, min=-3.205128251693168, max=-3.205128251693168, mean=-2.4584596303553163, count=36.0, positive=1, stdDev=13.693571915356243, zeros=30}
    Feedback Error: [ [ -1.2047296049644274E-8, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, -3.4695015457941736E-5, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, -3.6757255017505486E-8, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 4.751195326058166E-9, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, -2.1886783496682938E-8, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, -4.656496255250886E-8 ] ]
    Error Statistics: {meanExponent=-7.18807602925003, negative=5, min=-4.656496255250886E-8, max=-4.656496255250886E-8, mean=-9.66875571103667E-7, count=36.0, positive=1, stdDev=5.701105454307736E-6, zeros=30}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 9.6714e-07 +- 5.7011e-06 [0.0000e+00 - 3.4695e-05] (36#)
    relativeTol: 3.9528e-08 +- 7.5427e-08 [4.4376e-09 - 2.0817e-07] (6#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=9.6714e-07 +- 5.7011e-06 [0.0000e+00 - 3.4695e-05] (36#), relativeTol=3.9528e-08 +- 7.5427e-08 [4.4376e-09 - 2.0817e-07] (6#)}
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
      "class": "com.simiacryptus.mindseye.layers.java.LogActivationLayer",
      "id": "aeb301d3-a9db-47c0-8394-5339dcb59af3",
      "isFrozen": true,
      "name": "LogActivationLayer/aeb301d3-a9db-47c0-8394-5339dcb59af3"
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
    	[ [ -1.128 ], [ -0.664 ], [ -0.548 ] ],
    	[ [ -1.724 ], [ 0.004 ], [ -0.044 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 0.12044615307586706 ], [ -0.40947312950570314 ], [ -0.6014799920341214 ] ],
    	[ [ 0.5446471722415014 ], [ -5.521460917862246 ], [ -3.123565645063876 ] ]
    ]
    --------------------
    Derivative: 
    [
    	[ [ -0.8865248226950355 ], [ -1.506024096385542 ], [ -1.824817518248175 ] ],
    	[ [ -0.580046403712297 ], [ 250.0 ], [ -22.72727272727273 ] ]
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
    	[ [ 0.968 ], [ -1.412 ], [ -0.348 ], [ 1.08 ], [ 1.708 ], [ 1.72 ], [ -1.104 ], [ 0.4 ], ... ],
    	[ [ -0.844 ], [ 0.284 ], [ -1.868 ], [ 0.096 ], [ 1.06 ], [ 1.212 ], [ -1.996 ], [ -0.72 ], ... ],
    	[ [ -0.248 ], [ 0.508 ], [ 1.364 ], [ 1.3 ], [ -1.804 ], [ 1.048 ], [ -0.268 ], [ 0.496 ], ... ],
    	[ [ -1.68 ], [ 0.744 ], [ -0.196 ], [ 1.428 ], [ 1.924 ], [ 0.34 ], [ 0.192 ], [ 1.616 ], ... ],
    	[ [ 1.54 ], [ 0.524 ], [ 0.152 ], [ -0.692 ], [ 1.904 ], [ -0.908 ], [ -0.672 ], [ -0.156 ], ... ],
    	[ [ -1.308 ], [ -0.576 ], [ 1.612 ], [ -0.352 ], [ 0.56 ], [ 1.272 ], [ 0.008 ], [ -0.88 ], ... ],
    	[ [ -0.02 ], [ 0.788 ], [ -0.668 ], [ 1.108 ], [ -0.988 ], [ -0.272 ], [ -0.144 ], [ 0.316 ], ... ],
    	[ [ -1.328 ], [ 1.104 ], [ 0.564 ], [ 0.652 ], [ -1.688 ], [ 1.116 ], [ 1.46 ], [ -0.716 ], ... ],
    	...
    ]
```



First, we use a conjugate gradient descent method, which converges the fastest for purely linear functions.

Code from [LearningTester.java:225](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L225) executed in 5.25 seconds: 
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
    F(0.0) = LineSearchPoint{point=PointSample{avg=1.9137985897152718}, derivative=-2.273858428617093}
    New Minimum: 1.9137985897152718 > 1.9137985894878902
    F(1.0E-10) = LineSearchPoint{point=PointSample{avg=1.9137985894878902}, derivative=-2.273858413147103}, delta = -2.2738166904900936E-10
    New Minimum: 1.9137985894878902 > 1.9137985881235673
    F(7.000000000000001E-10) = LineSearchPoint{point=PointSample{avg=1.9137985881235673}, derivative=-2.273858320327169}, delta = -1.5917045459445944E-9
    New Minimum: 1.9137985881235673 > 1.9137985785733693
    F(4.900000000000001E-9) = LineSearchPoint{point=PointSample{avg=1.9137985785733693}, derivative=-2.2738576705878604}, delta = -1.1141902511724311E-8
    New Minimum: 1.9137985785733693 > 1.91379851172202
    F(3.430000000000001E-8) = LineSearchPoint{point=PointSample{avg=1.91379851172202}, derivative=-2.2738531224239567}, delta = -7.799325185686712E-8
    New Minimum: 1.91379851172202 > 1.9137980437663162
    F(2.4010000000000004E-7) = LineSe
```
...[skipping 419333 bytes](etc/272.txt)...
```
    9728753876
    F(474.7561509943) = LineSearchPoint{point=PointSample{avg=0.7254599109432321}, derivative=-0.001019329411137495}, delta = 0.025093366734589062
    F(36.51970392263846) = LineSearchPoint{point=PointSample{avg=0.6931633636729365}, derivative=6.641908471269906E-4}, delta = -0.007203180535706566
    0.6931633636729365 <= 0.7003665442086431
    F(13.506622124876383) = LineSearchPoint{point=PointSample{avg=0.6964022766804314}, derivative=-1.7969359404887758E-4}, delta = -0.003964267528211662
    Left bracket at 13.506622124876383
    F(18.40694161959125) = LineSearchPoint{point=PointSample{avg=0.6953960051857578}, derivative=-2.4545829019086997E-4}, delta = -0.004970539022885245
    Left bracket at 18.40694161959125
    F(23.294460865361792) = LineSearchPoint{point=PointSample{avg=0.695452329754047}, derivative=0.0013615097388757153}, delta = -0.004914214454596055
    0.695452329754047 > 0.6931633636729365
    Iteration 250 complete. Error: 0.6921343244798892 Total: 239669096212406.9000; Orientation: 0.0003; Line Search: 0.0160
    
```

Returns: 

```
    0.695452329754047
```



This training run resulted in the following regressed input:

Code from [LearningTester.java:144](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L144) executed in 0.00 seconds: 
```java
    return Arrays.stream(input_gd).map(x -> x.prettyPrint()).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [
    	[ [ 0.9822625972357484 ], [ -1.418981751924541 ], [ 0.34825173377348545 ], [ 1.3021522430382322 ], [ 1.578918501338576 ], [ -1.5983786591930231 ], [ 1.2462069369280426 ], [ 0.39999527431704157 ], ... ],
    	[ [ 1.2570464804413817 ], [ 0.28361444448293516 ], [ 1.5380802349544291 ], [ 0.09599999934252118 ], [ 1.0400017095363099 ], [ 1.4036644220869454 ], [ -2.930991691721913 ], [ 0.9406495055243742 ], ... ],
    	[ [ -0.24799999940602976 ], [ 0.6557257644580508 ], [ -1.3607939174080688 ], [ -1.2280997776713278 ], [ -2.046894660577777 ], [ 1.0654063395828708 ], [ -0.2679999714834637 ], [ 0.4954024353396616 ], ... ],
    	[ [ -1.492677176659493 ], [ -0.8306687676041711 ], [ -0.19599999999693304 ], [ 1.3728714306557361 ], [ -1.616800347915775 ], [ 0.4104363927298313 ], [ 0.1919832106719203 ], [ 1.6694152971616762 ], ... ],
    	[ [ -1.4525757881878303 ], [ 1.3339634054306622 ], [ 0.15200000000000005 ], [ 0.6913663098263392 ], [ 1.5696681024629602 ], [ -0.9065324494176652 ], [ 0.7251126196166601 ], [ 0.15599998991876957 ], ... ],
    	[ [ -1.5269453936924213 ], [ -0.6673546655469715 ], [ 1.7109819150156944 ], [ 0.3564763977238726 ], [ -0.5588091821983702 ], [ -1.2117611182608394 ], [ -2.2383313022717113 ], [ -1.6399891392436459 ], ... ],
    	[ [ 1.538733014347292 ], [ -0.7947924355366125 ], [ -0.6871707133580256 ], [ 1.3191229129763289 ], [ -1.0964441223429495 ], [ -0.2719999591029336 ], [ 0.1439999999781817 ], [ -0.31600056147514316 ], ... ],
    	[ [ -1.2552745118534554 ], [ 1.2061408924339123 ], [ -0.5746638470879173 ], [ 0.6726990309750307 ], [ 1.5519279122211822 ], [ 1.3203687159105528 ], [ 1.6279932096107808 ], [ 0.8774804238290742 ], ... ],
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
    th(0)=1.9137985897152718;dx=-2.273858428617093
    New Minimum: 1.9137985897152718 > 1.6529074567616489
    END: th(2.154434690031884)=1.6529074567616489; dx=-0.04002439852245306 delta=0.260891132953623
    Iteration 1 complete. Error: 1.6529074567616489 Total: 239669113900159.9000; Orientation: 0.0005; Line Search: 0.0029
    LBFGS Accumulation History: 1 points
    th(0)=1.6529074567616489;dx=-0.01774712205279952
    New Minimum: 1.6529074567616489 > 1.583495952230531
    END: th(4.641588833612779)=1.583495952230531; dx=-0.011150057936072938 delta=0.06941150453111788
    Iteration 2 complete. Error: 1.583495952230531 Total: 239669118200198.8800; Orientation: 0.0005; Line Search: 0.0029
    LBFGS Accumulation History: 1 points
    th(0)=1.583495952230531;dx=-0.08667227660432694
    New Minimum: 1.583495952230531 > 1.5056973124489812
    END: th(10.000000000000002)=1.5056973124489812; dx=-0.006277273425597889 delta=0.07779863978154977
    Iteration 3 complete. Error: 1.
```
...[skipping 130062 bytes](etc/273.txt)...
```
    ry: 1 points
    th(0)=0.6684141598540848;dx=-0.0014249711703075323
    Armijo: th(0.7765200851211426)=0.669121720039687; dx=0.0034415370283220345 delta=-7.075601856021407E-4
    Armijo: th(0.3882600425605713)=0.6684421255665847; dx=7.627961667710608E-4 delta=-2.7965712499855577E-5
    New Minimum: 0.6684141598540848 > 0.668341968224877
    END: th(0.12942001418685709)=0.668341968224877; dx=-4.7050429285668975E-5 delta=7.21916292077962E-5
    Iteration 249 complete. Error: 0.668341968224877 Total: 239671131546874.8800; Orientation: 0.0007; Line Search: 0.0064
    LBFGS Accumulation History: 1 points
    th(0)=0.668341968224877;dx=-8.60396557951284E-4
    New Minimum: 0.668341968224877 > 0.6682349889245996
    WOLF (strong): th(0.27882696814858343)=0.6682349889245996; dx=6.699179150970266E-4 delta=1.0697930027736557E-4
    END: th(0.13941348407429172)=0.6682429043442711; dx=-5.067354501295143E-4 delta=9.906388060587634E-5
    Iteration 250 complete. Error: 0.6682349889245996 Total: 239671137542250.8800; Orientation: 0.0005; Line Search: 0.0045
    
```

Returns: 

```
    0.6682429043442711
```



This training run resulted in the following regressed input:

Code from [LearningTester.java:154](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L154) executed in 0.00 seconds: 
```java
    return Arrays.stream(input_lbgfs).map(x -> x.prettyPrint()).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [
    	[ [ 0.8443279947354525 ], [ -1.4239737876072933 ], [ 0.9199680221968921 ], [ 0.858953508509983 ], [ 1.5084055785455284 ], [ -1.5345109767391143 ], [ 1.381789465122597 ], [ 0.4836713813506854 ], ... ],
    	[ [ 1.6187269125144335 ], [ 0.2840107949562866 ], [ 1.225977828652793 ], [ 0.09599508434250525 ], [ 0.9789168262735323 ], [ 1.5372565160698048 ], [ -1.1076849454659174 ], [ 1.3267605806131573 ], ... ],
    	[ [ -0.2980112074288781 ], [ 1.2579219474957815 ], [ -1.3581793116201524 ], [ -1.1242572710746421 ], [ -1.066907196860525 ], [ 1.0969190894892016 ], [ -0.2679992126090936 ], [ 0.5014648723964972 ], ... ],
    	[ [ -1.36369608697533 ], [ -1.085412806512268 ], [ 0.19611209240867142 ], [ 1.3272304261449792 ], [ -1.44921784818344 ], [ 1.3392021789084907 ], [ 0.8741697715627355 ], [ 1.0459316355800434 ], ... ],
    	[ [ -1.0531796942540104 ], [ 0.5124786914614747 ], [ 0.15271813562781586 ], [ 0.7002443591770346 ], [ 1.1987737598373436 ], [ -0.8418158955601677 ], [ 0.9856307686993792 ], [ -0.1560000978777809 ], ... ],
    	[ [ -1.6509619254787502 ], [ -1.1068996277367964 ], [ 1.753449409565303 ], [ 0.3519425253850664 ], [ -0.6136277917249449 ], [ -1.1239024432375582 ], [ -3.7226706826041664 ], [ -0.7527454831711199 ], ... ],
    	[ [ 0.019999999999999962 ], [ -0.84856313104133 ], [ -0.8545192568760289 ], [ 1.4896377086826669 ], [ -0.8258687688347834 ], [ 0.27197714616650703 ], [ -9.537426487649531 ], [ -0.3159941746824347 ], ... ],
    	[ [ -1.0198437057296834 ], [ 1.314994027117956 ], [ -0.7914779775133497 ], [ 0.8587463939673158 ], [ 1.0906626739090637 ], [ 1.4853255290443272 ], [ 1.7095018629368457 ], [ 1.2299347616287941 ], ... ],
    	...
    ]
```



Code from [LearningTester.java:95](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L95) executed in 0.00 seconds: 
```java
    return TestUtil.compare(runs);
```

Returns: 

![Result](etc/test.178.png)



Code from [LearningTester.java:98](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L98) executed in 0.00 seconds: 
```java
    return TestUtil.compareTime(runs);
```

Returns: 

![Result](etc/test.179.png)



### Performance
Now we execute larger-scale runs to benchmark performance:

Code from [PerformanceTester.java:66](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/PerformanceTester.java#L66) executed in 0.17 seconds: 
```java
    test(component, inputPrototype);
```
Logging: 
```
    100 batches
    Input Dimensions:
    	[100, 100, 1]
    Performance:
    	Evaluation performance: 0.013912s +- 0.013658s [0.006710s - 0.041223s]
    	Learning performance: 0.011666s +- 0.000229s [0.011424s - 0.012052s]
    
```

### Function Plots
Code from [ActivationLayerTestBase.java:110](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/java/ActivationLayerTestBase.java#L110) executed in 0.00 seconds: 
```java
    return plot("Value Plot", plotData, x -> new double[]{x[0], x[1]});
```

Returns: 

![Result](etc/test.180.png)



Code from [ActivationLayerTestBase.java:114](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/java/ActivationLayerTestBase.java#L114) executed in 0.00 seconds: 
```java
    return plot("Derivative Plot", plotData, x -> new double[]{x[0], x[2]});
```

Returns: 

![Result](etc/test.181.png)



