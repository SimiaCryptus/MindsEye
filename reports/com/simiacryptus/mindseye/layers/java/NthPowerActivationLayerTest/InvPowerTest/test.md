# NthPowerActivationLayer
## InvPowerTest
### Batch Execution
Code from [BatchingTester.java:66](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/BatchingTester.java#L66) executed in 0.00 seconds: 
```java
    return test(reference, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (120#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (120#)}
```



Code from [SingleDerivativeTester.java:77](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/SingleDerivativeTester.java#L77) executed in 0.00 seconds: 
```java
    return test(component, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ -0.948 ], [ 1.02 ], [ -0.864 ] ],
    	[ [ -0.488 ], [ 0.556 ], [ -1.74 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.06733898109229251, negative=4, min=-1.74, max=-1.74, mean=-0.4106666666666667, count=6.0, positive=2, stdDev=0.9352893788674296, zeros=0}
    Output: [
    	[ [ -1.0548523206751055 ], [ 0.9803921568627451 ], [ -1.1574074074074074 ] ],
    	[ [ -2.0491803278688523 ], [ 1.7985611510791366 ], [ -0.5747126436781609 ] ]
    ]
    Outputs Statistics: {meanExponent=0.06733898109229251, negative=4, min=-0.5747126436781609, max=-0.5747126436781609, mean=-0.34286656528127407, count=6.0, positive=2, stdDev=1.3211349962542216, zeros=0}
    Feedback for input 0
    Inputs Values: [
    	[ [ -0.948 ], [ 1.02 ], [ -0.864 ] ],
    	[ [ -0.488 ], [ 0.556 ], [ -1.74 ] ]
    ]
    Value Statistics: {meanExponent=-0.06733898109229251, negative=4, min=-1.74, max=-1.74, mean=-0.4106666666666667, count=6.0, positive=2, stdDev=0.9352893788674296, zeros=0}
    Implemented Feedback: [ [ -1.1127134184336558, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, -4.19914001
```
...[skipping 675 bytes](etc/297.txt)...
```
    , 0.0, 0.0, -1.339746970028255, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, -0.3303136063448342 ] ]
    Measured Statistics: {meanExponent=0.13469285673598494, negative=6, min=-0.3303136063448342, max=-0.3303136063448342, mean=-0.31050575356951954, count=36.0, positive=0, stdDev=0.8928030992395727, zeros=30}
    Feedback Error: [ [ -1.1738721595588864E-4, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, -8.606558764050476E-4, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 9.422299574890491E-5, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 5.81697934444847E-4, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, -1.550633067186613E-4, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, -1.8983541293426942E-5 ] ]
    Error Statistics: {meanExponent=-3.7979681594541677, negative=4, min=-1.8983541293426942E-5, max=-1.8983541293426942E-5, mean=-1.3226916949424239E-5, count=36.0, positive=2, stdDev=1.763728304776205E-4, zeros=30}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 5.0778e-05 +- 1.6942e-04 [0.0000e+00 - 8.6066e-04] (36#)
    relativeTol: 6.3460e-05 +- 2.5107e-05 [2.8736e-05 - 1.0247e-04] (6#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=5.0778e-05 +- 1.6942e-04 [0.0000e+00 - 8.6066e-04] (36#), relativeTol=6.3460e-05 +- 2.5107e-05 [2.8736e-05 - 1.0247e-04] (6#)}
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
      "class": "com.simiacryptus.mindseye.layers.java.NthPowerActivationLayer",
      "id": "f9ef8644-212d-4ce2-a5db-2605f447e265",
      "isFrozen": false,
      "name": "NthPowerActivationLayer/f9ef8644-212d-4ce2-a5db-2605f447e265",
      "power": -1.0
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
    	[ [ -0.228 ], [ -0.244 ], [ -0.488 ] ],
    	[ [ -1.164 ], [ -1.444 ], [ 1.792 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ -4.385964912280701 ], [ -4.098360655737705 ], [ -2.0491803278688523 ] ],
    	[ [ -0.859106529209622 ], [ -0.6925207756232687 ], [ 0.5580357142857143 ] ]
    ]
    --------------------
    Derivative: 
    [
    	[ [ -19.236688211757464 ], [ -16.79656006449879 ], [ -4.199140016124698 ] ],
    	[ [ -0.7380640285306032 ], [ -0.4795850246698537 ], [ -0.3114038584183673 ] ]
    ]
```



### Input Learning
In this test, we use a network to learn this target input, given it's pre-evaluated output:

Code from [LearningTester.java:127](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L127) executed in 0.00 seconds: 
```java
    return Arrays.stream(input_target).map(x -> x.prettyPrint()).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [
    	[ [ 0.98 ], [ 1.384 ], [ -0.616 ], [ 0.88 ], [ -0.552 ], [ 1.516 ], [ -1.636 ], [ 0.364 ], ... ],
    	[ [ -1.832 ], [ 1.932 ], [ -1.552 ], [ 1.38 ], [ -1.5 ], [ -1.54 ], [ 1.232 ], [ 0.332 ], ... ],
    	[ [ -1.908 ], [ 0.612 ], [ -1.632 ], [ -1.376 ], [ 1.576 ], [ -1.26 ], [ 1.604 ], [ 0.848 ], ... ],
    	[ [ 1.048 ], [ -0.924 ], [ 1.256 ], [ 1.752 ], [ 0.468 ], [ 1.38 ], [ 1.632 ], [ -1.292 ], ... ],
    	[ [ -0.504 ], [ 0.856 ], [ -0.956 ], [ -1.732 ], [ -1.448 ], [ 1.548 ], [ -0.784 ], [ -1.432 ], ... ],
    	[ [ 1.8 ], [ 0.648 ], [ 0.868 ], [ 1.064 ], [ -1.232 ], [ -1.144 ], [ -1.952 ], [ 1.88 ], ... ],
    	[ [ 1.992 ], [ -1.652 ], [ -1.116 ], [ 1.552 ], [ -1.964 ], [ -0.484 ], [ 1.532 ], [ -1.66 ], ... ],
    	[ [ -1.404 ], [ 1.272 ], [ 1.7 ], [ -1.74 ], [ 1.176 ], [ 0.756 ], [ -1.46 ], [ -1.3 ], ... ],
    	...
    ]
```



First, we use a conjugate gradient descent method, which converges the fastest for purely linear functions.

Code from [LearningTester.java:225](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L225) executed in 7.71 seconds: 
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
    F(0.0) = LineSearchPoint{point=PointSample{avg=4.947816877297777}, derivative=-0.16454397465673495}
    New Minimum: 4.947816877297777 > 4.947816877281317
    F(1.0E-10) = LineSearchPoint{point=PointSample{avg=4.947816877281317}, derivative=-0.16454397465268292}, delta = -1.645972247388272E-11
    New Minimum: 4.947816877281317 > 4.947816877182599
    F(7.000000000000001E-10) = LineSearchPoint{point=PointSample{avg=4.947816877182599}, derivative=-0.16454397462837053}, delta = -1.1517808928829254E-10
    New Minimum: 4.947816877182599 > 4.947816876491542
    F(4.900000000000001E-9) = LineSearchPoint{point=PointSample{avg=4.947816876491542}, derivative=-0.16454397445818403}, delta = -8.062350786985917E-10
    New Minimum: 4.947816876491542 > 4.947816871653949
    F(3.430000000000001E-8) = LineSearchPoint{point=PointSample{avg=4.947816871653949}, derivative=-0.16454397326687834}, delta = -5.6438285156446E-9
    New Minimum: 4.947816871653949 > 4.947816837790797
    F(2.4010000000000004E-7) = LineSearc
```
...[skipping 389634 bytes](etc/298.txt)...
```
    9355584}, derivative=9.924701085662746E-10}, delta = 1.9854447153022647E-6
    F(534.7609180072891) = LineSearchPoint{point=PointSample{avg=2.46943826966518}, derivative=-4.37643506110531E-10}, delta = -2.7782566291989497E-7
    F(3743.3264260510236) = LineSearchPoint{point=PointSample{avg=2.4694382633562544}, derivative=3.901382788855171E-10}, delta = -2.8413458874609887E-7
    2.4694382633562544 <= 2.469438547490843
    New Minimum: 2.469438100204913 > 2.469437938662796
    F(2272.630224796294) = LineSearchPoint{point=PointSample{avg=2.469437938662796}, derivative=4.301084888039882E-11}, delta = -6.088280470883944E-7
    Right bracket at 2272.630224796294
    New Minimum: 2.469437938662796 > 2.469437935071952
    F(2121.290223571178) = LineSearchPoint{point=PointSample{avg=2.469437935071952}, derivative=4.346850626978708E-12}, delta = -6.124188911549311E-7
    Right bracket at 2121.290223571178
    Converged to right
    Iteration 250 complete. Error: 2.469437935071952 Total: 239682248790393.7800; Orientation: 0.0003; Line Search: 0.0226
    
```

Returns: 

```
    2.469437935071952
```



This training run resulted in the following regressed input:

Code from [LearningTester.java:144](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L144) executed in 0.01 seconds: 
```java
    return Arrays.stream(input_gd).map(x -> x.prettyPrint()).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [
    	[ [ -240.34453731032636 ], [ 123.60753916079014 ], [ 527.5221467143926 ], [ -535.9936923374679 ], [ 935.269802204705 ], [ -495.02121583046153 ], [ 337.18381309588864 ], [ -742.9084383446703 ], ... ],
    	[ [ 351.9375149090975 ], [ -431.2218810584039 ], [ 492.4058985241409 ], [ -488.41283979433996 ], [ 447.3523761490696 ], [ -155.57204362969037 ], [ -447.27371025321156 ], [ -762.9647986904539 ], ... ],
    	[ [ -138.49192150675282 ], [ -6283.667237737342 ], [ 488.5469248182455 ], [ -124.5237360294916 ], [ -487.29268216652355 ], [ 419.9093195148131 ], [ 167.06580156599725 ], [ -553.7373971055375 ], ... ],
    	[ [ -521.4735849251309 ], [ 540.0896185245384 ], [ -520.3393834952284 ], [ 153.74797404541803 ], [ -2190.019276561305 ], [ 129.44573809913427 ], [ -469.1373326141713 ], [ -102.91920709796784 ], ... ],
    	[ [ -95280.16177651874 ], [ -572.7131066193903 ], [ 500.4954476323599 ], [ -179.07504633762764 ], [ 442.59418338433983 ], [ -495.13619940088165 ], [ 466.768728716703 ], [ 49.850514689343754 ], ... ],
    	[ [ -216.4811197254886 ], [ 140.12046577442953 ], [ -175.5681862777965 ], [ -483.91891727486956 ], [ -91.77381796047918 ], [ 529.7602746046894 ], [ 387.0459239901672 ], [ -379.0607324838335 ], ... ],
    	[ [ 212.2612729637479 ], [ 48.022896718198396 ], [ 535.823806461868 ], [ -325.9210754241148 ], [ 463.50813326752177 ], [ 39377.76353372937 ], [ -488.18847142381816 ], [ 316.6461268955079 ], ... ],
    	[ [ 437.8186598840672 ], [ -518.5953195830045 ], [ 124.47848843453988 ], [ -180.8926820186128 ], [ -487.81532500804667 ], [ -467.2183459088319 ], [ 446.59412610959635 ], [ -111.46075939486407 ], ... ],
    	...
    ]
```



Next, we run the same optimization using L-BFGS, which is nearly ideal for purely second-order or quadratic functions.

Code from [LearningTester.java:249](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L249) executed in 3.21 seconds: 
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
    th(0)=4.947816877297777;dx=-0.16454397465673495
    New Minimum: 4.947816877297777 > 4.665262062705238
    END: th(2.154434690031884)=4.665262062705238; dx=-0.10547311023368011 delta=0.2825548145925394
    Iteration 1 complete. Error: 4.665262062705238 Total: 239682277090172.7200; Orientation: 0.0006; Line Search: 0.0055
    LBFGS Accumulation History: 1 points
    th(0)=4.665262062705238;dx=-0.07154076568451577
    New Minimum: 4.665262062705238 > 4.391915544794528
    END: th(4.641588833612779)=4.391915544794528; dx=-0.048659485949702874 delta=0.27334651791070996
    Iteration 2 complete. Error: 4.391915544794528 Total: 239682285524962.7200; Orientation: 0.0005; Line Search: 0.0058
    LBFGS Accumulation History: 1 points
    th(0)=4.391915544794528;dx=-0.03420087002133031
    New Minimum: 4.391915544794528 > 4.11060838421965
    END: th(10.000000000000002)=4.11060838421965; dx=-0.023204092729163194 delta=0.2813071605748778
    Iteration 3 complete. Error: 4.110608384
```
...[skipping 117163 bytes](etc/299.txt)...
```
    6 Total: 239685455393453.5600; Orientation: 0.0005; Line Search: 0.0108
    LBFGS Accumulation History: 1 points
    th(0)=1.9337757188028586;dx=-5.293033302988841E-5
    New Minimum: 1.9337757188028586 > 1.9328220849506905
    WOLF (strong): th(34.556344994423384)=1.9328220849506905; dx=2.813223705905631E-5 delta=9.536338521680676E-4
    END: th(17.278172497211692)=1.9330073351289716; dx=-3.396635146061392E-5 delta=7.683836738869854E-4
    Iteration 249 complete. Error: 1.9328220849506905 Total: 239685467371095.5300; Orientation: 0.0005; Line Search: 0.0092
    LBFGS Accumulation History: 1 points
    th(0)=1.9330073351289716;dx=-4.3519389089439834E-5
    New Minimum: 1.9330073351289716 > 1.9323293855352455
    WOLF (strong): th(37.22469420834769)=1.9323293855352455; dx=1.988111374398571E-5 delta=6.779495937261348E-4
    END: th(18.612347104173846)=1.9323897125820535; dx=-2.1247745675809062E-5 delta=6.176225469181063E-4
    Iteration 250 complete. Error: 1.9323293855352455 Total: 239685478432816.5300; Orientation: 0.0005; Line Search: 0.0083
    
```

Returns: 

```
    1.9323897125820535
```



This training run resulted in the following regressed input:

Code from [LearningTester.java:154](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L154) executed in 0.04 seconds: 
```java
    return Arrays.stream(input_lbgfs).map(x -> x.prettyPrint()).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [
    	[ [ 1.2919843902542154 ], [ 1.164684944703124 ], [ 2.4447415677039483 ], [ -2.1149714951103187 ], [ -0.9454606748892129 ], [ -1.8620208690362086 ], [ 2.245386074410491 ], [ -2.467845997242889 ], ... ],
    	[ [ 2.1385647094228593 ], [ -1.9044937987672654 ], [ 1.8561873384985028 ], [ -1.9336043914098355 ], [ 1.988233128055876 ], [ -1.1972756065379442 ], [ -2.1026020218829435 ], [ -2.5373544186864954 ], ... ],
    	[ [ -1.4109206422676335 ], [ 0.6135371641815605 ], [ 1.8300201489994112 ], [ -1.1572860015461672 ], [ -1.8626668423171304 ], [ 2.160257676609162 ], [ 1.2196311682997871 ], [ -2.0996874994781813 ], ... ],
    	[ [ -2.0240796470869613 ], [ 2.06794455100497 ], [ -1.9007925137435095 ], [ 1.3162558046756736 ], [ 1.0379252895389166 ], [ 1.1481900507352971 ], [ -1.8915055018567832 ], [ -1.1301937422787607 ], ... ],
    	[ [ -0.504003680318593 ], [ -2.0397790354090732 ], [ 2.1406343214907144 ], [ -1.2607787595869466 ], [ 1.8594910657327213 ], [ -1.8431249764686324 ], [ 2.3873344195444437 ], [ -1.4910265421263582 ], ... ],
    	[ [ 1.8563665984521007 ], [ 0.6480253472771131 ], [ 0.8581443196887267 ], [ -2.1062536895264223 ], [ -1.0982093826532353 ], [ 1.9272507704948676 ], [ 1.7899727053985417 ], [ -2.0492441551342697 ], ... ],
    	[ [ 1.2941248546093094 ], [ -1.9322327528034677 ], [ 1.9366936743069114 ], [ -2.31421850386574 ], [ 1.817451396879392 ], [ -0.4839999999999852 ], [ -1.8758378502879054 ], [ 2.30566746013588 ], ... ],
    	[ [ 2.048463816279606 ], [ -1.8961807390560883 ], [ 1.3456064792525995 ], [ -1.260775501526673 ], [ -2.030945603225822 ], [ -2.418223357154879 ], [ 2.0047300264659924 ], [ -1.1212751420214686 ], ... ],
    	...
    ]
```



Code from [LearningTester.java:95](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L95) executed in 0.00 seconds: 
```java
    return TestUtil.compare(runs);
```

Returns: 

![Result](etc/test.201.png)



Code from [LearningTester.java:98](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L98) executed in 0.00 seconds: 
```java
    return TestUtil.compareTime(runs);
```

Returns: 

![Result](etc/test.202.png)



### Performance
Now we execute larger-scale runs to benchmark performance:

Code from [PerformanceTester.java:66](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/PerformanceTester.java#L66) executed in 0.31 seconds: 
```java
    test(component, inputPrototype);
```
Logging: 
```
    100 batches
    Input Dimensions:
    	[100, 100, 1]
    Performance:
    	Evaluation performance: 0.019137s +- 0.001256s [0.016978s - 0.020714s]
    	Learning performance: 0.013620s +- 0.004251s [0.010869s - 0.022007s]
    
```

### Function Plots
Code from [ActivationLayerTestBase.java:110](../../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/java/ActivationLayerTestBase.java#L110) executed in 0.00 seconds: 
```java
    return plot("Value Plot", plotData, x -> new double[]{x[0], x[1]});
```

Returns: 

![Result](etc/test.203.png)



Code from [ActivationLayerTestBase.java:114](../../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/java/ActivationLayerTestBase.java#L114) executed in 0.00 seconds: 
```java
    return plot("Derivative Plot", plotData, x -> new double[]{x[0], x[2]});
```

Returns: 

![Result](etc/test.204.png)



