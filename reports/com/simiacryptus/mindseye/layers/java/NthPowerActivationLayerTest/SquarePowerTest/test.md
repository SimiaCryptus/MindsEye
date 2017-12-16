# NthPowerActivationLayer
## SquarePowerTest
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
    	[ [ -1.524 ], [ -0.484 ], [ -0.664 ] ],
    	[ [ 0.464 ], [ -1.824 ], [ 0.604 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.10023697313592972, negative=4, min=0.604, max=0.604, mean=-0.5713333333333334, count=6.0, positive=2, stdDev=0.9079542328162191, zeros=0}
    Output: [
    	[ [ 2.322576 ], [ 0.234256 ], [ 0.44089600000000007 ] ],
    	[ [ 0.21529600000000002 ], [ 3.326976 ], [ 0.364816 ] ]
    ]
    Outputs Statistics: {meanExponent=-0.20047394627185944, negative=0, min=0.364816, max=0.364816, mean=1.1508026666666669, count=6.0, positive=6, stdDev=1.221048466996931, zeros=0}
    Feedback for input 0
    Inputs Values: [
    	[ [ -1.524 ], [ -0.484 ], [ -0.664 ] ],
    	[ [ 0.464 ], [ -1.824 ], [ 0.604 ] ]
    ]
    Value Statistics: {meanExponent=-0.10023697313592972, negative=4, min=0.604, max=0.604, mean=-0.5713333333333334, count=6.0, positive=2, stdDev=0.9079542328162191, zeros=0}
    Implemented Feedback: [ [ -3.048, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.928, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, -0.968, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, -3.
```
...[skipping 493 bytes](etc/309.txt)...
```
    .0, 0.0, 0.0, 0.0, -1.3278999999999375, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 1.2080999999997122 ] ]
    Measured Statistics: {meanExponent=0.20078952602967495, negative=4, min=1.2080999999997122, max=1.2080999999997122, mean=-0.19042777777773492, count=36.0, positive=2, stdDev=0.8549274542356469, zeros=30}
    Feedback Error: [ [ 1.0000000056287206E-4, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 9.999999998722142E-5, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 1.0000000014498411E-4, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 1.0000000107313056E-4, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 1.0000000006260557E-4, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 9.999999971221918E-5 ] ]
    Error Statistics: {meanExponent=-3.9999999988831156, negative=0, min=9.999999971221918E-5, max=9.999999971221918E-5, mean=1.6666666709528693E-5, count=36.0, positive=6, stdDev=3.7267799720838895E-5, zeros=30}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 1.6667e-05 +- 3.7268e-05 [0.0000e+00 - 1.0000e-04] (36#)
    relativeTol: 3.5781e-05 +- 1.5690e-05 [1.3706e-05 - 5.3876e-05] (6#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=1.6667e-05 +- 3.7268e-05 [0.0000e+00 - 1.0000e-04] (36#), relativeTol=3.5781e-05 +- 1.5690e-05 [1.3706e-05 - 5.3876e-05] (6#)}
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
      "id": "296a51f6-d52f-42bc-9b37-2bde713911c8",
      "isFrozen": false,
      "name": "NthPowerActivationLayer/296a51f6-d52f-42bc-9b37-2bde713911c8",
      "power": 2.0
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
    	[ [ -0.624 ], [ -0.828 ], [ 1.284 ] ],
    	[ [ 0.508 ], [ -0.772 ], [ -0.9 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 0.389376 ], [ 0.685584 ], [ 1.6486560000000001 ] ],
    	[ [ 0.258064 ], [ 0.5959840000000001 ], [ 0.81 ] ]
    ]
    --------------------
    Derivative: 
    [
    	[ [ -1.248 ], [ -1.656 ], [ 2.568 ] ],
    	[ [ 1.016 ], [ -1.544 ], [ -1.8 ] ]
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
    	[ [ 1.784 ], [ 0.252 ], [ -1.412 ], [ -1.168 ], [ 1.524 ], [ 1.952 ], [ -0.012 ], [ -0.424 ], ... ],
    	[ [ 0.108 ], [ -1.516 ], [ -1.552 ], [ -1.048 ], [ -1.84 ], [ -1.532 ], [ -0.696 ], [ -1.2 ], ... ],
    	[ [ -1.716 ], [ 1.324 ], [ 0.672 ], [ 0.384 ], [ -0.836 ], [ -1.916 ], [ 0.628 ], [ -0.7 ], ... ],
    	[ [ 1.276 ], [ 0.896 ], [ -1.56 ], [ 1.944 ], [ -0.008 ], [ 1.096 ], [ 0.172 ], [ 0.212 ], ... ],
    	[ [ 0.284 ], [ 1.032 ], [ 1.448 ], [ -0.984 ], [ 1.064 ], [ 1.168 ], [ -1.4 ], [ -1.6 ], ... ],
    	[ [ 0.928 ], [ -1.992 ], [ -0.212 ], [ 1.98 ], [ 1.18 ], [ -1.652 ], [ 1.332 ], [ 0.912 ], ... ],
    	[ [ -1.036 ], [ -1.836 ], [ -0.56 ], [ -0.792 ], [ -1.312 ], [ -1.188 ], [ -0.832 ], [ 0.892 ], ... ],
    	[ [ -0.18 ], [ -0.508 ], [ -0.748 ], [ 1.488 ], [ -1.408 ], [ 0.056 ], [ 1.224 ], [ -0.04 ], ... ],
    	...
    ]
```



First, we use a conjugate gradient descent method, which converges the fastest for purely linear functions.

Code from [LearningTester.java:225](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L225) executed in 1.97 seconds: 
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
    F(0.0) = LineSearchPoint{point=PointSample{avg=2.8998286757116394}, derivative=-0.00804150736440372}
    New Minimum: 2.8998286757116394 > 2.8998286757108405
    F(1.0E-10) = LineSearchPoint{point=PointSample{avg=2.8998286757108405}, derivative=-0.008041507364401129}, delta = -7.989164885202626E-13
    New Minimum: 2.8998286757108405 > 2.899828675706011
    F(7.000000000000001E-10) = LineSearchPoint{point=PointSample{avg=2.899828675706011}, derivative=-0.008041507364385584}, delta = -5.628386645639694E-12
    New Minimum: 2.899828675706011 > 2.899828675672242
    F(4.900000000000001E-9) = LineSearchPoint{point=PointSample{avg=2.899828675672242}, derivative=-0.008041507364276768}, delta = -3.9397374251848305E-11
    New Minimum: 2.899828675672242 > 2.8998286754358267
    F(3.430000000000001E-8) = LineSearchPoint{point=PointSample{avg=2.8998286754358267}, derivative=-0.008041507363515062}, delta = -2.7581270600762764E-10
    New Minimum: 2.8998286754358267 > 2.899828673780885
    F(2.4010000000000004
```
...[skipping 283045 bytes](etc/310.txt)...
```
    rivative=-4.688658460286222E-11}, delta = -1.6588701878188054E-8
    F(1945.7348759152635) = LineSearchPoint{point=PointSample{avg=3.121071954008999E-4}, derivative=1.0625744870875067E-10}, delta = 3.3013270176222246E-8
    F(149.67191353194335) = LineSearchPoint{point=PointSample{avg=3.1206436591201843E-4}, derivative=-5.869474792855743E-11}, delta = -9.816218705233342E-9
    New Minimum: 3.120490339428249E-4 > 3.1204874759901085E-4
    F(1047.7033947236034) = LineSearchPoint{point=PointSample{avg=3.1204874759901085E-4}, derivative=2.387880208173538E-11}, delta = -2.5434531712817832E-8
    3.1204874759901085E-4 <= 3.1207418213072366E-4
    New Minimum: 3.1204874759901085E-4 > 3.1204564404535974E-4
    F(788.0597781388881) = LineSearchPoint{point=PointSample{avg=3.1204564404535974E-4}, derivative=2.4740061300321567E-14}, delta = -2.8538085363924957E-8
    Right bracket at 788.0597781388881
    Converged to right
    Iteration 250 complete. Error: 3.1204564404535974E-4 Total: 239703751332949.2500; Orientation: 0.0004; Line Search: 0.0095
    
```

Returns: 

```
    3.1204564404535974E-4
```



This training run resulted in the following regressed input:

Code from [LearningTester.java:144](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L144) executed in 0.00 seconds: 
```java
    return Arrays.stream(input_gd).map(x -> x.prettyPrint()).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [
    	[ [ -1.784 ], [ -0.25168102167087514 ], [ 1.412 ], [ 1.168 ], [ -1.524 ], [ 1.952000000000016 ], [ 0.08589076509768227 ], [ 0.4240000000075744 ], ... ],
    	[ [ -0.12244289732618562 ], [ 1.516 ], [ -1.552 ], [ -1.048 ], [ 1.8400000000000003 ], [ 1.532 ], [ -0.6960000000000001 ], [ 1.2 ], ... ],
    	[ [ -1.716 ], [ -1.324 ], [ 0.6720000000000002 ], [ -0.38399999913931626 ], [ -0.836 ], [ -1.9159999999999997 ], [ 0.6280000000000002 ], [ -0.6999999999999998 ], ... ],
    	[ [ -1.276 ], [ 0.896 ], [ -1.56 ], [ 1.9440000000000006 ], [ 0.08812965666327 ], [ 1.096 ], [ -0.17374257251979824 ], [ -0.21220778386732062 ], ... ],
    	[ [ -0.28400309225928616 ], [ 1.032 ], [ -1.448 ], [ -0.984 ], [ 1.064 ], [ 1.168 ], [ 1.4 ], [ -1.6 ], ... ],
    	[ [ 0.928 ], [ 1.9919791139473793 ], [ 0.21099789612933353 ], [ 1.9800000394102744 ], [ 1.18 ], [ -1.652 ], [ -1.332 ], [ -0.912 ], ... ],
    	[ [ -1.036 ], [ 1.836 ], [ -0.5600000000000003 ], [ -0.7919999999999999 ], [ 1.312 ], [ 1.188 ], [ -0.832 ], [ -0.892 ], ... ],
    	[ [ 0.18091693757010202 ], [ -0.5079999999999991 ], [ -0.7479999999999999 ], [ -1.488 ], [ -1.408 ], [ -0.09699042327755775 ], [ -1.224 ], [ 0.057607269878124936 ], ... ],
    	...
    ]
```



Next, we run the same optimization using L-BFGS, which is nearly ideal for purely second-order or quadratic functions.

Code from [LearningTester.java:249](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L249) executed in 1.60 seconds: 
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
    th(0)=2.8998286757116394;dx=-0.00804150736440372
    New Minimum: 2.8998286757116394 > 2.882563800429017
    WOLFE (weak): th(2.154434690031884)=2.882563800429017; dx=-0.007985829940573836 delta=0.017264875282622327
    New Minimum: 2.882563800429017 > 2.86541857606054
    WOLFE (weak): th(4.308869380063768)=2.86541857606054; dx=-0.00793043316450261 delta=0.0344100996510992
    New Minimum: 2.86541857606054 > 2.798022137148447
    WOLFE (weak): th(12.926608140191302)=2.798022137148447; dx=-0.007711636880635448 delta=0.10180653856319255
    New Minimum: 2.798022137148447 > 2.517291164263168
    END: th(51.70643256076521)=2.517291164263168; dx=-0.006781123468390416 delta=0.38253751144847126
    Iteration 1 complete. Error: 2.517291164263168 Total: 239703780955882.2500; Orientation: 0.0005; Line Search: 0.0066
    LBFGS Accumulation History: 1 points
    th(0)=2.517291164263168;dx=-0.005756346476749356
    New Minimum: 2.517291164263168 > 1.9636650741497603
    END: th(111
```
...[skipping 160108 bytes](etc/311.txt)...
```
    3.096089450126573E-4; dx=2.063328681629205E-12 delta=2.06308755257463E-9
    New Minimum: 3.096089450126573E-4 > 3.0960849368910333E-4
    END: th(937.5000000000006)=3.0960849368910333E-4; dx=-1.1004092067276513E-12 delta=2.5144111065163524E-9
    Iteration 249 complete. Error: 3.0960849368910333E-4 Total: 239705359607066.6600; Orientation: 0.0005; Line Search: 0.0033
    LBFGS Accumulation History: 1 points
    th(0)=3.0960849368910333E-4;dx=-7.2476550730253965E-12
    Armijo: th(2019.7825219048923)=3.0962010207652007E-4; dx=1.8739348630348058E-11 delta=-1.1608387416736036E-8
    New Minimum: 3.0960849368910333E-4 > 3.09607736870906E-4
    WOLF (strong): th(1009.8912609524461)=3.09607736870906E-4; dx=5.748094436334093E-12 delta=7.568181973585837E-10
    New Minimum: 3.09607736870906E-4 > 3.096067831334721E-4
    END: th(336.63042031748205)=3.096067831334721E-4; dx=-2.9152389763119135E-12 delta=1.7105556312222749E-9
    Iteration 250 complete. Error: 3.096067831334721E-4 Total: 239705365090621.6200; Orientation: 0.0005; Line Search: 0.0044
    
```

Returns: 

```
    3.096067831334721E-4
```



This training run resulted in the following regressed input:

Code from [LearningTester.java:154](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L154) executed in 0.00 seconds: 
```java
    return Arrays.stream(input_lbgfs).map(x -> x.prettyPrint()).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [
    	[ [ -1.7840000007619672 ], [ -0.25199997788432954 ], [ 1.412 ], [ 1.168 ], [ -1.524 ], [ 1.9519996659556575 ], [ 0.06097768911978875 ], [ 0.424 ], ... ],
    	[ [ -0.1104626402610834 ], [ 1.516 ], [ -1.552 ], [ -1.048 ], [ 1.84 ], [ 1.532 ], [ -0.696 ], [ 1.2 ], ... ],
    	[ [ -1.716 ], [ -1.324 ], [ 0.672 ], [ -0.38399999999999995 ], [ -0.836 ], [ -1.9159999999725905 ], [ 0.628 ], [ -0.7 ], ... ],
    	[ [ -1.276 ], [ 0.896 ], [ -1.56 ], [ 1.9440000092797765 ], [ 0.0614652331854647 ], [ 1.096 ], [ -0.1720277832311907 ], [ 0.21200043239937738 ], ... ],
    	[ [ -0.2840000000087621 ], [ 1.032 ], [ -1.448 ], [ -0.984 ], [ 1.064 ], [ 1.168 ], [ 1.4 ], [ -1.6 ], ... ],
    	[ [ 0.928 ], [ 1.99200024987566 ], [ -0.21200043230316445 ], [ 1.9799990253790798 ], [ 1.18 ], [ -1.652 ], [ -1.332 ], [ -0.912 ], ... ],
    	[ [ -1.036 ], [ 1.8359999963572786 ], [ -0.56 ], [ -0.792 ], [ 1.312 ], [ 1.188 ], [ -0.832 ], [ -0.892 ], ... ],
    	[ [ 0.18000874007013107 ], [ -0.508 ], [ -0.748 ], [ -1.488 ], [ -1.408 ], [ 0.07443348931649994 ], [ -1.224 ], [ -0.06793557017446814 ], ... ],
    	...
    ]
```



Code from [LearningTester.java:95](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L95) executed in 0.01 seconds: 
```java
    return TestUtil.compare(runs);
```

Returns: 

![Result](etc/test.217.png)



Code from [LearningTester.java:98](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L98) executed in 0.00 seconds: 
```java
    return TestUtil.compareTime(runs);
```

Returns: 

![Result](etc/test.218.png)



### Performance
Now we execute larger-scale runs to benchmark performance:

Code from [PerformanceTester.java:66](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/PerformanceTester.java#L66) executed in 0.12 seconds: 
```java
    test(component, inputPrototype);
```
Logging: 
```
    100 batches
    Input Dimensions:
    	[100, 100, 1]
    Performance:
    	Evaluation performance: 0.003042s +- 0.000655s [0.001752s - 0.003510s]
    	Learning performance: 0.011877s +- 0.001345s [0.010669s - 0.014375s]
    
```

### Function Plots
Code from [ActivationLayerTestBase.java:110](../../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/java/ActivationLayerTestBase.java#L110) executed in 0.00 seconds: 
```java
    return plot("Value Plot", plotData, x -> new double[]{x[0], x[1]});
```

Returns: 

![Result](etc/test.219.png)



Code from [ActivationLayerTestBase.java:114](../../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/java/ActivationLayerTestBase.java#L114) executed in 0.00 seconds: 
```java
    return plot("Derivative Plot", plotData, x -> new double[]{x[0], x[2]});
```

Returns: 

![Result](etc/test.220.png)



