# FullyConnectedLayer
## FullyConnectedLayerTest
### Batch Execution
Code from [BatchingTester.java:66](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/BatchingTester.java#L66) executed in 0.02 seconds: 
```java
    return test(reference, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (60#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (60#)}
```



Code from [SingleDerivativeTester.java:77](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/SingleDerivativeTester.java#L77) executed in 0.00 seconds: 
```java
    return test(component, inputPrototype);
```
Logging: 
```
    Inputs: [ -1.176, -1.516, -1.58 ]
    Inputs Statistics: {meanExponent=0.14992120333019235, negative=3, min=-1.58, max=-1.58, mean=-1.4240000000000002, count=3.0, positive=0, stdDev=0.1772982421420649, zeros=0}
    Output: [ 0.032243343324068974, 1.262515574592192, -1.4070974671673335 ]
    Outputs Statistics: {meanExponent=-0.41399966891241285, negative=1, min=-1.4070974671673335, max=-1.4070974671673335, mean=-0.03744618308369082, count=3.0, positive=2, stdDev=1.090978435702356, zeros=0}
    Feedback for input 0
    Inputs Values: [ -1.176, -1.516, -1.58 ]
    Value Statistics: {meanExponent=0.14992120333019235, negative=3, min=-1.58, max=-1.58, mean=-1.4240000000000002, count=3.0, positive=0, stdDev=0.1772982421420649, zeros=0}
    Implemented Feedback: [ [ -0.17852244086657743, -0.046278651463009954, -0.5385588173955337 ], [ -0.593119375090142, 0.1408119497229224, 0.5927097643887032 ], [ 0.681562037830178, -0.8997232887668626, 0.7227181225387388 ] ]
    Implemented Statistics: {meanExponent=-0.445609593676686, negative=5, min=0.
```
...[skipping 1923 bytes](etc/226.txt)...
```
    9999999988046, 0.0 ], [ 0.0, 0.0, -1.5800000000032455 ] ]
    Measured Statistics: {meanExponent=0.14992120333018294, negative=9, min=-1.5800000000032455, max=-1.5800000000032455, mean=-0.4746666666666644, count=27.0, positive=0, stdDev=0.6790398450099433, zeros=18}
    Gradient Error: [ [ -5.100364575127969E-13, 0.0, 0.0 ], [ 0.0, 1.7104095917375162E-12, 0.0 ], [ 0.0, 0.0, -5.100364575127969E-13 ], [ 8.15347789284715E-13, 0.0, 0.0 ], [ 0.0, 8.15347789284715E-13, 0.0 ], [ 0.0, 0.0, -1.4050982599655981E-12 ], [ 1.1954881529163686E-12, 0.0, 0.0 ], [ 0.0, 1.1954881529163686E-12, 0.0 ], [ 0.0, 0.0, -3.2454039455842576E-12 ] ]
    Error Statistics: {meanExponent=-11.968327292240843, negative=4, min=-3.2454039455842576E-12, max=-3.2454039455842576E-12, mean=2.2780131690456916E-15, count=27.0, positive=5, stdDev=8.636818474562325E-13, zeros=18}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 5.6627e-13 +- 9.3707e-13 [0.0000e+00 - 4.3007e-12] (36#)
    relativeTol: 1.2163e-12 +- 2.1870e-12 [3.4091e-14 - 9.6456e-12] (18#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=5.6627e-13 +- 9.3707e-13 [0.0000e+00 - 4.3007e-12] (36#), relativeTol=1.2163e-12 +- 2.1870e-12 [3.4091e-14 - 9.6456e-12] (18#)}
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
      "class": "com.simiacryptus.mindseye.layers.java.FullyConnectedLayer",
      "id": "d2f4ed13-e333-4d82-aebb-e7323733f19a",
      "isFrozen": false,
      "name": "FullyConnectedLayer/d2f4ed13-e333-4d82-aebb-e7323733f19a",
      "outputDims": [
        3
      ],
      "inputDims": [
        3
      ],
      "weights": [
        [
          -0.17852244086657743,
          -0.046278651463009954,
          -0.5385588173955337
        ],
        [
          -0.593119375090142,
          0.1408119497229224,
          0.5927097643887032
        ],
        [
          0.681562037830178,
          -0.8997232887668626,
          0.7227181225387388
        ]
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
    [[ -0.804, -0.556, 1.016 ]]
    --------------------
    Output: 
    [ 1.165773445442308, -0.9552022696568172, 0.8377362726852488 ]
    --------------------
    Derivative: 
    [ -0.7633599097251211, 0.14040233902148358, 0.5045568716020542 ]
```



### Input Learning
In this test, we use a network to learn this target input, given it's pre-evaluated output:

Code from [LearningTester.java:127](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L127) executed in 0.00 seconds: 
```java
    return Arrays.stream(input_target).map(x -> x.prettyPrint()).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [ -0.372, -0.876, 1.332 ]
```



First, we use a conjugate gradient descent method, which converges the fastest for purely linear functions.

Code from [LearningTester.java:225](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L225) executed in 0.12 seconds: 
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
    F(0.0) = LineSearchPoint{point=PointSample{avg=2.5938678661348806}, derivative=-6.7756282465520385}
    New Minimum: 2.5938678661348806 > 2.593867865457318
    F(1.0E-10) = LineSearchPoint{point=PointSample{avg=2.593867865457318}, derivative=-6.775628245543572}, delta = -6.775624505905853E-10
    New Minimum: 2.593867865457318 > 2.5938678613919417
    F(7.000000000000001E-10) = LineSearchPoint{point=PointSample{avg=2.5938678613919417}, derivative=-6.775628239492769}, delta = -4.742938930490936E-9
    New Minimum: 2.5938678613919417 > 2.593867832934303
    F(4.900000000000001E-9) = LineSearchPoint{point=PointSample{avg=2.593867832934303}, derivative=-6.775628197137145}, delta = -3.320057784250707E-8
    New Minimum: 2.593867832934303 > 2.593867633730838
    F(3.430000000000001E-8) = LineSearchPoint{point=PointSample{avg=2.593867633730838}, derivative=-6.77562790064779}, delta = -2.3240404267710346E-7
    New Minimum: 2.593867633730838 > 2.5938662393068292
    F(2.4010000000000004E-7) = LineSearchPoi
```
...[skipping 289664 bytes](etc/227.txt)...
```
    ration 249 complete. Error: 5.728112193965182E-9 Total: 239645335832519.6600; Orientation: 0.0000; Line Search: 0.0004
    F(0.0) = LineSearchPoint{point=PointSample{avg=5.728112193965182E-9}, derivative=-1.383370827298703E-10}
    New Minimum: 5.728112193965182E-9 > 5.630916555042664E-9
    F(0.8084126200381135) = LineSearchPoint{point=PointSample{avg=5.630916555042664E-9}, derivative=-1.0212338639062721E-10}, delta = -9.719563892251748E-11
    F(5.658888340266794) = LineSearchPoint{point=PointSample{avg=5.662530513376815E-9}, derivative=1.1515879164570859E-10}, delta = -6.558168058836668E-11
    5.662530513376815E-9 <= 5.728112193965182E-9
    New Minimum: 5.630916555042664E-9 > 5.514509139207006E-9
    F(3.088153234899361) = LineSearchPoint{point=PointSample{avg=5.514509139207006E-9}, derivative=-9.938445812853916E-22}, delta = -2.1360305475817564E-10
    Left bracket at 3.088153234899361
    Converged to left
    Iteration 250 complete. Error: 5.514509139207006E-9 Total: 239645336120347.6600; Orientation: 0.0000; Line Search: 0.0002
    
```

Returns: 

```
    5.514509139207006E-9
```



Training Converged

Next, we run the same optimization using L-BFGS, which is nearly ideal for purely second-order or quadratic functions.

Code from [LearningTester.java:249](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L249) executed in 0.15 seconds: 
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
    th(0)=2.5938678661348806;dx=-6.7756282465520385
    Armijo: th(2.154434690031884)=11.400669685781216; dx=14.951138843734263 delta=-8.806801819646335
    New Minimum: 2.5938678661348806 > 1.146156185763558
    WOLF (strong): th(1.077217345015942)=1.146156185763558; dx=4.08775529859111 delta=1.4477116803713226
    New Minimum: 1.146156185763558 > 0.811050063745665
    END: th(0.3590724483386473)=0.811050063745665; dx=-3.154500398170989 delta=1.7828178023892156
    Iteration 1 complete. Error: 0.811050063745665 Total: 239645339358272.6600; Orientation: 0.0001; Line Search: 0.0003
    LBFGS Accumulation History: 1 points
    th(0)=0.811050063745665;dx=-1.531924481763475
    New Minimum: 0.811050063745665 > 0.23844143177379443
    WOLF (strong): th(0.7735981389354633)=0.23844143177379443; dx=0.05154700112519775 delta=0.5726086319718706
    END: th(0.3867990694677316)=0.37162442373222276; dx=-0.7401887403191384 delta=0.43942564001344225
    Iteration 2 complete. Error: 0.2
```
...[skipping 146074 bytes](etc/228.txt)...
```
    874E-15
    END: th(7.437686708773461)=6.4565284168947874E-15; dx=-1.2276071947841825E-16 delta=9.93788291742131E-16
    Iteration 249 complete. Error: 6.4565284168947874E-15 Total: 239645485807242.5000; Orientation: 0.0000; Line Search: 0.0002
    LBFGS Accumulation History: 1 points
    th(0)=6.4565284168947874E-15;dx=-2.33692058514147E-16
    Armijo: th(16.024010258970613)=2.4776809872810448E-14; dx=2.5202958687092543E-15 delta=-1.832028145591566E-14
    Armijo: th(8.012005129485306)=1.0100427816601328E-14; dx=1.1433019065499611E-15 delta=-3.6438993997065406E-15
    New Minimum: 6.4565284168947874E-15 > 6.445330136332194E-15
    WOLF (strong): th(2.6706683764951022)=6.445330136332194E-15; dx=2.253059307618518E-16 delta=1.1198280562593686E-17
    New Minimum: 6.445330136332194E-15 > 6.338807152131635E-15
    END: th(0.6676670941237756)=6.338807152131635E-15; dx=-1.1894256033883533E-16 delta=1.1772126476315204E-16
    Iteration 250 complete. Error: 6.338807152131635E-15 Total: 239645486116728.5000; Orientation: 0.0000; Line Search: 0.0003
    
```

Returns: 

```
    6.338807152131635E-15
```



Training Converged

Code from [LearningTester.java:95](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L95) executed in 0.00 seconds: 
```java
    return TestUtil.compare(runs);
```

Returns: 

![Result](etc/test.138.png)



Code from [LearningTester.java:98](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L98) executed in 0.00 seconds: 
```java
    return TestUtil.compareTime(runs);
```

Returns: 

![Result](etc/test.139.png)



### Model Learning
In this test, attempt to train a network to emulate a randomized network given an example input/output. The target state is:

Code from [LearningTester.java:176](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L176) executed in 0.00 seconds: 
```java
    return network_target.state().stream().map(Arrays::toString).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [0.5927097643887032, -0.593119375090142, -0.5385588173955337, 0.7227181225387388, 0.1408119497229224, -0.8997232887668626, -0.046278651463009954, -0.17852244086657743, 0.681562037830178]
```



First, we use a conjugate gradient descent method, which converges the fastest for purely linear functions.

Code from [LearningTester.java:225](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L225) executed in 0.01 seconds: 
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
    F(0.0) = LineSearchPoint{point=PointSample{avg=1.1758643024476483}, derivative=-4.201730022307811}
    New Minimum: 1.1758643024476483 > 1.1758643020274755
    F(1.0E-10) = LineSearchPoint{point=PointSample{avg=1.1758643020274755}, derivative=-4.201730021557107}, delta = -4.201727854535875E-10
    New Minimum: 1.1758643020274755 > 1.1758642995064372
    F(7.000000000000001E-10) = LineSearchPoint{point=PointSample{avg=1.1758642995064372}, derivative=-4.2017300170528795}, delta = -2.941211052487347E-9
    New Minimum: 1.1758642995064372 > 1.1758642818591714
    F(4.900000000000001E-9) = LineSearchPoint{point=PointSample{avg=1.1758642818591714}, derivative=-4.2017299855232855}, delta = -2.058847692332222E-8
    New Minimum: 1.1758642818591714 > 1.1758641583283131
    F(3.430000000000001E-8) = LineSearchPoint{point=PointSample{avg=1.1758641583283131}, derivative=-4.201729764816129}, delta = -1.4411933513258646E-7
    New Minimum: 1.1758641583283131 > 1.1758632936124866
    F(2.4010000000000004E-7) = Li
```
...[skipping 1773 bytes](etc/229.txt)...
```
    nt=PointSample{avg=2.0543252740130517E-32}, derivative=5.120501344586771E-16}, delta = -1.1758643024476483
    Right bracket at 0.5597048340587109
    Converged to right
    Iteration 1 complete. Error: 2.0543252740130517E-32 Total: 239645624947086.3800; Orientation: 0.0001; Line Search: 0.0031
    Zero gradient: 2.7093809539328583E-16
    F(0.0) = LineSearchPoint{point=PointSample{avg=2.0543252740130517E-32}, derivative=-7.340745153534125E-32}
    New Minimum: 2.0543252740130517E-32 > 1.0271626370065257E-33
    F(0.5597048340587109) = LineSearchPoint{point=PointSample{avg=1.0271626370065257E-33}, derivative=1.468149030706825E-32}, delta = -1.9516090103123992E-32
    1.0271626370065257E-33 <= 2.0543252740130517E-32
    New Minimum: 1.0271626370065257E-33 > 0.0
    F(0.4664206950489257) = LineSearchPoint{point=PointSample{avg=0.0}, derivative=0.0}, delta = -2.0543252740130517E-32
    Right bracket at 0.4664206950489257
    Converged to right
    Iteration 2 complete. Error: 0.0 Total: 239645625809430.3800; Orientation: 0.0000; Line Search: 0.0005
    
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
    th(0)=0.5257137869409627;dx=-1.8785393834415853
    Armijo: th(2.154434690031884)=4.267814169435006; dx=5.352397653370078 delta=-3.7421003824940433
    New Minimum: 0.5257137869409627 > 0.4494412789950591
    WOLF (strong): th(1.077217345015942)=0.4494412789950591; dx=1.736929134964246 delta=0.07627250794590362
    New Minimum: 0.4494412789950591 > 0.06755124002723366
    END: th(0.3590724483386473)=0.06755124002723366; dx=-0.6733832106396413 delta=0.45816254691372904
    Iteration 1 complete. Error: 0.06755124002723366 Total: 239645631314358.3800; Orientation: 0.0001; Line Search: 0.0005
    LBFGS Accumulation History: 1 points
    th(0)=0.06755124002723366;dx=-0.2413816566041943
    New Minimum: 0.06755124002723366 > 0.009865283989509388
    WOLF (strong): th(0.7735981389354633)=0.009865283989509388; dx=0.0922449068258015 delta=0.05768595603772427
    New Minimum: 0.009865283989509388 > 0.006446650937261637
    END: th(0.3867990694677316)=0.006446650937261637; dx=-
```
...[skipping 13017 bytes](etc/230.txt)...
```
    5767670625E-33 delta=0.0
    Armijo: th(3.545161493800662E-11)=1.0271626370065257E-33; dx=-3.6703725767670625E-33 delta=0.0
    Armijo: th(2.447849602862362E-11)=1.0271626370065257E-33; dx=-3.6703725767670625E-33 delta=0.0
    WOLFE (weak): th(1.8991936573932117E-11)=1.0271626370065257E-33; dx=-3.6703725767670625E-33 delta=0.0
    WOLFE (weak): th(2.1735216301277867E-11)=1.0271626370065257E-33; dx=-3.6703725767670625E-33 delta=0.0
    WOLFE (weak): th(2.310685616495074E-11)=1.0271626370065257E-33; dx=-3.6703725767670625E-33 delta=0.0
    Armijo: th(2.379267609678718E-11)=1.0271626370065257E-33; dx=-3.6703725767670625E-33 delta=0.0
    Armijo: th(2.344976613086896E-11)=1.0271626370065257E-33; dx=-3.6703725767670625E-33 delta=0.0
    WOLFE (weak): th(2.327831114790985E-11)=1.0271626370065257E-33; dx=-3.6703725767670625E-33 delta=0.0
    mu /= nu: th(0)=1.0271626370065257E-33;th'(0)=-3.6703725767670625E-33;
    Iteration 24 failed, aborting. Error: 1.0271626370065257E-33 Total: 239645643087670.3400; Orientation: 0.0000; Line Search: 0.0024
    
```

Returns: 

```
    1.0271626370065257E-33
```



Training Converged

Code from [LearningTester.java:95](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L95) executed in 0.00 seconds: 
```java
    return TestUtil.compare(runs);
```

Returns: 

![Result](etc/test.140.png)



Code from [LearningTester.java:98](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L98) executed in 0.00 seconds: 
```java
    return TestUtil.compareTime(runs);
```

Returns: 

![Result](etc/test.141.png)



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
    	[3]
    Performance:
    	Evaluation performance: 0.000159s +- 0.000025s [0.000131s - 0.000199s]
    	Learning performance: 0.000181s +- 0.000035s [0.000131s - 0.000221s]
    
```

