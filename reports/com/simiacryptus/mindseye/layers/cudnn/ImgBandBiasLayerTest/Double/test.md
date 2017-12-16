# ImgBandBiasLayer
## Double
### Batch Execution
Code from [BatchingTester.java:66](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/BatchingTester.java#L66) executed in 0.01 seconds: 
```java
    return test(reference, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (360#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (360#)}
```



Code from [SingleDerivativeTester.java:77](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/SingleDerivativeTester.java#L77) executed in 0.03 seconds: 
```java
    return test(component, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ 0.488, 0.592 ], [ -1.472, 1.004 ], [ -1.86, -0.836 ] ],
    	[ [ -1.132, 1.048 ], [ -1.02, 1.668 ], [ -1.156, 0.88 ] ],
    	[ [ 0.072, 1.184 ], [ 1.056, -0.312 ], [ -0.78, 1.296 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.0784583876527344, negative=8, min=1.296, max=1.296, mean=0.03999999999999998, count=18.0, positive=10, stdDev=1.0830442691270237, zeros=0}
    Output: [
    	[ [ 0.472, 2.172 ], [ -1.488, 2.584 ], [ -1.8760000000000001, 0.7440000000000001 ] ],
    	[ [ -1.148, 2.628 ], [ -1.036, 3.248 ], [ -1.172, 2.46 ] ],
    	[ [ 0.055999999999999994, 2.7640000000000002 ], [ 1.04, 1.268 ], [ -0.796, 2.8760000000000003 ] ]
    ]
    Outputs Statistics: {meanExponent=0.10424941007708878, negative=6, min=2.8760000000000003, max=2.8760000000000003, mean=0.8220000000000002, count=18.0, positive=12, stdDev=1.7037815457257293, zeros=0}
    Feedback for input 0
    Inputs Values: [
    	[ [ 0.488, 0.592 ], [ -1.472, 1.004 ], [ -1.86, -0.836 ] ],
    	[ [ -1.132, 1.048 ], [ -1.02, 1.668 ], [ -1.156, 0.88 ] ],
    	[ [ 0.072, 1.184 ], [ 1.056, -0.
```
...[skipping 2761 bytes](etc/99.txt)...
```
    9999999999998899, 0.9999999999998899, 0.9999999999998899, 0.9999999999998899, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ] ]
    Measured Statistics: {meanExponent=-2.0520352750816186E-13, negative=0, min=0.9999999999976694, max=0.9999999999976694, mean=0.49999999999976374, count=36.0, positive=18, stdDev=0.49999999999976374, zeros=18}
    Gradient Error: [ [ -1.1013412404281553E-13, -1.1013412404281553E-13, 2.864375403532904E-14, -1.1013412404281553E-13, -1.1013412404281553E-13, -1.1013412404281553E-13, -1.1013412404281553E-13, -1.1013412404281553E-13, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ] ]
    Error Statistics: {meanExponent=-12.479874305602943, negative=15, min=-2.3305801732931286E-12, max=-2.3305801732931286E-12, mean=-2.3624929173454094E-13, count=36.0, positive=3, stdDev=9.743612448831759E-13, zeros=18}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 9.4464e-14 +- 4.3831e-13 [0.0000e+00 - 2.3306e-12] (360#)
    relativeTol: 4.7232e-13 +- 5.2869e-13 [1.4322e-14 - 1.1653e-12] (36#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=9.4464e-14 +- 4.3831e-13 [0.0000e+00 - 2.3306e-12] (360#), relativeTol=4.7232e-13 +- 5.2869e-13 [1.4322e-14 - 1.1653e-12] (36#)}
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
      "class": "com.simiacryptus.mindseye.layers.cudnn.ImgBandBiasLayer",
      "id": "ae848dc6-db6f-46ed-98d5-6c0bc9b39b53",
      "isFrozen": false,
      "name": "ImgBandBiasLayer/ae848dc6-db6f-46ed-98d5-6c0bc9b39b53",
      "bias": [
        -0.016,
        1.58
      ]
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
    	[ [ 0.236, -1.984 ], [ -1.876, -0.628 ], [ -0.012, 1.616 ] ],
    	[ [ 0.56, 0.024 ], [ 1.748, -1.756 ], [ 0.524, -1.5 ] ],
    	[ [ 0.436, -0.492 ], [ 0.972, -1.116 ], [ -1.956, 0.788 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 0.21999999999999997, -0.4039999999999999 ], [ -1.892, 0.9520000000000001 ], [ -0.028, 3.196 ] ],
    	[ [ 0.544, 1.604 ], [ 1.732, -0.17599999999999993 ], [ 0.508, 0.08000000000000007 ] ],
    	[ [ 0.42, 1.088 ], [ 0.956, 0.46399999999999997 ], [ -1.972, 2.3680000000000003 ] ]
    ]
    --------------------
    Derivative: 
    [
    	[ [ 1.0, 1.0 ], [ 1.0, 1.0 ], [ 1.0, 1.0 ] ],
    	[ [ 1.0, 1.0 ], [ 1.0, 1.0 ], [ 1.0, 1.0 ] ],
    	[ [ 1.0, 1.0 ], [ 1.0, 1.0 ], [ 1.0, 1.0 ] ]
    ]
```



[GPU Log](etc/cuda.log)

### Input Learning
In this test, we use a network to learn this target input, given it's pre-evaluated output:

Code from [LearningTester.java:127](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L127) executed in 0.01 seconds: 
```java
    return Arrays.stream(input_target).map(x -> x.prettyPrint()).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [
    	[ [ -1.188, -1.932 ], [ 0.424, -1.868 ], [ 1.692, -0.824 ], [ -1.04, -0.692 ], [ -0.86, 0.68 ], [ -1.604, 1.5 ], [ -1.036, 0.224 ], [ -1.32, 1.624 ], ... ],
    	[ [ 0.552, -0.064 ], [ 0.364, -1.628 ], [ -1.288, -1.064 ], [ 1.424, 0.916 ], [ -1.216, 1.528 ], [ 1.416, -0.3 ], [ 1.932, -1.156 ], [ 1.876, 1.948 ], ... ],
    	[ [ 0.232, -0.716 ], [ 0.824, 0.0 ], [ -1.748, 0.708 ], [ 1.42, 0.152 ], [ -0.084, -0.328 ], [ -0.848, 0.188 ], [ -1.556, -1.924 ], [ -0.74, 1.5 ], ... ],
    	[ [ 0.964, -0.78 ], [ 1.236, 1.048 ], [ -1.956, 0.884 ], [ 0.336, -1.524 ], [ 1.048, 1.528 ], [ -1.064, -1.444 ], [ 0.848, -1.592 ], [ -1.496, 1.668 ], ... ],
    	[ [ -0.732, -0.968 ], [ -0.288, -0.652 ], [ 1.992, 0.232 ], [ 1.056, 0.544 ], [ 1.752, -0.612 ], [ 0.92, -0.196 ], [ -0.256, 0.836 ], [ -0.8, -1.404 ], ... ],
    	[ [ 0.652, -1.576 ], [ 1.664, -1.68 ], [ 0.22, -1.492 ], [ -0.18, 1.216 ], [ 1.712, -1.384 ], [ 0.504, -1.984 ], [ 0.232, 1.536 ], [ -1.24, -0.032 ], ... ],
    	[ [ 1.872, 1.488 ], [ 1.276, 0.536 ], [ -0.568, -1.256 ], [ -1.14, 0.64 ], [ -1.248, 0.468 ], [ -0.58, 0.604 ], [ -1.74, -1.284 ], [ 1.376, 0.936 ], ... ],
    	[ [ 1.228, 0.008 ], [ -1.52, -0.92 ], [ 0.428, 1.892 ], [ -1.376, -0.288 ], [ 1.488, -1.632 ], [ -0.824, -1.736 ], [ 1.712, 1.156 ], [ -1.668, -1.94 ], ... ],
    	...
    ]
```



First, we use a conjugate gradient descent method, which converges the fastest for purely linear functions.

Code from [LearningTester.java:225](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L225) executed in 0.07 seconds: 
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
    F(0.0) = LineSearchPoint{point=PointSample{avg=2.6723520288000135}, derivative=-5.3447040576E-4}
    New Minimum: 2.6723520288000135 > 2.6723520287999643
    F(1.0E-10) = LineSearchPoint{point=PointSample{avg=2.6723520287999643}, derivative=-5.344704057599947E-4}, delta = -4.929390229335695E-14
    New Minimum: 2.6723520287999643 > 2.6723520287996223
    F(7.000000000000001E-10) = LineSearchPoint{point=PointSample{avg=2.6723520287996223}, derivative=-5.344704057599626E-4}, delta = -3.9124259387790516E-13
    New Minimum: 2.6723520287996223 > 2.6723520287973557
    F(4.900000000000001E-9) = LineSearchPoint{point=PointSample{avg=2.6723520287973557}, derivative=-5.344704057597382E-4}, delta = -2.6578739209526248E-12
    New Minimum: 2.6723520287973557 > 2.672352028781705
    F(3.430000000000001E-8) = LineSearchPoint{point=PointSample{avg=2.672352028781705}, derivative=-5.344704057581669E-4}, delta = -1.830846585448853E-11
    New Minimum: 2.672352028781705 > 2.6723520286716798
    F(2.4010000000000004
```
...[skipping 2407 bytes](etc/100.txt)...
```
    nt=PointSample{avg=4.696187576393836E-33}, derivative=5.0777990392945004E-36}, delta = -2.9844873268205796E-27
    4.696187576393836E-33 <= 2.984492023008156E-27
    Converged to right
    Iteration 2 complete. Error: 4.696187576393836E-33 Total: 239572083950167.9400; Orientation: 0.0006; Line Search: 0.0047
    Zero gradient: 9.691426702394065E-19
    F(0.0) = LineSearchPoint{point=PointSample{avg=4.696187576393836E-33}, derivative=-9.392375152787671E-37}
    New Minimum: 4.696187576393836E-33 > 4.686326815078573E-33
    F(9999.999999999665) = LineSearchPoint{point=PointSample{avg=4.686326815078573E-33}, derivative=9.372653630157146E-37}, delta = -9.86076131526293E-36
    4.686326815078573E-33 <= 4.696187576393836E-33
    New Minimum: 4.686326815078573E-33 > 0.0
    F(5005.254860746021) = LineSearchPoint{point=PointSample{avg=0.0}, derivative=0.0}, delta = -4.696187576393836E-33
    Right bracket at 5005.254860746021
    Converged to right
    Iteration 3 complete. Error: 0.0 Total: 239572095326505.9000; Orientation: 0.0006; Line Search: 0.0095
    
```

Returns: 

```
    0.0
```



Training Converged

Next, we run the same optimization using L-BFGS, which is nearly ideal for purely second-order or quadratic functions.

Code from [LearningTester.java:249](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L249) executed in 0.04 seconds: 
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
    th(0)=2.6723520288000135;dx=-5.3447040576E-4
    New Minimum: 2.6723520288000135 > 2.6712006712566145
    WOLFE (weak): th(2.154434690031884)=2.6712006712566145; dx=-5.343552576017036E-4 delta=0.001151357543399012
    New Minimum: 2.6712006712566145 > 2.6700495617924176
    WOLFE (weak): th(4.308869380063768)=2.6700495617924176; dx=-5.342401094434071E-4 delta=0.002302467007595954
    New Minimum: 2.6700495617924176 > 2.6654476047275972
    WOLFE (weak): th(12.926608140191302)=2.6654476047275972; dx=-5.337795168102212E-4 delta=0.006904424072416315
    New Minimum: 2.6654476047275972 > 2.6447879176146056
    WOLFE (weak): th(51.70643256076521)=2.6447879176146056; dx=-5.317068499608846E-4 delta=0.027564111185407913
    New Minimum: 2.6447879176146056 > 2.5359604089886703
    WOLFE (weak): th(258.53216280382605)=2.5359604089886703; dx=-5.206526267644229E-4 delta=0.1363916198113433
    New Minimum: 2.5359604089886703 > 1.9075874142650608
    END: th(1551.1929768229563)=1.
```
...[skipping 170 bytes](etc/101.txt)...
```
    e Search: 0.0189
    LBFGS Accumulation History: 1 points
    th(0)=1.9075874142650608;dx=-3.815174828530131E-4
    New Minimum: 1.9075874142650608 > 0.8456279731105125
    END: th(3341.943960201201)=0.8456279731105125; dx=-2.5401647809983384E-4 delta=1.0619594411545483
    Iteration 2 complete. Error: 0.8456279731105125 Total: 239572128636204.8800; Orientation: 0.0010; Line Search: 0.0044
    LBFGS Accumulation History: 1 points
    th(0)=0.8456279731105125;dx=-1.691255946221018E-4
    New Minimum: 0.8456279731105125 > 0.0662972330918639
    END: th(7200.000000000001)=0.0662972330918639; dx=-4.735516649418848E-5 delta=0.7793307400186487
    Iteration 3 complete. Error: 0.0662972330918639 Total: 239572136159348.8800; Orientation: 0.0010; Line Search: 0.0048
    LBFGS Accumulation History: 1 points
    th(0)=0.0662972330918639;dx=-1.3259446618372769E-5
    MAX ALPHA: th(0)=0.0662972330918639;th'(0)=-1.3259446618372769E-5;
    Iteration 4 failed, aborting. Error: 0.0662972330918639 Total: 239572142475609.8800; Orientation: 0.0010; Line Search: 0.0041
    
```

Returns: 

```
    0.0662972330918639
```



This training run resulted in the following regressed input:

Code from [LearningTester.java:154](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L154) executed in 0.01 seconds: 
```java
    return Arrays.stream(input_lbgfs).map(x -> x.prettyPrint()).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [
    	[ [ -1.0884553448375796, -1.314571126207772 ], [ 0.4120294402019874, -1.3091638662717282 ], [ 1.1923866357987376, -0.8744023570442635 ], [ -0.7199450327689267, -0.5067713378623315 ], [ -0.8549597642955736, 0.8816094281770542 ], [ -1.1617193169365878, 1.0255878143208694 ], [ -0.9156643725568209, 0.05200195658645074 ], [ -1.191473989537128, 1.1199764295573646 ], ... ],
    	[ [ 0.42347398953712806, -0.24922866213766848 ], [ 0.31359764295573644, -1.3167654452516726 ], [ -1.0385083326308955, -0.7477252095472464 ], [ 1.1146555336408326, 0.8744180554384825 ], [ -1.3067242426796744, 1.3345809548426386 ], [ 1.2458920449756106, -0.561462227167117 ], [ 1.6472266826999111, -0.8863473898131902 ], [ 1.6674602477293594, 1.6418056809560992 ], ... ],
    	[ [ 0.49031207985185055, -0.6844985268473353 ], [ 0.5965593638377608, -0.1493169827436307 ], [ -1.4191246202861805, 0.3721942961925945 ], [ 1.1440470951826571, 0.4147222860932236 ], [ -0.19929539173875277, 0.007175674344352401 ], [ -0.858710500871906, 0.3127458336845521 ], [ -1.
```
...[skipping 935 bytes](etc/102.txt)...
```
    12462611629015186, -1.3792247261134603 ], [ -0.05084396007407471, 0.9551678022959365 ], [ 1.2640490517691079, -1.2655544609459806 ], [ 0.16441411941427453, -1.8750049028917801 ], [ 0.2647615320787713, 1.5372600589261067 ], [ -1.1688066706749778, -0.09752306415754258 ], ... ],
    	[ [ 1.2816623931190634, 1.2196074487392967 ], [ 1.2564690866453478, 0.2209852684733531 ], [ -0.5257880259754293, -0.9920176549806696 ], [ -1.2653758631476055, 0.3312855631038859 ], [ -1.2782414142265581, 0.26387045397073267 ], [ -0.40422177980813095, 0.7526869532805773 ], [ -1.6782571126207773, -0.8517997883454402 ], [ 1.3545789982561878, 0.8918979375862692 ], ... ],
    	[ [ 1.1643670242316173, 0.26757213877795727 ], [ -1.4059646671873538, -0.8475466117488712 ], [ 0.24214130839927828, 1.8907399410738932 ], [ -0.9475799651237599, -0.1689244314829275 ], [ 1.3947556394681124, -1.2842237363945816 ], [ -0.9544160988520318, -1.757421001743812 ], [ 1.129222746675703, 1.2864160988520317 ], [ -1.4865515146406512, -1.417075545665766 ], ... ],
    	...
    ]
```



Code from [LearningTester.java:95](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L95) executed in 0.00 seconds: 
```java
    return TestUtil.compare(runs);
```

Returns: 

![Result](etc/test.65.png)



Code from [LearningTester.java:98](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L98) executed in 0.00 seconds: 
```java
    return TestUtil.compareTime(runs);
```

Returns: 

![Result](etc/test.66.png)



### Model Learning
In this test, attempt to train a network to emulate a randomized network given an example input/output. The target state is:

Code from [LearningTester.java:176](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L176) executed in 0.00 seconds: 
```java
    return network_target.state().stream().map(Arrays::toString).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [1.58, -0.016]
```



First, we use a conjugate gradient descent method, which converges the fastest for purely linear functions.

Code from [LearningTester.java:225](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L225) executed in 0.00 seconds: 
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

Returns: 

```
    0.0
```



This training run resulted in the following configuration:

Code from [LearningTester.java:189](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L189) executed in 0.00 seconds: 
```java
    return network_gd.state().stream().map(Arrays::toString).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [2.048, 0.78, 2.108, 0.20799999999999996, 1.304, 0.9840000000000001, 2.464, -0.18399999999999994, 2.56, 0.768, 2.144, -0.248, -0.3039999999999998, 3.492, 3.468, 0.0, 0.4520000000000002, 1.62, 2.072, 1.352, 0.17200000000000015, 0.40000000000000013, 3.176, 1.4000000000000001, 0.21599999999999997, 2.3600000000000003, 0.21199999999999997, 2.784, 0.552, 3.12, 0.4520000000000002, 1.472, 2.752, 3.436, 2.524, 2.052, 0.43600000000000017, 0.32400000000000007, 3.104, 3.188, 1.292, 1.488, 2.88, 2.5, 0.31200000000000006, 1.96, 2.308, -0.08799999999999986, 3.4080000000000004, 3.092, 1.1680000000000001, 0.9960000000000001, 0.8520000000000001, 3.108, 1.028, 0.3720000000000001, 2.3520000000000003, 1.752, 2.324, 2.3440000000000003, 0.932, 2.9240000000000004, 0.9680000000000001, 1.1400000000000001, 0.796, 1.844, 2.236, 3.024, -0.3799999999999999, 1.596, 1.456, 0.8840000000000001, 2.8280000000000003, 0.896, 2.588, 2.98, 3.488, -0.05599999999999983, 1.804, 0.8760000000000001, 3.224, 1.8760000000000001, 2.364, 0.924, 2.204, 3.548,
```
...[skipping 216457 bytes](etc/103.txt)...
```
    -0.6880000000000001, 0.944, 0.284, -1.488, 1.012, -0.776, -0.744, -1.108, -1.208, 0.996, 0.196, 1.652, 1.46, -1.404, 1.076, 1.668, 0.044, 0.504, -0.036000000000000004, 1.744, 1.132, -1.416, -0.128, 0.632, 1.952, 0.828, 1.012, -0.552, -1.556, -1.392, -1.28, 0.24, 0.324, 1.292, -1.868, -1.448, -1.756, 1.856, -0.74, -1.556, -1.6320000000000001, 1.688, -0.28800000000000003, 1.296, 0.62, -0.07200000000000001, 1.844, 1.228, -1.692, 0.12000000000000001, 1.84, 0.9319999999999999, 1.68, -1.224, -0.876, -0.524, 1.632, 1.252, 0.188, -0.992, 1.088, -1.348, 0.128, -0.892, 0.12400000000000001, -0.264, 0.58, -1.3, 0.15599999999999997, -1.776, 0.732, -1.696, 1.18, -1.952, -1.896, 1.768, -0.268, -1.74, -1.172, 1.02, 1.676, 1.48, -0.972, -0.264, 0.14800000000000002, 0.284, 0.348, -0.556, 0.46399999999999997, -0.28800000000000003, 1.52, 1.692, 0.45999999999999996, 0.39199999999999996, -1.764, -0.84, 0.62, 0.236, 1.48, 0.608, -0.936, -0.976, 0.952, 1.752, -0.896, -0.328, 0.996, -1.788, 1.352, -0.14400000000000002]
    [1.58, -0.016]
```



Next, we run the same optimization using L-BFGS, which is nearly ideal for purely second-order or quadratic functions.

Code from [LearningTester.java:249](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L249) executed in 0.14 seconds: 
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
    th(0)=2.547215999999466;dx=-5.09443199999999
    Armijo: th(2.154434690031884)=3.394724315592003; dx=5.881189026808488 delta=-0.8475083155925369
    New Minimum: 2.547215999999466 > 0.01518782219569062
    WOLF (strong): th(1.077217345015942)=0.01518782219569062; dx=0.3933785134042482 delta=2.532028177803775
    END: th(0.3590724483386473)=1.0463660883761574; dx=-3.265161828865243 delta=1.5008499116233085
    Iteration 1 complete. Error: 0.01518782219569062 Total: 239572307822066.7000; Orientation: 0.0001; Line Search: 0.0058
    LBFGS Accumulation History: 1 points
    th(0)=1.0463660883761574;dx=-2.092732176752706
    New Minimum: 1.0463660883761574 > 0.053634426503121904
    END: th(0.7735981389354633)=0.053634426503121904; dx=-0.47379845952645394 delta=0.9927316618730355
    Iteration 2 complete. Error: 0.053634426503121904 Total: 239572311940859.7000; Orientation: 0.0001; Line Search: 0.0028
    LBFGS Accumulation History: 1 points
    th(0)=0.053634426503121904
```
...[skipping 11876 bytes](etc/104.txt)...
```
    827779764965016E-35
    Iteration 25 complete. Error: 1.1269097587095814E-35 Total: 239572428150772.5600; Orientation: 0.0000; Line Search: 0.0029
    LBFGS Accumulation History: 1 points
    th(0)=1.4013547757847624E-35;dx=-1.6927118651840947E-36
    New Minimum: 1.4013547757847624E-35 > 1.1895024819021665E-35
    END: th(3.643262599123092)=1.1895024819021665E-35; dx=-1.2422248141297673E-36 delta=2.1185229388259586E-36
    Iteration 26 complete. Error: 1.1895024819021665E-35 Total: 239572431489009.5600; Orientation: 0.0000; Line Search: 0.0022
    LBFGS Accumulation History: 1 points
    th(0)=1.1895024819021665E-35;dx=-9.116273835960318E-37
    New Minimum: 1.1895024819021665E-35 > 1.1269097587095814E-35
    WOLF (strong): th(7.8491713284465146)=1.1269097587095814E-35; dx=8.520005925178031E-37 delta=6.259272319258509E-37
    New Minimum: 1.1269097587095814E-35 > 0.0
    END: th(3.9245856642232573)=0.0; dx=0.0 delta=1.1895024819021665E-35
    Iteration 27 complete. Error: 0.0 Total: 239572436121332.5600; Orientation: 0.0000; Line Search: 0.0030
    
```

Returns: 

```
    0.0
```



Training Converged

Code from [LearningTester.java:95](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L95) executed in 0.00 seconds: 
```java
    return TestUtil.compare(runs);
```

Returns: 

![Result](etc/test.67.png)



Code from [LearningTester.java:98](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L98) executed in 0.00 seconds: 
```java
    return TestUtil.compareTime(runs);
```

Returns: 

![Result](etc/test.68.png)



### Performance
Now we execute larger-scale runs to benchmark performance:

Code from [PerformanceTester.java:66](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/PerformanceTester.java#L66) executed in 0.48 seconds: 
```java
    test(component, inputPrototype);
```
Logging: 
```
    100 batches
    Input Dimensions:
    	[100, 100, 2]
    Performance:
    	Evaluation performance: 0.013498s +- 0.002050s [0.009542s - 0.015226s]
    	Learning performance: 0.065261s +- 0.021712s [0.048744s - 0.104499s]
    
```

