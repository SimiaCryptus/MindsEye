# ImgConcatLayer
## Float
### Batch Execution
Code from [BatchingTester.java:66](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/BatchingTester.java#L66) executed in 0.01 seconds: 
```java
    return test(reference, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (160#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (160#)}
```



Code from [SingleDerivativeTester.java:77](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/SingleDerivativeTester.java#L77) executed in 0.02 seconds: 
```java
    return test(component, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ -1.004 ], [ -0.856 ] ],
    	[ [ -0.132 ], [ -1.444 ] ]
    ],
    [
    	[ [ -1.568 ], [ 1.492 ] ],
    	[ [ -1.144 ], [ -0.424 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.196412849518594, negative=4, min=-1.444, max=-1.444, mean=-0.859, count=4.0, positive=0, stdDev=0.47217263791964903, zeros=0},
    {meanExponent=0.01372669063370191, negative=3, min=-0.424, max=-0.424, mean=-0.4109999999999999, count=4.0, positive=1, stdDev=1.172339114761595, zeros=0}
    Output: [
    	[ [ -1.004, -1.568 ], [ -0.856, 1.492 ] ],
    	[ [ -0.132, -1.144 ], [ -1.444, -0.424 ] ]
    ]
    Outputs Statistics: {meanExponent=-0.09134307944244605, negative=7, min=-0.424, max=-0.424, mean=-0.635, count=8.0, positive=1, stdDev=0.9213245899247453, zeros=0}
    Feedback for input 0
    Inputs Values: [
    	[ [ -1.004 ], [ -0.856 ] ],
    	[ [ -0.132 ], [ -1.444 ] ]
    ]
    Value Statistics: {meanExponent=-0.196412849518594, negative=4, min=-1.444, max=-1.444, mean=-0.859, count=4.0, positive=0, stdDev=0.47217263791964903, zeros=0}
    Implemented Feedback: [ [ 1.0, 0.0, 0.0, 0.0, 0
```
...[skipping 1936 bytes](etc/120.txt)...
```
    998899, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9999999999998899, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9999999999998899 ] ]
    Measured Statistics: {meanExponent=-4.7830642341045674E-14, negative=0, min=0.9999999999998899, max=0.9999999999998899, mean=0.12499999999998623, count=32.0, positive=4, stdDev=0.3307189138830374, zeros=28}
    Feedback Error: [ [ 0.0, 0.0, 0.0, 0.0, -1.1013412404281553E-13, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, -1.1013412404281553E-13, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.1013412404281553E-13, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.1013412404281553E-13 ] ]
    Error Statistics: {meanExponent=-12.958078098036825, negative=4, min=-1.1013412404281553E-13, max=-1.1013412404281553E-13, mean=-1.3766765505351941E-14, count=32.0, positive=0, stdDev=3.6423437884903677E-14, zeros=28}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 1.3767e-14 +- 3.6423e-14 [0.0000e+00 - 1.1013e-13] (64#)
    relativeTol: 5.5067e-14 +- 0.0000e+00 [5.5067e-14 - 5.5067e-14] (8#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=1.3767e-14 +- 3.6423e-14 [0.0000e+00 - 1.1013e-13] (64#), relativeTol=5.5067e-14 +- 0.0000e+00 [5.5067e-14 - 5.5067e-14] (8#)}
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
      "class": "com.simiacryptus.mindseye.layers.cudnn.ImgConcatLayer",
      "id": "c059ece3-937c-4627-9a02-3610217de6db",
      "isFrozen": false,
      "name": "ImgConcatLayer/c059ece3-937c-4627-9a02-3610217de6db",
      "maxBands": -1
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
    	[ [ 1.8 ], [ -1.008 ] ],
    	[ [ 0.06 ], [ -1.26 ] ]
    ],
    [
    	[ [ 1.016 ], [ -1.62 ] ],
    	[ [ 0.32 ], [ -1.904 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 1.8, 1.016 ], [ -1.008, -1.62 ] ],
    	[ [ 0.06, 0.32 ], [ -1.26, -1.904 ] ]
    ]
    --------------------
    Derivative: 
    [
    	[ [ 1.0 ], [ 1.0 ] ],
    	[ [ 1.0 ], [ 1.0 ] ]
    ],
    [
    	[ [ 1.0 ], [ 1.0 ] ],
    	[ [ 1.0 ], [ 1.0 ] ]
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
    	[ [ -1.656 ], [ -0.92 ], [ 0.168 ], [ -0.272 ], [ 0.996 ], [ -0.3 ], [ -1.156 ], [ 0.016 ], ... ],
    	[ [ 1.216 ], [ 0.276 ], [ 1.304 ], [ 1.636 ], [ 1.66 ], [ 0.268 ], [ -1.128 ], [ 1.076 ], ... ],
    	[ [ -1.656 ], [ -1.924 ], [ -1.12 ], [ 1.52 ], [ -1.82 ], [ 0.032 ], [ -1.676 ], [ -1.332 ], ... ],
    	[ [ -0.912 ], [ -0.948 ], [ -1.972 ], [ -1.316 ], [ -0.396 ], [ 1.016 ], [ -0.676 ], [ -0.612 ], ... ],
    	[ [ -1.932 ], [ -1.88 ], [ 0.172 ], [ 0.92 ], [ 1.956 ], [ -1.832 ], [ 0.864 ], [ 0.608 ], ... ],
    	[ [ 1.332 ], [ 1.032 ], [ 1.108 ], [ -1.032 ], [ -1.556 ], [ 1.184 ], [ -0.028 ], [ -0.832 ], ... ],
    	[ [ 1.82 ], [ 0.868 ], [ -0.876 ], [ 1.392 ], [ 0.32 ], [ 0.296 ], [ 1.82 ], [ 0.672 ], ... ],
    	[ [ 1.092 ], [ 1.908 ], [ 1.316 ], [ -1.416 ], [ 0.848 ], [ 0.724 ], [ -0.992 ], [ -1.496 ], ... ],
    	...
    ]
    [
    	[ [ 1.404 ], [ -1.076 ], [ -0.664 ], [ 1.372 ], [ 1.972 ], [ 0.92 ], [ -0.868 ], [ 0.668 ], ... ],
    	[ [ -0.4 ], [ -0.008 ], [ 0.1 ], [ -1.732 ], [ 1.296 ], [ -1.852 ], [ -0.984 ], [ 1.82 ], ... ],
    	[ [ 0.168 ], [ -0.68 ], [ -1.424 ], [ -0.604 ], [ -0.052 ], [ 1.884 ], [ 1.068 ], [ 0.784 ], ... ],
    	[ [ -1.384 ], [ 0.276 ], [ -1.888 ], [ -0.112 ], [ 1.056 ], [ -0.516 ], [ 0.128 ], [ 0.076 ], ... ],
    	[ [ 1.16 ], [ -0.652 ], [ -1.264 ], [ -1.552 ], [ -1.976 ], [ -0.516 ], [ -1.2 ], [ 0.728 ], ... ],
    	[ [ 1.204 ], [ -0.464 ], [ 1.1 ], [ 0.912 ], [ -0.128 ], [ 1.812 ], [ 1.588 ], [ -1.916 ], ... ],
    	[ [ -0.204 ], [ 0.916 ], [ -0.948 ], [ 0.208 ], [ -1.996 ], [ 0.32 ], [ -1.968 ], [ -0.004 ], ... ],
    	[ [ 0.496 ], [ -1.288 ], [ 1.78 ], [ -1.084 ], [ -1.248 ], [ -0.552 ], [ -1.012 ], [ -0.668 ], ... ],
    	...
    ]
```



First, we use a conjugate gradient descent method, which converges the fastest for purely linear functions.

Code from [LearningTester.java:225](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L225) executed in 0.11 seconds: 
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
    F(0.0) = LineSearchPoint{point=PointSample{avg=2.670320089600003}, derivative=-7.9368874304E-4}
    New Minimum: 2.670320089600003 > 2.6703200895999255
    F(1.0E-10) = LineSearchPoint{point=PointSample{avg=2.6703200895999255}, derivative=-7.936887430399842E-4}, delta = -7.771561172376096E-14
    New Minimum: 2.6703200895999255 > 2.670320089599458
    F(7.000000000000001E-10) = LineSearchPoint{point=PointSample{avg=2.670320089599458}, derivative=-7.936887430398889E-4}, delta = -5.453415496958769E-13
    New Minimum: 2.670320089599458 > 2.6703200895961072
    F(4.900000000000001E-9) = LineSearchPoint{point=PointSample{avg=2.6703200895961072}, derivative=-7.936887430392222E-4}, delta = -3.895994638014599E-12
    New Minimum: 2.6703200895961072 > 2.6703200895727863
    F(3.430000000000001E-8) = LineSearchPoint{point=PointSample{avg=2.6703200895727863}, derivative=-7.936887430345554E-4}, delta = -2.7216895404080788E-11
    New Minimum: 2.6703200895727863 > 2.6703200894094308
    F(2.4010000000000004E-7
```
...[skipping 5019 bytes](etc/121.txt)...
```
    ve=-1.751006206004587E-36}, delta = 0.0
    F(5971.648142696749) = LineSearchPoint{point=PointSample{avg=0.6860982320000002}, derivative=1.5519403581427163E-36}, delta = 0.0
    0.6860982320000002 <= 0.6860982320000002
    F(3464.9244341836115) = LineSearchPoint{point=PointSample{avg=0.6860982320000002}, derivative=-2.575540990139901E-37}, delta = 0.0
    Left bracket at 3464.9244341836115
    F(3821.718545196021) = LineSearchPoint{point=PointSample{avg=0.6860982320000002}, derivative=-1.3748919993306339E-37}, delta = 0.0
    Left bracket at 3821.718545196021
    F(3996.68417123222) = LineSearchPoint{point=PointSample{avg=0.6860982320000002}, derivative=-9.197363011852285E-38}, delta = 0.0
    Left bracket at 3996.68417123222
    F(4107.179371775535) = LineSearchPoint{point=PointSample{avg=0.6860982320000002}, derivative=4.51443008747809E-37}, delta = 0.0
    Right bracket at 4107.179371775535
    Converged to left
    Iteration 3 failed, aborting. Error: 0.6860982320000002 Total: 239575985809306.0300; Orientation: 0.0004; Line Search: 0.0267
    
```

Returns: 

```
    0.6860982320000002
```



This training run resulted in the following regressed input:

Code from [LearningTester.java:144](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L144) executed in 0.01 seconds: 
```java
    return Arrays.stream(input_gd).map(x -> x.prettyPrint()).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [
    	[ [ -0.12600000000000006 ], [ -0.9980000000000001 ], [ -0.248 ], [ 0.55 ], [ 1.484 ], [ 0.31000000000000005 ], [ -1.012 ], [ 0.342 ], ... ],
    	[ [ 0.408 ], [ 0.134 ], [ 0.702 ], [ -0.04799999999999999 ], [ 1.478 ], [ -0.792 ], [ -1.0559999999999998 ], [ 1.4480000000000002 ], ... ],
    	[ [ -0.7439999999999999 ], [ -1.302 ], [ -1.272 ], [ 0.458 ], [ -0.936 ], [ 0.958 ], [ -0.3039999999999999 ], [ -0.274 ], ... ],
    	[ [ -1.148 ], [ -0.33599999999999997 ], [ -1.93 ], [ -0.714 ], [ 0.33 ], [ 0.25 ], [ -0.274 ], [ -0.268 ], ... ],
    	[ [ -0.386 ], [ -1.266 ], [ -0.5459999999999999 ], [ -0.31600000000000006 ], [ -0.009999999999999983 ], [ -1.1740000000000002 ], [ -0.1679999999999999 ], [ 0.668 ], ... ],
    	[ [ 1.268 ], [ 0.28400000000000003 ], [ 1.104 ], [ -0.059999999999999956 ], [ -0.8420000000000001 ], [ 1.498 ], [ 0.78 ], [ -1.3739999999999999 ], ... ],
    	[ [ 0.808 ], [ 0.892 ], [ -0.912 ], [ 0.7999999999999999 ], [ -0.838 ], [ 0.308 ], [ -0.07399999999999991 ], [ 0.334 ], ... ],
    	[ [ 0.794 ], [ 0.30999999999999994 ], [ 1.548 ], [ -1.25 ], [ -0.19999999999999998 ], [ 0.08599999999999991 ], [ -1.002 ], [ -1.082 ], ... ],
    	...
    ]
    [
    	[ [ 0.472 ], [ -1.764 ], [ -0.208 ], [ 0.632 ], [ -0.784 ], [ 0.372 ], [ -1.544 ], [ 1.324 ], ... ],
    	[ [ 1.816 ], [ -1.952 ], [ -0.576 ], [ -0.204 ], [ 1.164 ], [ -0.072 ], [ 1.112 ], [ -0.9 ], ... ],
    	[ [ -1.628 ], [ 0.192 ], [ -0.16 ], [ -1.98 ], [ 0.008 ], [ -0.48 ], [ 1.532 ], [ 0.496 ], ... ],
    	[ [ 0.4 ], [ -0.032 ], [ 0.74 ], [ 1.556 ], [ 0.36 ], [ -0.86 ], [ -0.016 ], [ -1.724 ], ... ],
    	[ [ -1.148 ], [ 0.696 ], [ 1.244 ], [ 1.676 ], [ -0.82 ], [ -1.968 ], [ -1.164 ], [ -0.416 ], ... ],
    	[ [ 0.444 ], [ -0.84 ], [ -1.472 ], [ -0.584 ], [ -1.128 ], [ -1.924 ], [ 1.144 ], [ -0.792 ], ... ],
    	[ [ -0.52 ], [ -0.948 ], [ 0.452 ], [ -2.0 ], [ 1.432 ], [ 0.636 ], [ -1.784 ], [ -1.136 ], ... ],
    	[ [ 0.62 ], [ -0.208 ], [ -0.288 ], [ -1.408 ], [ 0.868 ], [ 0.812 ], [ -0.72 ], [ -1.464 ], ... ],
    	...
    ]
```



Next, we run the same optimization using L-BFGS, which is nearly ideal for purely second-order or quadratic functions.

Code from [LearningTester.java:249](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L249) executed in 0.09 seconds: 
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
    th(0)=2.670320089600003;dx=-7.9368874304E-4
    New Minimum: 2.670320089600003 > 2.668610507436589
    WOLFE (weak): th(2.154434690031884)=2.668610507436589; dx=-7.933467529277814E-4 delta=0.001709582163414325
    New Minimum: 2.668610507436589 > 2.6669016620685526
    WOLFE (weak): th(4.308869380063768)=2.6669016620685526; dx=-7.930047628155628E-4 delta=0.0034184275314506074
    New Minimum: 2.6669016620685526 > 2.660073648549928
    WOLFE (weak): th(12.926608140191302)=2.660073648549928; dx=-7.916368023666882E-4 delta=0.010246441050075106
    New Minimum: 2.660073648549928 > 2.629493473197852
    WOLFE (weak): th(51.70643256076521)=2.629493473197852; dx=-7.854809803467529E-4 delta=0.04082661640215113
    New Minimum: 2.629493473197852 > 2.47043094887098
    WOLFE (weak): th(258.53216280382605)=2.47043094887098; dx=-7.526499295737638E-4 delta=0.1998891407290233
    New Minimum: 2.47043094887098 > 1.6301330432908252
    END: th(1551.1929768229563)=1.6301330432908252;
```
...[skipping 2410 bytes](etc/122.txt)...
```
    4500.000000000002)=0.6860983429245198; dx=-4.43698068669374E-10 delta=1.0981527197140295E-5
    Iteration 6 complete. Error: 0.6860983429245198 Total: 239576066303732.9400; Orientation: 0.0005; Line Search: 0.0072
    LBFGS Accumulation History: 1 points
    th(0)=0.6860983429245198;dx=-4.436980686693736E-11
    New Minimum: 0.6860983429245198 > 0.686098329802646
    WOLF (strong): th(9694.956105143481)=0.686098329802646; dx=4.166285912679304E-11 delta=1.3121873809751605E-8
    New Minimum: 0.686098329802646 > 0.6860982321032155
    END: th(4847.478052571741)=0.6860982321032155; dx=-1.353473870072106E-12 delta=1.1082130424444614E-7
    Iteration 7 complete. Error: 0.6860982321032155 Total: 239576076059967.9400; Orientation: 0.0005; Line Search: 0.0075
    LBFGS Accumulation History: 1 points
    th(0)=0.6860982321032155;dx=-4.128689409132897E-14
    MAX ALPHA: th(0)=0.6860982321032155;th'(0)=-4.128689409132897E-14;
    Iteration 8 failed, aborting. Error: 0.6860982321032155 Total: 239576083561453.9400; Orientation: 0.0005; Line Search: 0.0053
    
```

Returns: 

```
    0.6860982321032155
```



This training run resulted in the following regressed input:

Code from [LearningTester.java:154](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L154) executed in 0.01 seconds: 
```java
    return Arrays.stream(input_lbgfs).map(x -> x.prettyPrint()).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [
    	[ [ -0.12600688064847804 ], [ -0.9979846952452093 ], [ -0.24800781826305054 ], [ 0.5500015434578348 ], [ 1.4839949801558272 ], [ 0.3100077461388527 ], [ -1.012006260380376 ], [ 0.34198980163841936 ], ... ],
    	[ [ 0.40800845295599186 ], [ 0.1340033321379422 ], [ 0.7019887053506115 ], [ -0.04801295350593981 ], [ 1.4779751171517304 ], [ -0.7919883158799431 ], [ -1.0560063469294136 ], [ 1.4479774395509022 ], ... ],
    	[ [ -0.7439956436984478 ], [ -1.3020016877062304 ], [ -1.272000173098075 ], [ 0.4579884745531783 ], [ -0.9359811034601547 ], [ 0.9580035052360171 ], [ -0.30398730614117253 ], [ -0.2739956581232873 ], ... ],
    	[ [ -1.147993970417057 ], [ -0.33599573024748536 ], [ -1.9299720013863817 ], [ -0.7139864839253167 ], [ 0.3299873205660122 ], [ 0.2500081500343608 ], [ -0.27400564011227446 ], [ -0.2680086549037459 ], ... ],
    	[ [ -0.38599842769248616 ], [ -1.2659906094294355 ], [ -0.5459821276237645 ], [ -0.31600585648486806 ], [ -0.01001005411318497 ], [ -1.1739962062671914 ], [ -0.1680091741979707 ], [ 0.66798
```
...[skipping 437 bytes](etc/123.txt)...
```
    [ 0.7939984565421654 ], [ 0.3099965524633413 ], [ 1.5479798052245928 ], [ -1.2499965524633412 ], [ -0.19998667144823123 ], [ 0.0859926000572971 ], [ -1.0019938982928591 ], [ -1.0820050342690122 ], ... ],
    	...
    ]
    [
    	[ [ 0.472 ], [ -1.764 ], [ -0.208 ], [ 0.632 ], [ -0.784 ], [ 0.372 ], [ -1.544 ], [ 1.324 ], ... ],
    	[ [ 1.816 ], [ -1.952 ], [ -0.576 ], [ -0.204 ], [ 1.164 ], [ -0.072 ], [ 1.112 ], [ -0.9 ], ... ],
    	[ [ -1.628 ], [ 0.192 ], [ -0.16 ], [ -1.98 ], [ 0.008 ], [ -0.48 ], [ 1.532 ], [ 0.496 ], ... ],
    	[ [ 0.4 ], [ -0.032 ], [ 0.74 ], [ 1.556 ], [ 0.36 ], [ -0.86 ], [ -0.016 ], [ -1.724 ], ... ],
    	[ [ -1.148 ], [ 0.696 ], [ 1.244 ], [ 1.676 ], [ -0.82 ], [ -1.968 ], [ -1.164 ], [ -0.416 ], ... ],
    	[ [ 0.444 ], [ -0.84 ], [ -1.472 ], [ -0.584 ], [ -1.128 ], [ -1.924 ], [ 1.144 ], [ -0.792 ], ... ],
    	[ [ -0.52 ], [ -0.948 ], [ 0.452 ], [ -2.0 ], [ 1.432 ], [ 0.636 ], [ -1.784 ], [ -1.136 ], ... ],
    	[ [ 0.62 ], [ -0.208 ], [ -0.288 ], [ -1.408 ], [ 0.868 ], [ 0.812 ], [ -0.72 ], [ -1.464 ], ... ],
    	...
    ]
```



Code from [LearningTester.java:95](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L95) executed in 0.04 seconds: 
```java
    return TestUtil.compare(runs);
```

Returns: 

![Result](etc/test.75.png)



Code from [LearningTester.java:98](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L98) executed in 0.00 seconds: 
```java
    return TestUtil.compareTime(runs);
```

Returns: 

![Result](etc/test.76.png)



### Performance
Now we execute larger-scale runs to benchmark performance:

Code from [PerformanceTester.java:66](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/PerformanceTester.java#L66) executed in 0.50 seconds: 
```java
    test(component, inputPrototype);
```
Logging: 
```
    100 batches
    Input Dimensions:
    	[100, 100, 1]
    	[100, 100, 1]
    Performance:
    	Evaluation performance: 0.013053s +- 0.000874s [0.011672s - 0.014349s]
    	Learning performance: 0.072906s +- 0.025905s [0.048058s - 0.110319s]
    
```

