# ImgConcatLayer
## BandLimitTest
### Batch Execution
Code from [BatchingTester.java:66](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/BatchingTester.java#L66) executed in 0.01 seconds: 
```java
    return test(reference, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (280#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (240#)}
```



Code from [SingleDerivativeTester.java:77](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/SingleDerivativeTester.java#L77) executed in 0.03 seconds: 
```java
    return test(component, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ -0.576, 1.416 ], [ -0.752, -1.052 ] ],
    	[ [ -1.1, -0.016 ], [ -0.276, -1.936 ] ]
    ],
    [
    	[ [ 1.64, -0.24 ], [ -1.564, -0.308 ] ],
    	[ [ -1.848, 1.508 ], [ -1.264, -0.492 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.27711919749524166, negative=7, min=-1.936, max=-1.936, mean=-0.5365, count=8.0, positive=1, stdDev=0.9179399490162743, zeros=0},
    {meanExponent=-0.060417744985606626, negative=6, min=-0.492, max=-0.492, mean=-0.32100000000000006, count=8.0, positive=2, stdDev=1.2243753509443092, zeros=0}
    Output: [
    	[ [ -0.576, 1.416, 1.64 ], [ -0.752, -1.052, -1.564 ] ],
    	[ [ -1.1, -0.016, -1.848 ], [ -0.276, -1.936, -1.264 ] ]
    ]
    Outputs Statistics: {meanExponent=-0.11995199519666266, negative=10, min=-1.264, max=-1.264, mean=-0.6106666666666667, count=12.0, positive=2, stdDev=1.1074406931700174, zeros=0}
    Feedback for input 0
    Inputs Values: [
    	[ [ -0.576, 1.416 ], [ -0.752, -1.052 ] ],
    	[ [ -1.1, -0.016 ], [ -0.276, -1.936 ] ]
    ]
    Value Statistics: {meanExponent=-0.27711919749524166, negative=7, min=-1.
```
...[skipping 3356 bytes](etc/111.txt)...
```
    0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ] ]
    Measured Statistics: {meanExponent=-4.7830642341045674E-14, negative=0, min=0.0, max=0.0, mean=0.04166666666666208, count=96.0, positive=4, stdDev=0.1998263134713413, zeros=92}
    Feedback Error: [ [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ] ]
    Error Statistics: {meanExponent=-12.958078098036825, negative=4, min=0.0, max=0.0, mean=-4.588921835117314E-15, count=96.0, positive=0, stdDev=2.2007695994873667E-14, zeros=92}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 6.3410e-15 +- 2.5591e-14 [0.0000e+00 - 1.1013e-13] (192#)
    relativeTol: 5.0728e-14 +- 1.4391e-14 [2.9976e-15 - 5.5067e-14] (12#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=6.3410e-15 +- 2.5591e-14 [0.0000e+00 - 1.1013e-13] (192#), relativeTol=5.0728e-14 +- 1.4391e-14 [2.9976e-15 - 5.5067e-14] (12#)}
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
      "id": "3900dacd-7091-4292-ad27-2538ec5c08f1",
      "isFrozen": false,
      "name": "ImgConcatLayer/3900dacd-7091-4292-ad27-2538ec5c08f1",
      "maxBands": 3
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
    	[ [ -1.316, -0.048 ], [ -1.72, 1.42 ] ],
    	[ [ -1.012, 0.312 ], [ -0.812, -1.22 ] ]
    ],
    [
    	[ [ 0.124, 1.012 ], [ -1.056, -0.156 ] ],
    	[ [ 1.772, -0.208 ], [ -0.692, 1.328 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ -1.316, -0.048, 0.124 ], [ -1.72, 1.42, -1.056 ] ],
    	[ [ -1.012, 0.312, 1.772 ], [ -0.812, -1.22, -0.692 ] ]
    ]
    --------------------
    Derivative: 
    [
    	[ [ 1.0, 1.0 ], [ 1.0, 1.0 ] ],
    	[ [ 1.0, 1.0 ], [ 1.0, 1.0 ] ]
    ],
    [
    	[ [ 1.0, 0.0 ], [ 1.0, 0.0 ] ],
    	[ [ 1.0, 0.0 ], [ 1.0, 0.0 ] ]
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
    	[ [ 1.664 ], [ 0.144 ], [ 1.036 ], [ -0.104 ], [ -0.556 ], [ -0.052 ], [ 0.712 ], [ -1.148 ], ... ],
    	[ [ -1.056 ], [ 1.092 ], [ -1.104 ], [ 0.972 ], [ 1.468 ], [ 1.104 ], [ 0.156 ], [ -0.712 ], ... ],
    	[ [ 0.08 ], [ 0.252 ], [ -0.336 ], [ 1.484 ], [ -0.512 ], [ -0.564 ], [ 0.804 ], [ 1.228 ], ... ],
    	[ [ 1.38 ], [ -1.132 ], [ 0.844 ], [ -1.468 ], [ -1.18 ], [ 0.748 ], [ -0.848 ], [ 1.868 ], ... ],
    	[ [ -0.96 ], [ 1.716 ], [ -1.94 ], [ 0.232 ], [ 0.32 ], [ 0.652 ], [ -0.888 ], [ 0.768 ], ... ],
    	[ [ -1.584 ], [ 0.328 ], [ -0.26 ], [ -1.82 ], [ 1.304 ], [ -1.308 ], [ 1.532 ], [ 0.66 ], ... ],
    	[ [ -0.34 ], [ -0.82 ], [ 0.248 ], [ 1.78 ], [ -0.6 ], [ 0.844 ], [ -1.36 ], [ -0.848 ], ... ],
    	[ [ 1.928 ], [ 1.988 ], [ -0.9 ], [ 0.924 ], [ -1.112 ], [ -1.552 ], [ 1.948 ], [ -1.9 ], ... ],
    	...
    ]
    [
    	[ [ 1.84 ], [ -0.94 ], [ -1.552 ], [ -1.124 ], [ 1.38 ], [ 1.432 ], [ -1.08 ], [ 1.76 ], ... ],
    	[ [ 1.116 ], [ -0.116 ], [ 0.68 ], [ -1.396 ], [ -0.2 ], [ 0.072 ], [ 0.328 ], [ 1.96 ], ... ],
    	[ [ -1.5 ], [ 1.4 ], [ 0.652 ], [ 0.54 ], [ 1.548 ], [ 0.572 ], [ -1.876 ], [ 1.16 ], ... ],
    	[ [ 1.544 ], [ 0.672 ], [ -1.436 ], [ 1.88 ], [ 0.904 ], [ -0.96 ], [ -1.716 ], [ -0.756 ], ... ],
    	[ [ -1.8 ], [ -1.06 ], [ 0.22 ], [ 1.124 ], [ 0.42 ], [ 0.556 ], [ -0.488 ], [ 0.348 ], ... ],
    	[ [ -1.648 ], [ -0.948 ], [ -0.636 ], [ -1.084 ], [ -0.12 ], [ 0.976 ], [ -1.836 ], [ 0.388 ], ... ],
    	[ [ 0.94 ], [ 0.848 ], [ 0.176 ], [ -1.272 ], [ 0.124 ], [ 0.204 ], [ -1.748 ], [ 0.488 ], ... ],
    	[ [ 0.46 ], [ 1.748 ], [ -0.892 ], [ 1.972 ], [ -1.796 ], [ -0.944 ], [ 1.14 ], [ -0.244 ], ... ],
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
    F(0.0) = LineSearchPoint{point=PointSample{avg=2.6142408048000205}, derivative=-7.832617126400001E-4}
    New Minimum: 2.6142408048000205 > 2.614240804799943
    F(1.0E-10) = LineSearchPoint{point=PointSample{avg=2.614240804799943}, derivative=-7.832617126399844E-4}, delta = -7.771561172376096E-14
    New Minimum: 2.614240804799943 > 2.614240804799466
    F(7.000000000000001E-10) = LineSearchPoint{point=PointSample{avg=2.614240804799466}, derivative=-7.832617126398904E-4}, delta = -5.546674231027282E-13
    New Minimum: 2.614240804799466 > 2.614240804796176
    F(4.900000000000001E-9) = LineSearchPoint{point=PointSample{avg=2.614240804796176}, derivative=-7.832617126392325E-4}, delta = -3.844480289671992E-12
    New Minimum: 2.614240804796176 > 2.614240804773145
    F(3.430000000000001E-8) = LineSearchPoint{point=PointSample{avg=2.614240804773145}, derivative=-7.832617126346269E-4}, delta = -2.687539080170609E-11
    New Minimum: 2.614240804773145 > 2.6142408046119296
    F(2.4010000000000004E-7) =
```
...[skipping 5091 bytes](etc/112.txt)...
```
    .532977544742155E-36}, delta = 0.0
    F(5971.6481426938535) = LineSearchPoint{point=PointSample{avg=0.6560865231999966}, derivative=1.8844675125914983E-36}, delta = 0.0
    0.6560865231999966 <= 0.6560865231999966
    F(4075.6602072145542) = LineSearchPoint{point=PointSample{avg=0.6560865231999966}, derivative=-3.848723510994441E-37}, delta = 0.0
    Left bracket at 4075.6602072145542
    F(4397.21333641278) = LineSearchPoint{point=PointSample{avg=0.6560865231999966}, derivative=3.2319556172684383E-37}, delta = 0.0
    Right bracket at 4397.21333641278
    F(4250.441335685583) = LineSearchPoint{point=PointSample{avg=0.6560865231999966}, derivative=2.5809739317819398E-37}, delta = 0.0
    Right bracket at 4250.441335685583
    F(4180.281652377628) = LineSearchPoint{point=PointSample{avg=0.6560865231999966}, derivative=2.4740035153782234E-37}, delta = 0.0
    Right bracket at 4180.281652377628
    Converged to right
    Iteration 3 failed, aborting. Error: 0.6560865231999966 Total: 239574055548649.9700; Orientation: 0.0004; Line Search: 0.0270
    
```

Returns: 

```
    0.6560865231999966
```



This training run resulted in the following regressed input:

Code from [LearningTester.java:144](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L144) executed in 0.03 seconds: 
```java
    return Arrays.stream(input_gd).map(x -> x.prettyPrint()).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [
    	[ [ 1.752 ], [ -0.39799999999999996 ], [ -0.2580000000000001 ], [ -0.614 ], [ 0.4119999999999999 ], [ 0.69 ], [ -0.18400000000000008 ], [ 0.3060000000000001 ], ... ],
    	[ [ 0.029999999999999947 ], [ 0.48800000000000004 ], [ -0.21200000000000005 ], [ -0.21199999999999994 ], [ 0.634 ], [ 0.588 ], [ 0.24200000000000002 ], [ 0.624 ], ... ],
    	[ [ -0.71 ], [ 0.826 ], [ 0.158 ], [ 1.012 ], [ 0.5179999999999999 ], [ 0.004000000000000057 ], [ -0.5359999999999999 ], [ 1.194 ], ... ],
    	[ [ 1.462 ], [ -0.22999999999999995 ], [ -0.29599999999999993 ], [ 0.20599999999999988 ], [ -0.13799999999999993 ], [ -0.10599999999999996 ], [ -1.282 ], [ 0.556 ], ... ],
    	[ [ -1.3800000000000001 ], [ 0.32799999999999996 ], [ -0.86 ], [ 0.678 ], [ 0.37 ], [ 0.6040000000000001 ], [ -0.688 ], [ 0.558 ], ... ],
    	[ [ -1.616 ], [ -0.30999999999999994 ], [ -0.448 ], [ -1.4520000000000002 ], [ 0.592 ], [ -0.16600000000000006 ], [ -0.15200000000000002 ], [ 0.524 ], ... ],
    	[ [ 0.29999999999999993 ], [ 0.01400000000000001 ], [ 0.212 ], [ 0.254 ], [ -0.238 ], [ 0.524 ], [ -1.554 ], [ -0.18000000000000002 ], ... ],
    	[ [ 1.194 ], [ 1.8679999999999999 ], [ -0.896 ], [ 1.448 ], [ -1.4540000000000002 ], [ -1.248 ], [ 1.5439999999999998 ], [ -1.072 ], ... ],
    	...
    ]
    [
    	[ [ -1.496 ], [ -1.252 ], [ -1.168 ], [ 0.692 ], [ 1.764 ], [ 0.116 ], [ -1.356 ], [ 0.396 ], ... ],
    	[ [ 1.712 ], [ -1.8 ], [ -1.172 ], [ -0.492 ], [ -0.024 ], [ -1.612 ], [ 0.648 ], [ -1.932 ], ... ],
    	[ [ 0.068 ], [ 0.876 ], [ -1.716 ], [ 1.568 ], [ 1.372 ], [ -1.84 ], [ 1.472 ], [ 1.616 ], ... ],
    	[ [ -0.22 ], [ 0.628 ], [ 0.952 ], [ 1.684 ], [ 1.872 ], [ 1.096 ], [ -1.736 ], [ 1.24 ], ... ],
    	[ [ 0.94 ], [ 0.544 ], [ -0.368 ], [ 0.504 ], [ 0.172 ], [ 0.396 ], [ 1.268 ], [ 1.36 ], ... ],
    	[ [ 1.948 ], [ -1.324 ], [ 0.404 ], [ 1.26 ], [ 1.924 ], [ -1.912 ], [ -0.668 ], [ -1.3 ], ... ],
    	[ [ -0.7 ], [ 0.108 ], [ -1.228 ], [ 0.788 ], [ -1.668 ], [ 0.228 ], [ -1.272 ], [ 0.056 ], ... ],
    	[ [ -0.916 ], [ 0.708 ], [ 0.44 ], [ -0.648 ], [ -0.572 ], [ 0.22 ], [ -0.872 ], [ 1.5 ], ... ],
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
    th(0)=2.6142408048000205;dx=-7.832617126400001E-4
    New Minimum: 2.6142408048000205 > 2.612553682152812
    WOLFE (weak): th(2.154434690031884)=2.612553682152812; dx=-7.82924215398983E-4 delta=0.0016871226472083833
    New Minimum: 2.612553682152812 > 2.6108672866213665
    WOLFE (weak): th(4.308869380063768)=2.6108672866213665; dx=-7.825867181579659E-4 delta=0.003373518178654056
    New Minimum: 2.6108672866213665 > 2.6041289756532335
    WOLFE (weak): th(12.926608140191302)=2.6041289756532335; dx=-7.812367291938976E-4 delta=0.010111829146786988
    New Minimum: 2.6041289756532335 > 2.5739505452179343
    WOLFE (weak): th(51.70643256076521)=2.5739505452179343; dx=-7.751617788555902E-4 delta=0.04029025958208621
    New Minimum: 2.5739505452179343 > 2.4169776936893568
    WOLFE (weak): th(258.53216280382605)=2.4169776936893568; dx=-7.427620437179505E-4 delta=0.19726311111066375
    New Minimum: 2.4169776936893568 > 1.5877191431251305
    END: th(1551.1929768229563)=
```
...[skipping 2441 bytes](etc/113.txt)...
```
    0.000000000002)=0.6560866326672565; dx=-4.378690162971334E-10 delta=1.0837258149321016E-5
    Iteration 6 complete. Error: 0.6560866326672565 Total: 239574164735554.8400; Orientation: 0.0006; Line Search: 0.0084
    LBFGS Accumulation History: 1 points
    th(0)=0.6560866326672565;dx=-4.378690162971318E-11
    New Minimum: 0.6560866326672565 > 0.6560866197177768
    WOLF (strong): th(9694.956105143481)=0.6560866197177768; dx=4.1115516226347886E-11 delta=1.294947971075544E-8
    New Minimum: 0.6560866197177768 > 0.6560865233018578
    END: th(4847.478052571741)=0.6560865233018578; dx=-1.3356927016827011E-12 delta=1.0936539873718232E-7
    Iteration 7 complete. Error: 0.6560865233018578 Total: 239574174561894.8400; Orientation: 0.0005; Line Search: 0.0076
    LBFGS Accumulation History: 1 points
    th(0)=0.6560865233018578;dx=-4.07444904052719E-14
    MAX ALPHA: th(0)=0.6560865233018578;th'(0)=-4.07444904052719E-14;
    Iteration 8 failed, aborting. Error: 0.6560865233018578 Total: 239574181581197.8000; Orientation: 0.0005; Line Search: 0.0047
    
```

Returns: 

```
    0.6560865233018578
```



This training run resulted in the following regressed input:

Code from [LearningTester.java:154](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L154) executed in 0.01 seconds: 
```java
    return Arrays.stream(input_lbgfs).map(x -> x.prettyPrint()).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [
    	[ [ 1.7519821708982832 ], [ -0.3979923692598639 ], [ -0.2579869743698624 ], [ -0.6140007645164977 ], [ 0.412000201947754 ], [ 0.6899815506301815 ], [ -0.18399532635197724 ], [ 0.3060007068171392 ], ... ],
    	[ [ 0.030011987041688216 ], [ 0.48799120084785835 ], [ -0.21199163359304557 ], [ -0.2119852289642736 ], [ 0.6340051496677288 ], [ 0.5879959610449186 ], [ 0.24200399568056272 ], [ 0.623985113565557 ], ... ],
    	[ [ -0.7100005048693852 ], [ 0.8260011684120057 ], [ 0.1579910133249438 ], [ 1.0119966245875391 ], [ 0.5179878687099161 ], [ 0.00400075009165804 ], [ -0.5359972592804803 ], [ 1.1939893400435528 ], ... ],
    	[ [ 1.4619777424725333 ], [ -0.2299984276924861 ], [ -0.2960012405362035 ], [ 0.20598827260542424 ], [ -0.13799199421403496 ], [ -0.10599081137718959 ], [ -1.2819795600023198 ], [ 0.5559944897112817 ], ... ],
    	[ [ -1.3799920086388746 ], [ 0.32800080779101626 ], [ -0.8599984132676466 ], [ 0.67798126213339 ], [ 0.36999727370532004 ], [ 0.603990566154917 ], [ -0.6879857194088191 ], [ 0.5579844355980969
```
...[skipping 420 bytes](etc/114.txt)...
```
    , ... ],
    	[ [ 1.1939891092461197 ], [ 1.8680008366406955 ], [ -0.8960052217919268 ], [ 1.4479907392529918 ], [ -1.4540030436411506 ], [ -1.247986065604969 ], [ 1.5440027118698403 ], [ -1.0719790262832556 ], ... ],
    	...
    ]
    [
    	[ [ -1.496 ], [ -1.252 ], [ -1.168 ], [ 0.692 ], [ 1.764 ], [ 0.116 ], [ -1.356 ], [ 0.396 ], ... ],
    	[ [ 1.712 ], [ -1.8 ], [ -1.172 ], [ -0.492 ], [ -0.024 ], [ -1.612 ], [ 0.648 ], [ -1.932 ], ... ],
    	[ [ 0.068 ], [ 0.876 ], [ -1.716 ], [ 1.568 ], [ 1.372 ], [ -1.84 ], [ 1.472 ], [ 1.616 ], ... ],
    	[ [ -0.22 ], [ 0.628 ], [ 0.952 ], [ 1.684 ], [ 1.872 ], [ 1.096 ], [ -1.736 ], [ 1.24 ], ... ],
    	[ [ 0.94 ], [ 0.544 ], [ -0.368 ], [ 0.504 ], [ 0.172 ], [ 0.396 ], [ 1.268 ], [ 1.36 ], ... ],
    	[ [ 1.948 ], [ -1.324 ], [ 0.404 ], [ 1.26 ], [ 1.924 ], [ -1.912 ], [ -0.668 ], [ -1.3 ], ... ],
    	[ [ -0.7 ], [ 0.108 ], [ -1.228 ], [ 0.788 ], [ -1.668 ], [ 0.228 ], [ -1.272 ], [ 0.056 ], ... ],
    	[ [ -0.916 ], [ 0.708 ], [ 0.44 ], [ -0.648 ], [ -0.572 ], [ 0.22 ], [ -0.872 ], [ 1.5 ], ... ],
    	...
    ]
```



Code from [LearningTester.java:95](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L95) executed in 0.00 seconds: 
```java
    return TestUtil.compare(runs);
```

Returns: 

![Result](etc/test.71.png)



Code from [LearningTester.java:98](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L98) executed in 0.00 seconds: 
```java
    return TestUtil.compareTime(runs);
```

Returns: 

![Result](etc/test.72.png)



### Performance
Now we execute larger-scale runs to benchmark performance:

Code from [PerformanceTester.java:66](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/PerformanceTester.java#L66) executed in 0.58 seconds: 
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
    	Evaluation performance: 0.015107s +- 0.000581s [0.014265s - 0.015896s]
    	Learning performance: 0.082115s +- 0.030337s [0.053830s - 0.123614s]
    
```

