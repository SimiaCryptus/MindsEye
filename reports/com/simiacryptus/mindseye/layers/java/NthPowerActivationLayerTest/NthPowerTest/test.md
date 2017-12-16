# NthPowerActivationLayer
## NthPowerTest
### Batch Execution
Code from [BatchingTester.java:66](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/BatchingTester.java#L66) executed in 0.00 seconds: 
```java
    return test(reference, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (120#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (50#)}
```



Code from [SingleDerivativeTester.java:77](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/SingleDerivativeTester.java#L77) executed in 0.00 seconds: 
```java
    return test(component, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ -1.564 ], [ -0.228 ], [ 0.948 ] ],
    	[ [ 1.96 ], [ -1.988 ], [ -0.336 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.059001389688339345, negative=4, min=-0.336, max=-0.336, mean=-0.20133333333333336, count=6.0, positive=2, stdDev=1.3577072176610423, zeros=0}
    Output: [
    	[ [ 0.0 ], [ 0.0 ], [ 0.875025769835362 ] ],
    	[ [ 5.37824 ], [ 0.0 ], [ 0.0 ] ]
    ]
    Outputs Statistics: {meanExponent=0.33633051086817783, negative=0, min=0.0, max=0.0, mean=1.042210961639227, count=6.0, positive=2, stdDev=1.9652783290630955, zeros=4}
    Feedback for input 0
    Inputs Values: [
    	[ [ -1.564 ], [ -0.228 ], [ 0.948 ] ],
    	[ [ 1.96 ], [ -1.988 ], [ -0.336 ] ]
    ]
    Value Statistics: {meanExponent=-0.059001389688339345, negative=4, min=-0.336, max=-0.336, mean=-0.20133333333333336, count=6.0, positive=2, stdDev=1.3577072176610423, zeros=0}
    Implemented Feedback: [ [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 6.859999999999999, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.
```
...[skipping 262 bytes](etc/303.txt)...
```
     [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 6.860262502232928, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 2.3077399730342396, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ] ]
    Measured Statistics: {meanExponent=0.5997638033030684, negative=0, min=0.0, max=0.0, mean=0.25466673542408796, count=36.0, positive=2, stdDev=1.1791484700827661, zeros=34}
    Feedback Error: [ [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 2.6250223292834107E-4, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 1.8256313085895925E-4, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ] ]
    Error Statistics: {meanExponent=-3.659726961470205, negative=0, min=0.0, max=0.0, mean=1.2362926771869454E-5, count=36.0, positive=2, stdDev=5.183692606525044E-5, zeros=34}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 1.2363e-05 +- 5.1837e-05 [0.0000e+00 - 2.6250e-04] (36#)
    relativeTol: 2.9344e-05 +- 1.0212e-05 [1.9132e-05 - 3.9556e-05] (2#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=1.2363e-05 +- 5.1837e-05 [0.0000e+00 - 2.6250e-04] (36#), relativeTol=2.9344e-05 +- 1.0212e-05 [1.9132e-05 - 3.9556e-05] (2#)}
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
      "id": "1703c19b-2a82-469a-9176-0728668e1227",
      "isFrozen": false,
      "name": "NthPowerActivationLayer/1703c19b-2a82-469a-9176-0728668e1227",
      "power": 2.5
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
    	[ [ -0.696 ], [ 1.568 ], [ -0.324 ] ],
    	[ [ -1.02 ], [ 0.156 ], [ -0.972 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 0.0 ], [ 3.0786842212629684 ], [ 0.0 ] ],
    	[ [ 0.0 ], [ 0.009611949842565764 ], [ 0.0 ] ]
    ]
    --------------------
    Derivative: 
    [
    	[ [ 0.0 ], [ 4.908616424207539 ], [ 0.0 ] ],
    	[ [ 0.0 ], [ 0.15403765773342568 ], [ 0.0 ] ]
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
    	[ [ 1.784 ], [ 0.104 ], [ -1.284 ], [ 0.752 ], [ -0.08 ], [ -0.276 ], [ -0.94 ], [ -1.064 ], ... ],
    	[ [ -1.812 ], [ -1.972 ], [ -1.484 ], [ 0.824 ], [ 1.744 ], [ 0.984 ], [ -0.104 ], [ -0.116 ], ... ],
    	[ [ 0.224 ], [ 0.304 ], [ 1.96 ], [ -0.412 ], [ 0.688 ], [ 1.148 ], [ 0.852 ], [ 0.152 ], ... ],
    	[ [ -1.6 ], [ -0.128 ], [ 1.584 ], [ 1.312 ], [ -0.216 ], [ -1.708 ], [ 1.756 ], [ 0.348 ], ... ],
    	[ [ -1.036 ], [ 1.668 ], [ 1.496 ], [ 0.852 ], [ 0.636 ], [ 0.372 ], [ -1.72 ], [ 0.768 ], ... ],
    	[ [ 1.824 ], [ 0.472 ], [ 1.748 ], [ -1.092 ], [ -1.68 ], [ 0.964 ], [ 1.952 ], [ 0.592 ], ... ],
    	[ [ 0.152 ], [ 1.064 ], [ 0.812 ], [ 1.844 ], [ 0.712 ], [ -0.716 ], [ -0.7 ], [ 1.0 ], ... ],
    	[ [ -0.188 ], [ 1.768 ], [ 0.18 ], [ 1.22 ], [ -0.224 ], [ -0.704 ], [ -0.388 ], [ 0.804 ], ... ],
    	...
    ]
```



First, we use a conjugate gradient descent method, which converges the fastest for purely linear functions.

Code from [LearningTester.java:225](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L225) executed in 3.53 seconds: 
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
    F(0.0) = LineSearchPoint{point=PointSample{avg=4.132354025687139}, derivative=-0.028841132430560464}
    New Minimum: 4.132354025687139 > 4.132354025684255
    F(1.0E-10) = LineSearchPoint{point=PointSample{avg=4.132354025684255}, derivative=-0.028841132430527213}, delta = -2.8839153287663066E-12
    New Minimum: 4.132354025684255 > 4.132354025666945
    F(7.000000000000001E-10) = LineSearchPoint{point=PointSample{avg=4.132354025666945}, derivative=-0.028841132430327713}, delta = -2.0193624550302047E-11
    New Minimum: 4.132354025666945 > 4.132354025545809
    F(4.900000000000001E-9) = LineSearchPoint{point=PointSample{avg=4.132354025545809}, derivative=-0.02884113242893119}, delta = -1.4132961467794303E-10
    New Minimum: 4.132354025545809 > 4.132354024697902
    F(3.430000000000001E-8) = LineSearchPoint{point=PointSample{avg=4.132354024697902}, derivative=-0.028841132419155542}, delta = -9.892371366504449E-10
    New Minimum: 4.132354024697902 > 4.1323540187623875
    F(2.4010000000000004E-7) =
```
...[skipping 295808 bytes](etc/304.txt)...
```
    LineSearchPoint{point=PointSample{avg=1.357810005833295}, derivative=1.5244732185480252E-11}, delta = -1.557406528629457E-6
    Right bracket at 243.69363150936468
    Converged to right
    Iteration 249 complete. Error: 1.357810005833295 Total: 239698451076302.5600; Orientation: 0.0004; Line Search: 0.0152
    F(0.0) = LineSearchPoint{point=PointSample{avg=1.357810005833295}, derivative=-1.845403264437191E-8}
    New Minimum: 1.357810005833295 > 1.3578087062062862
    F(243.69363150936468) = LineSearchPoint{point=PointSample{avg=1.3578087062062862}, derivative=7.828049138068314E-9}, delta = -1.299627008677362E-6
    1.3578087062062862 <= 1.357810005833295
    New Minimum: 1.3578087062062862 > 1.3578084231637217
    F(171.110122414428) = LineSearchPoint{point=PointSample{avg=1.3578084231637217}, derivative=-2.5297958579054562E-11}, delta = -1.5826695731746554E-6
    Left bracket at 171.110122414428
    Converged to left
    Iteration 250 complete. Error: 1.3578084231637217 Total: 239698459142046.5600; Orientation: 0.0003; Line Search: 0.0065
    
```

Returns: 

```
    1.3578084231637217
```



This training run resulted in the following regressed input:

Code from [LearningTester.java:144](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L144) executed in 0.00 seconds: 
```java
    return Arrays.stream(input_gd).map(x -> x.prettyPrint()).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [
    	[ [ 1.784 ], [ -1.516 ], [ -1.436 ], [ -0.288 ], [ 0.2149545115839343 ], [ 0.23341380037764065 ], [ 0.07897973306345553 ], [ 0.23305659457608707 ], ... ],
    	[ [ -0.6457314541481138 ], [ -0.017505939032945506 ], [ -0.392 ], [ 0.8240000000000003 ], [ -1.044 ], [ 0.9839999999999999 ], [ 0.23305191527173716 ], [ -1.676 ], ... ],
    	[ [ 0.27285850037943443 ], [ -1.82 ], [ 1.9599999999999989 ], [ -0.732 ], [ 0.6879999999208259 ], [ 1.1479999999999997 ], [ 0.8520000000000002 ], [ 0.2477867356520071 ], ... ],
    	[ [ 0.09756729832232212 ], [ 0.2329289287945655 ], [ -0.996 ], [ -1.904 ], [ 0.23275544928775957 ], [ 0.231728250838484 ], [ -0.44 ], [ -1.312 ], ... ],
    	[ [ 0.23333575891730285 ], [ 1.668 ], [ -1.288 ], [ 0.8520000000000002 ], [ -1.12 ], [ 0.02381998280844256 ], [ 0.055750900031123776 ], [ 0.7680000000000057 ], ... ],
    	[ [ 1.824 ], [ -1.348 ], [ -1.848 ], [ -0.86 ], [ 0.23102017102925346 ], [ 0.9639999999999997 ], [ 1.952 ], [ -0.24 ], ... ],
    	[ [ 0.24813472110664614 ], [ 1.0639999999999998 ], [ 0.8120000000000003 ], [ -1.4 ], [ 0.7119999999954905 ], [ 0.20680584124770574 ], [ -0.908 ], [ -1.976 ], ... ],
    	[ [ -1.28 ], [ 1.768 ], [ 0.25487339428707256 ], [ 1.22 ], [ 0.23322646000635755 ], [ -1.324 ], [ -1.468 ], [ -0.272 ], ... ],
    	...
    ]
```



Next, we run the same optimization using L-BFGS, which is nearly ideal for purely second-order or quadratic functions.

Code from [LearningTester.java:249](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L249) executed in 1.08 seconds: 
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
    th(0)=4.132354025687139;dx=-0.028841132430560464
    New Minimum: 4.132354025687139 > 4.070984229343241
    WOLFE (weak): th(2.154434690031884)=4.070984229343241; dx=-0.02813191534876588 delta=0.061369796343898386
    New Minimum: 4.070984229343241 > 4.011127113072824
    WOLFE (weak): th(4.308869380063768)=4.011127113072824; dx=-0.027436851399531895 delta=0.12122691261431484
    New Minimum: 4.011127113072824 > 3.7862262764283283
    END: th(12.926608140191302)=3.7862262764283283; dx=-0.024794173684359933 delta=0.3461277492588106
    Iteration 1 complete. Error: 3.7862262764283283 Total: 239698494088091.5300; Orientation: 0.0010; Line Search: 0.0097
    LBFGS Accumulation History: 1 points
    th(0)=3.7862262764283283;dx=-0.02136527638114972
    New Minimum: 3.7862262764283283 > 3.267432678249305
    END: th(27.849533001676672)=3.267432678249305; dx=-0.016093628037815607 delta=0.5187935981790233
    Iteration 2 complete. Error: 3.267432678249305 Total: 2396984999526
```
...[skipping 60851 bytes](etc/305.txt)...
```
    3613 > 1.356773409565662
    WOLFE (weak): th(221.01186334718258)=1.356773409565662; dx=-1.028930484046721E-8 delta=2.291067950910275E-6
    New Minimum: 1.356773409565662 > 1.3567711530257485
    WOLFE (weak): th(442.02372669436517)=1.3567711530257485; dx=-1.0129605568925363E-8 delta=4.547607864502368E-6
    New Minimum: 1.3567711530257485 > 1.3567625038565605
    WOLFE (weak): th(1326.0711800830954)=1.3567625038565605; dx=-9.417475473313968E-9 delta=1.319677705247102E-5
    New Minimum: 1.3567625038565605 > 1.3567338940729305
    END: th(5304.284720332382)=1.3567338940729305; dx=-4.424704687008935E-9 delta=4.18065606824225E-5
    Iteration 122 complete. Error: 1.3567338940729305 Total: 239699546277037.4700; Orientation: 0.0005; Line Search: 0.0098
    LBFGS Accumulation History: 1 points
    th(0)=1.3567338940729305;dx=-2.3941988989644724E-8
    MAX ALPHA: th(0)=1.3567338940729305;th'(0)=-2.3941988989644724E-8;
    Iteration 123 failed, aborting. Error: 1.3567338940729305 Total: 239699551207733.4700; Orientation: 0.0005; Line Search: 0.0032
    
```

Returns: 

```
    1.3567338940729305
```



This training run resulted in the following regressed input:

Code from [LearningTester.java:154](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L154) executed in 0.00 seconds: 
```java
    return Arrays.stream(input_lbgfs).map(x -> x.prettyPrint()).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [
    	[ [ 1.784 ], [ -1.516 ], [ -1.436 ], [ -0.288 ], [ 0.23741156286629314 ], [ 0.23712568792134256 ], [ 0.07905423416620215 ], [ 0.23734446796988637 ], ... ],
    	[ [ 0.23742430773968287 ], [ 0.2374197003446803 ], [ -0.392 ], [ 0.8240000000000051 ], [ -1.044 ], [ 0.984 ], [ 0.23666950772991843 ], [ -1.676 ], ... ],
    	[ [ 0.2752757135327058 ], [ -1.82 ], [ 1.9600000002477125 ], [ -0.732 ], [ 0.6880000009126894 ], [ 1.148 ], [ 0.852 ], [ 0.2516233215738733 ], ... ],
    	[ [ 0.09773986048841653 ], [ 0.23653918246845135 ], [ -0.996 ], [ -1.904 ], [ 0.23736367360351077 ], [ 0.23532233047112097 ], [ -0.44 ], [ -1.312 ], ... ],
    	[ [ 0.23699838856684888 ], [ 1.668 ], [ -1.288 ], [ 0.852 ], [ -1.12 ], [ 0.37615335800801214 ], [ 0.05576946823506708 ], [ 0.7680000000010776 ], ... ],
    	[ [ 1.824 ], [ -1.348 ], [ -1.848 ], [ -0.86 ], [ 0.23739285948219968 ], [ 0.964 ], [ 1.952000003979285 ], [ -0.24 ], ... ],
    	[ [ 0.2515511074710917 ], [ 1.064 ], [ 0.8120000000000226 ], [ -1.4 ], [ 0.7120000001502812 ], [ 0.23741310904133955 ], [ -0.908 ], [ -1.976 ], ... ],
    	[ [ -1.28 ], [ 1.768 ], [ 0.2593198538179171 ], [ 1.22 ], [ 0.23686319370128303 ], [ -1.324 ], [ -1.468 ], [ -0.272 ], ... ],
    	...
    ]
```



Code from [LearningTester.java:95](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L95) executed in 0.00 seconds: 
```java
    return TestUtil.compare(runs);
```

Returns: 

![Result](etc/test.209.png)



Code from [LearningTester.java:98](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L98) executed in 0.00 seconds: 
```java
    return TestUtil.compareTime(runs);
```

Returns: 

![Result](etc/test.210.png)



### Performance
Now we execute larger-scale runs to benchmark performance:

Code from [PerformanceTester.java:66](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/PerformanceTester.java#L66) executed in 0.20 seconds: 
```java
    test(component, inputPrototype);
```
Logging: 
```
    100 batches
    Input Dimensions:
    	[100, 100, 1]
    Performance:
    	Evaluation performance: 0.009370s +- 0.000791s [0.008593s - 0.010882s]
    	Learning performance: 0.012534s +- 0.001031s [0.011125s - 0.013886s]
    
```

### Function Plots
Code from [ActivationLayerTestBase.java:110](../../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/java/ActivationLayerTestBase.java#L110) executed in 0.00 seconds: 
```java
    return plot("Value Plot", plotData, x -> new double[]{x[0], x[1]});
```

Returns: 

![Result](etc/test.211.png)



Code from [ActivationLayerTestBase.java:114](../../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/java/ActivationLayerTestBase.java#L114) executed in 0.00 seconds: 
```java
    return plot("Derivative Plot", plotData, x -> new double[]{x[0], x[2]});
```

Returns: 

![Result](etc/test.212.png)



