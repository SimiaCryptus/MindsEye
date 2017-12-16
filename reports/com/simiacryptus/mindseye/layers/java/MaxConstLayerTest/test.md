# MaxConstLayer
## MaxConstLayerTest
### Batch Execution
Code from [BatchingTester.java:66](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/BatchingTester.java#L66) executed in 0.00 seconds: 
```java
    return test(reference, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (120#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (51#)}
```



Code from [SingleDerivativeTester.java:77](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/SingleDerivativeTester.java#L77) executed in 0.00 seconds: 
```java
    return test(component, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ 1.4 ], [ -0.876 ], [ 0.572 ] ],
    	[ [ 0.4 ], [ -1.124 ], [ 1.668 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.04649158008298876, negative=2, min=1.668, max=1.668, mean=0.3399999999999999, count=6.0, positive=4, stdDev=1.0461484916906714, zeros=0}
    Output: [
    	[ [ 1.4 ], [ 0.0 ], [ 0.572 ] ],
    	[ [ 0.4 ], [ 0.0 ], [ 1.668 ] ]
    ]
    Outputs Statistics: {meanExponent=-0.06805497447476389, negative=0, min=1.668, max=1.668, mean=0.6733333333333333, count=6.0, positive=4, stdDev=0.6466762885882102, zeros=2}
    Feedback for input 0
    Inputs Values: [
    	[ [ 1.4 ], [ -0.876 ], [ 0.572 ] ],
    	[ [ 0.4 ], [ -1.124 ], [ 1.668 ] ]
    ]
    Value Statistics: {meanExponent=-0.04649158008298876, negative=2, min=1.668, max=1.668, mean=0.3399999999999999, count=6.0, positive=4, stdDev=1.0461484916906714, zeros=0}
    Implemented Feedback: [ [ 1.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 1.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 1.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 
```
...[skipping 344 bytes](etc/276.txt)...
```
     0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.9999999999998899, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.9999999999998899 ] ]
    Measured Statistics: {meanExponent=-4.7830642341045674E-14, negative=0, min=0.9999999999998899, max=0.9999999999998899, mean=0.11111111111109888, count=36.0, positive=4, stdDev=0.31426968052731985, zeros=32}
    Feedback Error: [ [ -1.1013412404281553E-13, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, -1.1013412404281553E-13, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, -1.1013412404281553E-13, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, -1.1013412404281553E-13 ] ]
    Error Statistics: {meanExponent=-12.958078098036825, negative=4, min=-1.1013412404281553E-13, max=-1.1013412404281553E-13, mean=-1.223712489364617E-14, count=36.0, positive=0, stdDev=3.461181597809566E-14, zeros=32}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 1.2237e-14 +- 3.4612e-14 [0.0000e+00 - 1.1013e-13] (36#)
    relativeTol: 5.5067e-14 +- 0.0000e+00 [5.5067e-14 - 5.5067e-14] (4#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=1.2237e-14 +- 3.4612e-14 [0.0000e+00 - 1.1013e-13] (36#), relativeTol=5.5067e-14 +- 0.0000e+00 [5.5067e-14 - 5.5067e-14] (4#)}
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
      "class": "com.simiacryptus.mindseye.layers.java.MaxConstLayer",
      "id": "20a72860-570d-4a73-babf-1e2358b03328",
      "isFrozen": true,
      "name": "MaxConstLayer/20a72860-570d-4a73-babf-1e2358b03328",
      "value": 0.0
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
    	[ [ 1.688 ], [ -1.568 ], [ 0.828 ] ],
    	[ [ 0.716 ], [ -0.8 ], [ -1.612 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 1.688 ], [ 0.0 ], [ 0.828 ] ],
    	[ [ 0.716 ], [ 0.0 ], [ 0.0 ] ]
    ]
    --------------------
    Derivative: 
    [
    	[ [ 1.0 ], [ 0.0 ], [ 1.0 ] ],
    	[ [ 1.0 ], [ 0.0 ], [ 0.0 ] ]
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
    	[ [ 0.636 ], [ 1.696 ], [ -0.304 ], [ -1.888 ], [ 1.272 ], [ 1.628 ], [ 1.7 ], [ 1.72 ], ... ],
    	[ [ 0.816 ], [ -0.484 ], [ -0.232 ], [ 0.416 ], [ -1.52 ], [ 1.088 ], [ 1.46 ], [ -0.952 ], ... ],
    	[ [ -0.292 ], [ -0.6 ], [ 0.196 ], [ -1.48 ], [ -0.984 ], [ -1.692 ], [ 1.484 ], [ -0.48 ], ... ],
    	[ [ 1.164 ], [ 1.044 ], [ -1.204 ], [ 0.516 ], [ 1.2 ], [ -1.276 ], [ -1.404 ], [ 1.044 ], ... ],
    	[ [ 0.216 ], [ 1.176 ], [ -0.684 ], [ -1.164 ], [ -0.208 ], [ 0.172 ], [ 0.96 ], [ 1.356 ], ... ],
    	[ [ 1.712 ], [ -0.756 ], [ 1.76 ], [ 0.776 ], [ 1.952 ], [ 0.172 ], [ -1.176 ], [ 1.648 ], ... ],
    	[ [ -0.092 ], [ -0.544 ], [ 0.548 ], [ -0.296 ], [ 0.564 ], [ 0.784 ], [ 0.852 ], [ 0.596 ], ... ],
    	[ [ -1.828 ], [ 1.52 ], [ -0.492 ], [ -0.124 ], [ -1.276 ], [ 1.784 ], [ 0.02 ], [ -0.348 ], ... ],
    	...
    ]
```



First, we use a conjugate gradient descent method, which converges the fastest for purely linear functions.

Code from [LearningTester.java:225](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L225) executed in 0.04 seconds: 
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
    F(0.0) = LineSearchPoint{point=PointSample{avg=0.8258343807999963}, derivative=-1.9878584704000003E-4}
    New Minimum: 0.8258343807999963 > 0.8258343807999756
    F(1.0E-10) = LineSearchPoint{point=PointSample{avg=0.8258343807999756}, derivative=-1.9878584703999605E-4}, delta = -2.0650148258027912E-14
    New Minimum: 0.8258343807999756 > 0.8258343807998625
    F(7.000000000000001E-10) = LineSearchPoint{point=PointSample{avg=0.8258343807998625}, derivative=-1.987858470399722E-4}, delta = -1.3378187446733136E-13
    New Minimum: 0.8258343807998625 > 0.8258343807990275
    F(4.900000000000001E-9) = LineSearchPoint{point=PointSample{avg=0.8258343807990275}, derivative=-1.987858470398052E-4}, delta = -9.687806112879116E-13
    New Minimum: 0.8258343807990275 > 0.8258343807931755
    F(3.430000000000001E-8) = LineSearchPoint{point=PointSample{avg=0.8258343807931755}, derivative=-1.9878584703863635E-4}, delta = -6.820766174087112E-12
    New Minimum: 0.8258343807931755 > 0.8258343807522704
    F(2.40100
```
...[skipping 3242 bytes](etc/277.txt)...
```
    lta = 0.0
    Right bracket at 5997.6799784203595
    F(5620.555279099984) = LineSearchPoint{point=PointSample{avg=0.3288697632000001}, derivative=9.070694992786899E-31}, delta = 0.0
    Right bracket at 5620.555279099984
    F(5395.359297619261) = LineSearchPoint{point=PointSample{avg=0.3288697632000001}, derivative=5.778129854539405E-31}, delta = 0.0
    Right bracket at 5395.359297619261
    F(5255.622382120227) = LineSearchPoint{point=PointSample{avg=0.3288697632000001}, derivative=3.735816184302753E-31}, delta = 0.0
    Right bracket at 5255.622382120227
    F(5166.803129479999) = LineSearchPoint{point=PointSample{avg=0.3288697632000001}, derivative=2.4379042257340137E-31}, delta = 0.0
    Right bracket at 5166.803129479999
    F(5109.484809337583) = LineSearchPoint{point=PointSample{avg=0.3288697632000001}, derivative=1.5999944876961E-31}, delta = 0.0
    Right bracket at 5109.484809337583
    Converged to right
    Iteration 2 failed, aborting. Error: 0.3288697632000001 Total: 239671886371416.1200; Orientation: 0.0003; Line Search: 0.0222
    
```

Returns: 

```
    0.3288697632000001
```



This training run resulted in the following regressed input:

Code from [LearningTester.java:144](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L144) executed in 0.01 seconds: 
```java
    return Arrays.stream(input_gd).map(x -> x.prettyPrint()).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [
    	[ [ -0.416 ], [ -1.396 ], [ -4.0841658874222166E-15 ], [ -0.34 ], [ 1.2720000000000085 ], [ -1.776 ], [ 1.7000000000000026 ], [ -1.12 ], ... ],
    	[ [ 0.8159999999999944 ], [ -9.816584436554072E-15 ], [ -9.904102276998946E-15 ], [ -0.936 ], [ -1.882849095124116E-15 ], [ 1.0880000000000052 ], [ 1.4599999999999989 ], [ -0.028 ], ... ],
    	[ [ -1.128 ], [ -7.0962382293960795E-15 ], [ -0.896 ], [ -1.3468023223999516E-14 ], [ -5.214604659833706E-16 ], [ -1.644 ], [ -0.772 ], [ -9.560108543028451E-16 ], ... ],
    	[ [ 1.1639999999999964 ], [ 1.0440000000000038 ], [ -0.276 ], [ -1.508 ], [ 1.200000000000004 ], [ -0.672 ], [ -0.096 ], [ -0.656 ], ... ],
    	[ [ 0.21599999999999034 ], [ 1.1759999999999964 ], [ -1.3842405097013164E-14 ], [ -0.328 ], [ -9.93327489048047E-15 ], [ 0.17199999999999688 ], [ 0.9600000000000048 ], [ 1.3560000000000025 ], ... ],
    	[ [ 1.7120000000000015 ], [ -0.584 ], [ 1.7600000000000127 ], [ -0.768 ], [ -1.656 ], [ -1.732 ], [ -3.0704175689370623E-15 ], [ -1.02 ], ... ],
    	[ [ -1.08 ], [ -1.1377319257819088E-14 ], [ -0.368 ], [ -1.3380505383554641E-14 ], [ -1.072 ], [ -1.084 ], [ -0.556 ], [ 0.5960000000000014 ], ... ],
    	[ [ -1.652 ], [ -0.332 ], [ -9.383857336577224E-15 ], [ -5.3872092895998065E-15 ], [ -0.956 ], [ 1.7840000000000047 ], [ -0.572 ], [ -7.818868175846297E-16 ], ... ],
    	...
    ]
```



Next, we run the same optimization using L-BFGS, which is nearly ideal for purely second-order or quadratic functions.

Code from [LearningTester.java:249](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L249) executed in 0.05 seconds: 
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
    th(0)=0.8258343807999963;dx=-1.9878584704000003E-4
    New Minimum: 0.8258343807999963 > 0.8254062019434661
    WOLFE (weak): th(2.154434690031884)=0.8254062019434661; dx=-1.9870019281504996E-4 delta=4.281788565301259E-4
    New Minimum: 0.8254062019434661 > 0.8249782076233636
    WOLFE (weak): th(4.308869380063768)=0.8249782076233636; dx=-1.9861453859009988E-4 delta=8.561731766326686E-4
    New Minimum: 0.8249782076233636 > 0.8232680757073048
    WOLFE (weak): th(12.926608140191302)=0.8232680757073048; dx=-1.982719216902996E-4 delta=0.002566305092691512
    New Minimum: 0.8232680757073048 > 0.8156090202988604
    WOLFE (weak): th(51.70643256076521)=0.8156090202988604; dx=-1.9673014564119834E-4 delta=0.010225360501135872
    New Minimum: 0.8156090202988604 > 0.7757705081517347
    WOLFE (weak): th(258.53216280382605)=0.7757705081517347; dx=-1.8850734004599167E-4 delta=0.05006387264826162
    New Minimum: 0.7757705081517347 > 0.5653110145637401
    END: th(1551.192976
```
...[skipping 2436 bytes](etc/278.txt)...
```
    02)=0.32886979098195485; dx=-1.1112781576392767E-10 delta=2.7504134410549774E-6
    Iteration 6 complete. Error: 0.32886979098195485 Total: 239671934633361.0600; Orientation: 0.0005; Line Search: 0.0042
    LBFGS Accumulation History: 1 points
    th(0)=0.32886979098195485;dx=-1.1112781576392607E-11
    New Minimum: 0.32886979098195485 > 0.32886977143721174
    WOLF (strong): th(9694.956105143481)=0.32886977143721174; dx=3.5089622680834503E-12 delta=1.9544743101729267E-8
    New Minimum: 0.32886977143721174 > 0.3288697632258505
    END: th(4847.478052571741)=0.3288697632258505; dx=-3.389886174752645E-13 delta=2.775610435934439E-8
    Iteration 7 complete. Error: 0.3288697632258505 Total: 239671940284768.0600; Orientation: 0.0005; Line Search: 0.0044
    LBFGS Accumulation History: 1 points
    th(0)=0.3288697632258505;dx=-1.0340640818668368E-14
    MAX ALPHA: th(0)=0.3288697632258505;th'(0)=-1.0340640818668368E-14;
    Iteration 8 failed, aborting. Error: 0.3288697632258505 Total: 239671944108893.0600; Orientation: 0.0005; Line Search: 0.0025
    
```

Returns: 

```
    0.3288697632258505
```



This training run resulted in the following regressed input:

Code from [LearningTester.java:154](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L154) executed in 0.01 seconds: 
```java
    return Arrays.stream(input_lbgfs).map(x -> x.prettyPrint()).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [
    	[ [ -0.416 ], [ -1.396 ], [ 4.067804760600024E-6 ], [ -0.34 ], [ 1.2719915181943289 ], [ -1.776 ], [ 1.699997403528876 ], [ -1.12 ], ... ],
    	[ [ 0.8160055391383974 ], [ 9.780041232931923E-6 ], [ 9.866590270391515E-6 ], [ -0.936 ], [ 1.8752291449574489E-6 ], [ 1.08799489360679 ], [ 1.4600012116865244 ], [ -0.028 ], ... ],
    	[ [ -1.128 ], [ 7.0681713925319425E-6 ], [ -0.896 ], [ 1.3415100806234094E-5 ], [ 5.192942247574487E-7 ], [ -1.644 ], [ -0.772 ], [ 9.52039412055322E-7 ], ... ],
    	[ [ 1.1640035485105358 ], [ 1.0439961341429935 ], [ -0.276 ], [ -1.508 ], [ 1.1999959898945978 ], [ -0.672 ], [ -0.096 ], [ -0.656 ], ... ],
    	[ [ 0.21600963579283716 ], [ 1.1760035485105358 ], [ 1.379014663522557E-5 ], [ -0.328 ], [ 9.895439949544676E-6 ], [ 0.17200308691566937 ], [ 0.9599951244042231 ], [ 1.3559975766269512 ], ... ],
    	[ [ 1.711998499816684 ], [ -0.584 ], [ 1.7599873349908517 ], [ -0.768 ], [ -1.656 ], [ -1.732 ], [ 3.0580659902383086E-6 ], [ -1.02 ], ... ],
    	[ [ -1.08 ], [ 1.1337923907204313E-5 ], [ -0.368 ], [ 1.3328551768774502E-5 ], [ -1.072 ], [ -1.084 ], [ -0.556 ], [ 0.5959986152154007 ], ... ],
    	[ [ -1.652 ], [ -0.332 ], [ 9.347296045634015E-6 ], [ 5.3660403224936105E-6 ], [ -0.956 ], [ 1.7839953263519772 ], [ -0.572 ], [ 7.78941337136168E-7 ], ... ],
    	...
    ]
```



Code from [LearningTester.java:95](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L95) executed in 0.00 seconds: 
```java
    return TestUtil.compare(runs);
```

Returns: 

![Result](etc/test.184.png)



Code from [LearningTester.java:98](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L98) executed in 0.00 seconds: 
```java
    return TestUtil.compareTime(runs);
```

Returns: 

![Result](etc/test.185.png)



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
    	Evaluation performance: 0.009492s +- 0.008687s [0.004005s - 0.026768s]
    	Learning performance: 0.018864s +- 0.005413s [0.013973s - 0.029363s]
    
```

### Function Plots
Code from [ActivationLayerTestBase.java:110](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/java/ActivationLayerTestBase.java#L110) executed in 0.00 seconds: 
```java
    return plot("Value Plot", plotData, x -> new double[]{x[0], x[1]});
```

Returns: 

![Result](etc/test.186.png)



Code from [ActivationLayerTestBase.java:114](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/java/ActivationLayerTestBase.java#L114) executed in 0.00 seconds: 
```java
    return plot("Derivative Plot", plotData, x -> new double[]{x[0], x[2]});
```

Returns: 

![Result](etc/test.187.png)



