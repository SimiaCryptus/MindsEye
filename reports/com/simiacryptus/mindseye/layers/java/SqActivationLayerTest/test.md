# SqActivationLayer
## SqActivationLayerTest
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
    	[ [ -0.092 ], [ 1.572 ], [ 0.464 ] ],
    	[ [ -0.78 ], [ -1.172 ], [ -1.308 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.18260194867256233, negative=4, min=-1.308, max=-1.308, mean=-0.2193333333333333, count=6.0, positive=2, stdDev=1.0082285896010335, zeros=0}
    Output: [
    	[ [ 0.008464 ], [ 2.471184 ], [ 0.21529600000000002 ] ],
    	[ [ 0.6084 ], [ 1.373584 ], [ 1.7108640000000002 ] ]
    ]
    Outputs Statistics: {meanExponent=-0.36520389734512465, negative=0, min=1.7108640000000002, max=1.7108640000000002, mean=1.064632, count=6.0, positive=6, stdDev=0.869527245861796, zeros=0}
    Feedback for input 0
    Inputs Values: [
    	[ [ -0.092 ], [ 1.572 ], [ 0.464 ] ],
    	[ [ -0.78 ], [ -1.172 ], [ -1.308 ] ]
    ]
    Value Statistics: {meanExponent=-0.18260194867256233, negative=4, min=-1.308, max=-1.308, mean=-0.2193333333333333, count=6.0, positive=2, stdDev=1.0082285896010335, zeros=0}
    Implemented Feedback: [ [ -0.184, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, -1.56, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 3.144, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 
```
...[skipping 502 bytes](etc/338.txt)...
```
    [ 0.0, 0.0, 0.0, 0.0, 0.9280999999999873, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, -2.6159000000003374 ] ]
    Measured Statistics: {meanExponent=0.11838830457686951, negative=4, min=-2.6159000000003374, max=-2.6159000000003374, mean=-0.07309444444446428, count=36.0, positive=2, stdDev=0.8392837710231645, zeros=30}
    Feedback Error: [ [ 9.999999999926734E-5, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 1.0000000005261356E-4, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 1.0000000098209227E-4, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 9.99999986022182E-5, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 9.999999998722142E-5, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 9.999999966270323E-5 ] ]
    Error Statistics: {meanExponent=-4.000000000516726, negative=0, min=9.999999966270323E-5, max=9.999999966270323E-5, mean=1.6666666646836556E-5, count=36.0, positive=6, stdDev=3.726779958065502E-5, zeros=30}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 1.6667e-05 +- 3.7268e-05 [0.0000e+00 - 1.0000e-04] (36#)
    relativeTol: 6.9015e-05 +- 9.1567e-05 [1.5903e-05 - 2.7181e-04] (6#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=1.6667e-05 +- 3.7268e-05 [0.0000e+00 - 1.0000e-04] (36#), relativeTol=6.9015e-05 +- 9.1567e-05 [1.5903e-05 - 2.7181e-04] (6#)}
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
      "class": "com.simiacryptus.mindseye.layers.java.SqActivationLayer",
      "id": "282049db-f1c0-47b4-8b1a-3ec7ea0ff937",
      "isFrozen": true,
      "name": "SqActivationLayer/282049db-f1c0-47b4-8b1a-3ec7ea0ff937"
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
    	[ [ 0.276 ], [ 0.992 ], [ 0.412 ] ],
    	[ [ -1.56 ], [ 0.504 ], [ 0.404 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 0.07617600000000001 ], [ 0.9840639999999999 ], [ 0.16974399999999998 ] ],
    	[ [ 2.4336 ], [ 0.254016 ], [ 0.16321600000000003 ] ]
    ]
    --------------------
    Derivative: 
    [
    	[ [ 0.552 ], [ 1.984 ], [ 0.824 ] ],
    	[ [ -3.12 ], [ 1.008 ], [ 0.808 ] ]
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
    	[ [ -1.08 ], [ -1.508 ], [ -0.06 ], [ 1.976 ], [ -1.1 ], [ 1.924 ], [ 0.18 ], [ -1.996 ], ... ],
    	[ [ -0.604 ], [ 0.268 ], [ 0.532 ], [ -1.288 ], [ 1.356 ], [ -0.456 ], [ -1.624 ], [ 0.644 ], ... ],
    	[ [ -0.088 ], [ 1.216 ], [ 1.152 ], [ -1.86 ], [ -0.028 ], [ 1.004 ], [ 0.016 ], [ 0.256 ], ... ],
    	[ [ -1.964 ], [ -0.216 ], [ 0.032 ], [ 0.464 ], [ -1.636 ], [ -0.66 ], [ -1.532 ], [ -0.292 ], ... ],
    	[ [ 0.46 ], [ -0.876 ], [ -1.18 ], [ -0.74 ], [ 1.236 ], [ 0.232 ], [ -1.276 ], [ 1.764 ], ... ],
    	[ [ -0.904 ], [ -0.184 ], [ 0.296 ], [ 1.204 ], [ -1.212 ], [ -0.068 ], [ 0.58 ], [ 1.908 ], ... ],
    	[ [ 0.572 ], [ 0.808 ], [ 0.168 ], [ 0.12 ], [ -0.728 ], [ 0.296 ], [ 0.18 ], [ -1.584 ], ... ],
    	[ [ 1.88 ], [ -1.308 ], [ 1.844 ], [ 1.468 ], [ -0.492 ], [ -0.808 ], [ 0.112 ], [ -0.256 ], ... ],
    	...
    ]
```



First, we use a conjugate gradient descent method, which converges the fastest for purely linear functions.

Code from [LearningTester.java:225](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L225) executed in 2.53 seconds: 
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
    F(0.0) = LineSearchPoint{point=PointSample{avg=2.8229081236280837}, derivative=-0.0077301590806980345}
    New Minimum: 2.8229081236280837 > 2.822908123627291
    F(1.0E-10) = LineSearchPoint{point=PointSample{avg=2.822908123627291}, derivative=-0.007730159080695565}, delta = -7.926992395823618E-13
    New Minimum: 2.822908123627291 > 2.8229081236226627
    F(7.000000000000001E-10) = LineSearchPoint{point=PointSample{avg=2.8229081236226627}, derivative=-0.007730159080680752}, delta = -5.420996984639714E-12
    New Minimum: 2.8229081236226627 > 2.8229081235901927
    F(4.900000000000001E-9) = LineSearchPoint{point=PointSample{avg=2.8229081235901927}, derivative=-0.007730159080577064}, delta = -3.789102365203689E-11
    New Minimum: 2.8229081235901927 > 2.822908123362942
    F(3.430000000000001E-8) = LineSearchPoint{point=PointSample{avg=2.822908123362942}, derivative=-0.007730159079851242}, delta = -2.651416863841405E-10
    New Minimum: 2.822908123362942 > 2.8229081217720755
    F(2.401000000000000
```
...[skipping 285175 bytes](etc/339.txt)...
```
    LineSearchPoint{point=PointSample{avg=0.0029484521383609775}, derivative=-4.2632791615416574E-11}, delta = -1.938108710021827E-8
    F(2200.390434968384) = LineSearchPoint{point=PointSample{avg=0.002948586973765289}, derivative=1.8560069514608302E-10}, delta = 1.1545431721120616E-7
    F(169.26080268987567) = LineSearchPoint{point=PointSample{avg=0.002948459597351816}, derivative=-6.019270550503634E-11}, delta = -1.192209626160734E-8
    F(1184.8256188291298) = LineSearchPoint{point=PointSample{avg=0.0029484608807930275}, derivative=6.271615913246185E-11}, delta = -1.0638655050226181E-8
    0.0029484608807930275 <= 0.0029484715194480777
    New Minimum: 0.002948445044747584 > 0.0029484446297247755
    F(666.6265308150001) = LineSearchPoint{point=PointSample{avg=0.0029484446297247755}, derivative=4.111699921112135E-15}, delta = -2.688972330217032E-8
    Right bracket at 666.6265308150001
    Converged to right
    Iteration 250 complete. Error: 0.0029484446297247755 Total: 239727719882425.2800; Orientation: 0.0004; Line Search: 0.0096
    
```

Returns: 

```
    0.0029484446297247755
```



This training run resulted in the following regressed input:

Code from [LearningTester.java:144](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L144) executed in 0.00 seconds: 
```java
    return Arrays.stream(input_gd).map(x -> x.prettyPrint()).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [
    	[ [ -1.08 ], [ 1.508 ], [ -0.09888777247774075 ], [ 1.9760000039097743 ], [ 1.1 ], [ 1.9239999999999995 ], [ 0.18137489882579963 ], [ -1.9959157047110725 ], ... ],
    	[ [ 0.6039999999999998 ], [ -0.2680116762480211 ], [ 0.5319999999999997 ], [ -1.288 ], [ 1.356 ], [ -0.4560000000002084 ], [ -1.624 ], [ 0.6439999999999999 ], ... ],
    	[ [ -0.11095550825818348 ], [ -1.216 ], [ 1.152 ], [ 1.8600000000000003 ], [ -0.07952883864810512 ], [ 1.004 ], [ 0.08891334777942549 ], [ -0.2557158962639314 ], ... ],
    	[ [ -1.9640000000025428 ], [ -0.21626763967162435 ], [ -0.09108315073228532 ], [ 0.46400000000001007 ], [ -1.636 ], [ 0.6599999999999998 ], [ -1.532 ], [ -0.29199923606986494 ], ... ],
    	[ [ 0.46000000000005997 ], [ -0.8760000000000001 ], [ 1.18 ], [ 0.7399999999999999 ], [ 1.236 ], [ 0.23208269887814803 ], [ 1.276 ], [ -1.764 ], ... ],
    	[ [ -0.9040000000000001 ], [ -0.18520423425660373 ], [ 0.29600177184464593 ], [ 1.204 ], [ 1.212 ], [ 0.10056930398849362 ], [ -0.5799999999999997 ], [ -1.9080000000000004 ], ... ],
    	[ [ -0.5720000000000002 ], [ -0.8080000000000002 ], [ 0.1702694704450873 ], [ -0.13065040298873862 ], [ 0.7279999999999999 ], [ 0.29600151168772043 ], [ -0.18144361826073935 ], [ 1.584 ], ... ],
    	[ [ -1.8799999999999997 ], [ -1.308 ], [ 1.8439999999999999 ], [ -1.468 ], [ -0.49199999999942307 ], [ -0.8080000000000002 ], [ 0.12415003676197235 ], [ 0.2560208854134738 ], ... ],
    	...
    ]
```



Next, we run the same optimization using L-BFGS, which is nearly ideal for purely second-order or quadratic functions.

Code from [LearningTester.java:249](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L249) executed in 1.54 seconds: 
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
    th(0)=2.8229081236280837;dx=-0.0077301590806980345
    New Minimum: 2.8229081236280837 > 2.806311200532396
    WOLFE (weak): th(2.154434690031884)=2.806311200532396; dx=-0.007677103817860759 delta=0.01659692309568772
    New Minimum: 2.806311200532396 > 2.7898282953722844
    WOLFE (weak): th(4.308869380063768)=2.7898282953722844; dx=-0.007624314082830471 delta=0.03307982825579936
    New Minimum: 2.7898282953722844 > 2.725025452435209
    WOLFE (weak): th(12.926608140191302)=2.725025452435209; dx=-0.007415795718612495 delta=0.09788267119287486
    New Minimum: 2.725025452435209 > 2.4549114970260906
    END: th(51.70643256076521)=2.4549114970260906; dx=-0.0065286305880629435 delta=0.36799662660199317
    Iteration 1 complete. Error: 2.4549114970260906 Total: 239727759264443.2800; Orientation: 0.0014; Line Search: 0.0125
    LBFGS Accumulation History: 1 points
    th(0)=2.4549114970260906;dx=-0.005550090393353677
    New Minimum: 2.4549114970260906 > 1.92051872854859
```
...[skipping 132315 bytes](etc/340.txt)...
```
    
    LBFGS Accumulation History: 1 points
    th(0)=0.0029462073024068202;dx=-6.561559767248696E-12
    New Minimum: 0.0029462073024068202 > 0.0029462022547563794
    WOLFE (weak): th(777.2293544795931)=0.0029462022547563794; dx=-6.427249164182085E-12 delta=5.047650440855034E-9
    New Minimum: 0.0029462022547563794 > 0.00294619731154807
    WOLFE (weak): th(1554.4587089591862)=0.00294619731154807; dx=-6.2928046724360885E-12 delta=9.990858750117543E-9
    New Minimum: 0.00294619731154807 > 0.0029461785852327874
    END: th(4663.376126877559)=0.0029461785852327874; dx=-5.753671861589382E-12 delta=2.8717174032841242E-8
    Iteration 214 complete. Error: 0.0029461785852327874 Total: 239729268186545.7500; Orientation: 0.0005; Line Search: 0.0056
    LBFGS Accumulation History: 1 points
    th(0)=0.0029461785852327874;dx=-1.2929642065650554E-11
    MAX ALPHA: th(0)=0.0029461785852327874;th'(0)=-1.2929642065650554E-11;
    Iteration 215 failed, aborting. Error: 0.0029461785852327874 Total: 239729272367463.7200; Orientation: 0.0005; Line Search: 0.0030
    
```

Returns: 

```
    0.0029461785852327874
```



This training run resulted in the following regressed input:

Code from [LearningTester.java:154](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L154) executed in 0.00 seconds: 
```java
    return Arrays.stream(input_lbgfs).map(x -> x.prettyPrint()).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [
    	[ [ -1.08 ], [ 1.508 ], [ -0.08106486847595265 ], [ 1.9759850538878778 ], [ 1.1 ], [ 1.923990211736502 ], [ 0.18005806853122705 ], [ -1.9960002122467433 ], ... ],
    	[ [ 0.604 ], [ 0.2680000062609138 ], [ 0.532 ], [ -1.288 ], [ 1.356 ], [ -0.456 ], [ -1.624 ], [ 0.644 ], ... ],
    	[ [ -0.09718066580822432 ], [ -1.216 ], [ 1.152 ], [ 1.859999997913335 ], [ -0.06514927759805812 ], [ 1.004 ], [ 0.06841238551091121 ], [ -0.25599966570585087 ], ... ],
    	[ [ -1.964000361300488 ], [ -0.2160024695923173 ], [ 0.07129414557754844 ], [ -0.464 ], [ -1.636 ], [ 0.66 ], [ -1.532 ], [ -0.2919999999365751 ], ... ],
    	[ [ 0.46 ], [ -0.876 ], [ 1.18 ], [ 0.74 ], [ 1.236 ], [ 0.23200048836533782 ], [ 1.276 ], [ -1.7639999957443717 ], ... ],
    	[ [ -0.904 ], [ -0.18404505207148072 ], [ 0.29600000010577165 ], [ 1.204 ], [ 1.212 ], [ 0.08510132909952642 ], [ 0.58 ], [ -1.9080171102619248 ], ... ],
    	[ [ -0.572 ], [ -0.808 ], [ 0.16814610935337942 ], [ -0.12249276689844889 ], [ 0.728 ], [ 0.2960000000887934 ], [ -0.18006159081047673 ], [ 1.584 ], ... ],
    	[ [ -1.8800000021083434 ], [ -1.308 ], [ 1.8439999965216851 ], [ -1.468 ], [ -0.492 ], [ -0.808 ], [ 0.11567557323512691 ], [ 0.2560000214621164 ], ... ],
    	...
    ]
```



Code from [LearningTester.java:95](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L95) executed in 0.00 seconds: 
```java
    return TestUtil.compare(runs);
```

Returns: 

![Result](etc/test.240.png)



Code from [LearningTester.java:98](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L98) executed in 0.00 seconds: 
```java
    return TestUtil.compareTime(runs);
```

Returns: 

![Result](etc/test.241.png)



### Performance
Now we execute larger-scale runs to benchmark performance:

Code from [PerformanceTester.java:66](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/PerformanceTester.java#L66) executed in 0.14 seconds: 
```java
    test(component, inputPrototype);
```
Logging: 
```
    100 batches
    Input Dimensions:
    	[100, 100, 1]
    Performance:
    	Evaluation performance: 0.011527s +- 0.014959s [0.003248s - 0.041423s]
    	Learning performance: 0.012450s +- 0.000849s [0.011362s - 0.013829s]
    
```

### Function Plots
Code from [ActivationLayerTestBase.java:110](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/java/ActivationLayerTestBase.java#L110) executed in 0.00 seconds: 
```java
    return plot("Value Plot", plotData, x -> new double[]{x[0], x[1]});
```

Returns: 

![Result](etc/test.242.png)



Code from [ActivationLayerTestBase.java:114](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/java/ActivationLayerTestBase.java#L114) executed in 0.00 seconds: 
```java
    return plot("Derivative Plot", plotData, x -> new double[]{x[0], x[2]});
```

Returns: 

![Result](etc/test.243.png)



