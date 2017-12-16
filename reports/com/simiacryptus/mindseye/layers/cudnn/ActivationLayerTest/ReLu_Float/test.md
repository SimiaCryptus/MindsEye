# ActivationLayer
## ReLu_Float
### Batch Execution
Code from [BatchingTester.java:66](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/BatchingTester.java#L66) executed in 0.01 seconds: 
```java
    return test(reference, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (20#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (6#)}
```



Code from [SingleDerivativeTester.java:77](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/SingleDerivativeTester.java#L77) executed in 0.00 seconds: 
```java
    return test(component, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ -1.904 ] ]
    ]
    Inputs Statistics: {meanExponent=0.27966694404845555, negative=1, min=-1.904, max=-1.904, mean=-1.904, count=1.0, positive=0, stdDev=0.0, zeros=0}
    Output: [
    	[ [ 0.0 ] ]
    ]
    Outputs Statistics: {meanExponent=NaN, negative=0, min=0.0, max=0.0, mean=0.0, count=1.0, positive=0, stdDev=0.0, zeros=1}
    Feedback for input 0
    Inputs Values: [
    	[ [ -1.904 ] ]
    ]
    Value Statistics: {meanExponent=0.27966694404845555, negative=1, min=-1.904, max=-1.904, mean=-1.904, count=1.0, positive=0, stdDev=0.0, zeros=0}
    Implemented Feedback: [ [ 0.0 ] ]
    Implemented Statistics: {meanExponent=NaN, negative=0, min=0.0, max=0.0, mean=0.0, count=1.0, positive=0, stdDev=0.0, zeros=1}
    Measured Feedback: [ [ 0.0 ] ]
    Measured Statistics: {meanExponent=NaN, negative=0, min=0.0, max=0.0, mean=0.0, count=1.0, positive=0, stdDev=0.0, zeros=1}
    Feedback Error: [ [ 0.0 ] ]
    Error Statistics: {meanExponent=NaN, negative=0, min=0.0, max=0.0, mean=0.0, count=1.0, positive=0, stdDev=0.0, zeros=1}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (1#)
    relativeTol: 0.0000e+00 +- 0.0000e+00 [Infinity - -Infinity] (0#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (1#), relativeTol=0.0000e+00 +- 0.0000e+00 [Infinity - -Infinity] (0#)}
```



### Reference Implementation
Code from [EquivalencyTester.java:61](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/EquivalencyTester.java#L61) executed in 0.00 seconds: 
```java
    System.out.println(new GsonBuilder().setPrettyPrinting().create().toJson(this.reference.getJson()));
```
Logging: 
```
    {
      "class": "com.simiacryptus.mindseye.layers.java.ReLuActivationLayer",
      "id": "d998ee0e-96f4-454f-b009-c06619ad9125",
      "isFrozen": true,
      "name": "ReLuActivationLayer/d998ee0e-96f4-454f-b009-c06619ad9125",
      "weights": [
        1.0
      ]
    }
    
```

Code from [EquivalencyTester.java:64](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/EquivalencyTester.java#L64) executed in 0.00 seconds: 
```java
    return test(subject, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ 1.324 ] ]
    ]
    Error: [
    	[ [ 0.0 ] ]
    ]
    Accuracy:
    absoluteTol: 0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (1#)
    relativeTol: 0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (1#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (1#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (1#)}
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
      "class": "com.simiacryptus.mindseye.layers.cudnn.ActivationLayer",
      "id": "a3941a59-07d3-40cb-a6af-a32876b7039e",
      "isFrozen": false,
      "name": "ActivationLayer/a3941a59-07d3-40cb-a6af-a32876b7039e",
      "mode": 1
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
    	[ [ 0.54 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 0.54 ] ]
    ]
    --------------------
    Derivative: 
    [
    	[ [ 1.0 ] ]
    ]
```



[GPU Log](etc/cuda.log)

### Input Learning
In this test, we use a network to learn this target input, given it's pre-evaluated output:

Code from [LearningTester.java:127](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L127) executed in 0.00 seconds: 
```java
    return Arrays.stream(input_target).map(x -> x.prettyPrint()).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [
    	[ [ 1.892 ], [ 0.816 ], [ 0.468 ], [ 1.368 ], [ -0.228 ], [ -0.548 ], [ 0.564 ], [ 1.304 ], ... ],
    	[ [ 0.944 ], [ 0.124 ], [ -1.98 ], [ 1.996 ], [ 0.772 ], [ -1.112 ], [ -1.828 ], [ 0.256 ], ... ],
    	[ [ 1.564 ], [ 0.6 ], [ 1.532 ], [ 1.896 ], [ 1.272 ], [ 0.516 ], [ 1.032 ], [ 0.224 ], ... ],
    	[ [ 1.508 ], [ 1.432 ], [ 1.22 ], [ 1.488 ], [ -1.444 ], [ 0.308 ], [ 0.6 ], [ -0.532 ], ... ],
    	[ [ -0.92 ], [ -0.312 ], [ -0.056 ], [ -1.348 ], [ 0.964 ], [ -1.796 ], [ -1.216 ], [ 0.968 ], ... ],
    	[ [ 0.696 ], [ 0.356 ], [ -1.268 ], [ 0.08 ], [ 1.84 ], [ -0.552 ], [ -0.88 ], [ 1.868 ], ... ],
    	[ [ -0.648 ], [ 1.008 ], [ 0.848 ], [ 0.492 ], [ 0.736 ], [ 1.296 ], [ -1.788 ], [ -0.42 ], ... ],
    	[ [ -1.648 ], [ -0.096 ], [ 1.276 ], [ 0.688 ], [ 1.072 ], [ -1.6 ], [ -1.908 ], [ -1.844 ], ... ],
    	...
    ]
```



First, we use a conjugate gradient descent method, which converges the fastest for purely linear functions.

Code from [LearningTester.java:225](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L225) executed in 0.04 seconds: 
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
    F(0.0) = LineSearchPoint{point=PointSample{avg=0.827618303999995}, derivative=-1.9882720576000002E-4}
    New Minimum: 0.827618303999995 > 0.8276183039999742
    F(1.0E-10) = LineSearchPoint{point=PointSample{avg=0.8276183039999742}, derivative=-1.9882720575999604E-4}, delta = -2.0872192862952943E-14
    New Minimum: 0.8276183039999742 > 0.827618303999859
    F(7.000000000000001E-10) = LineSearchPoint{point=PointSample{avg=0.827618303999859}, derivative=-1.9882720575997218E-4}, delta = -1.3600232051658168E-13
    New Minimum: 0.827618303999859 > 0.827618303999023
    F(4.900000000000001E-9) = LineSearchPoint{point=PointSample{avg=0.827618303999023}, derivative=-1.9882720575980516E-4}, delta = -9.720002580593246E-13
    New Minimum: 0.827618303999023 > 0.8276183039931748
    F(3.430000000000001E-8) = LineSearchPoint{point=PointSample{avg=0.8276183039931748}, derivative=-1.9882720575863607E-4}, delta = -6.820211062574799E-12
    New Minimum: 0.8276183039931748 > 0.8276183039522597
    F(2.40100000000
```
...[skipping 1599 bytes](etc/23.txt)...
```
    .8275789901422579 > 0.8273431396457241
    F(1.3841287201) = LineSearchPoint{point=PointSample{avg=0.8273431396457241}, derivative=-1.987721652708341E-4}, delta = -2.751643542708848E-4
    Loops = 12
    New Minimum: 0.8273431396457241 > 0.33055028960000055
    F(5000.000000000922) = LineSearchPoint{point=PointSample{avg=0.33055028960000055}, derivative=1.2300595348402334E-17}, delta = -0.4970680143999945
    Right bracket at 5000.000000000922
    Converged to right
    Iteration 1 complete. Error: 0.33055028960000055 Total: 239435277807709.7500; Orientation: 0.0004; Line Search: 0.0300
    Zero gradient: 1.5064422145823633E-15
    F(0.0) = LineSearchPoint{point=PointSample{avg=0.33055028960000055}, derivative=-2.2693681458758152E-30}
    F(5000.000000000922) = LineSearchPoint{point=PointSample{avg=0.33055028960000055}, derivative=0.0}, delta = 0.0
    0.33055028960000055 <= 0.33055028960000055
    Converged to right
    Iteration 2 failed, aborting. Error: 0.33055028960000055 Total: 239435282243398.7200; Orientation: 0.0003; Line Search: 0.0029
    
```

Returns: 

```
    0.33055028960000055
```



This training run resulted in the following regressed input:

Code from [LearningTester.java:144](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L144) executed in 0.07 seconds: 
```java
    return Arrays.stream(input_gd).map(x -> x.prettyPrint()).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [
    	[ [ 1.892 ], [ -0.684 ], [ -0.076 ], [ 1.368 ], [ -2.9820590441431705E-13 ], [ -1.376 ], [ 0.564 ], [ 1.304 ], ... ],
    	[ [ -0.06 ], [ 0.124 ], [ -3.623767952376511E-13 ], [ -1.02 ], [ -1.72 ], [ -1.64 ], [ -1.3578027591165664E-13 ], [ 0.256 ], ... ],
    	[ [ 1.564 ], [ -0.408 ], [ -1.112 ], [ 1.896 ], [ -0.716 ], [ 0.516 ], [ -0.12 ], [ -0.636 ], ... ],
    	[ [ 1.508 ], [ 1.432 ], [ -0.084 ], [ -0.404 ], [ -0.952 ], [ 0.308 ], [ -1.0 ], [ -0.564 ], ... ],
    	[ [ -0.444 ], [ -9.892087149410145E-14 ], [ -1.624256285026604E-13 ], [ -1.708 ], [ -0.224 ], [ -0.772 ], [ -1.248 ], [ -1.288 ], ... ],
    	[ [ -0.708 ], [ -1.624 ], [ -1.653122083666858E-13 ], [ 0.08 ], [ 1.84 ], [ -1.556 ], [ -0.904 ], [ -0.812 ], ... ],
    	[ [ -0.672 ], [ -1.332 ], [ -1.228 ], [ -0.86 ], [ 0.736 ], [ -1.972 ], [ -0.904 ], [ -1.1 ], ... ],
    	[ [ -1.88 ], [ -1.504 ], [ -1.772 ], [ -0.292 ], [ 1.072 ], [ -3.2174263253637037E-13 ], [ -3.3573144264664734E-13 ], [ -0.508 ], ... ],
    	...
    ]
```



Next, we run the same optimization using L-BFGS, which is nearly ideal for purely second-order or quadratic functions.

Code from [LearningTester.java:249](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L249) executed in 0.16 seconds: 
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
    th(0)=0.827618303999995;dx=-1.9882720576000002E-4
    New Minimum: 0.827618303999995 > 0.8271900360580033
    WOLFE (weak): th(2.154434690031884)=0.8271900360580033; dx=-1.9874153371411774E-4 delta=4.2826794199168905E-4
    New Minimum: 0.8271900360580033 > 0.8267619526908239
    WOLFE (weak): th(4.308869380063768)=0.8267619526908239; dx=-1.9865586166823544E-4 delta=8.563513091711661E-4
    New Minimum: 0.8267619526908239 > 0.8250514649704243
    WOLFE (weak): th(12.926608140191302)=0.8250514649704243; dx=-1.9831317348470627E-4 delta=0.0025668390295707777
    New Minimum: 0.8250514649704243 > 0.8173908160444752
    WOLFE (weak): th(51.70643256076521)=0.8173908160444752; dx=-1.9677107665882506E-4 delta=0.010227487955519865
    New Minimum: 0.8173908160444752 > 0.7775440152294502
    WOLFE (weak): th(258.53216280382605)=0.7775440152294502; dx=-1.8854656025412518E-4 delta=0.05007428877054487
    New Minimum: 0.7775440152294502 > 0.5670407341415123
    END: th(1551.19297
```
...[skipping 2426 bytes](etc/24.txt)...
```
    002)=0.33055031738773427; dx=-1.1115093664644945E-10 delta=2.7509856827534485E-6
    Iteration 6 complete. Error: 0.33055031738773427 Total: 239435493854671.5300; Orientation: 0.0010; Line Search: 0.0135
    LBFGS Accumulation History: 1 points
    th(0)=0.33055031738773427;dx=-1.1115093664644793E-11
    New Minimum: 0.33055031738773427 > 0.33055029781576106
    WOLF (strong): th(9694.956105143481)=0.33055029781576106; dx=3.499824642531511E-12 delta=1.957197320878734E-8
    New Minimum: 0.33055029781576106 > 0.3305502896258571
    END: th(4847.478052571741)=0.3305502896258571; dx=-3.390591463158328E-13 delta=2.7761877186005535E-8
    Iteration 7 complete. Error: 0.3305502896258571 Total: 239435509493430.5000; Orientation: 0.0012; Line Search: 0.0106
    LBFGS Accumulation History: 1 points
    th(0)=0.3305502896258571;dx=-1.0342792257890987E-14
    MAX ALPHA: th(0)=0.3305502896258571;th'(0)=-1.0342792257890987E-14;
    Iteration 8 failed, aborting. Error: 0.3305502896258571 Total: 239435518967822.5000; Orientation: 0.0006; Line Search: 0.0066
    
```

Returns: 

```
    0.3305502896258571
```



This training run resulted in the following regressed input:

Code from [LearningTester.java:154](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L154) executed in 0.01 seconds: 
```java
    return Arrays.stream(input_lbgfs).map(x -> x.prettyPrint()).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [
    	[ [ 1.8919879696837931 ], [ -0.684 ], [ -0.076 ], [ 1.367990768102671 ], [ 1.1655270377889412E-5 ], [ -1.376 ], [ 0.5640081933088794 ], [ 1.3039995384051335 ], ... ],
    	[ [ -0.06 ], [ 0.12400014424839577 ], [ 1.4165192464216992E-5 ], [ -1.02 ], [ -1.72 ], [ -1.64 ], [ 5.308340964187234E-6 ], [ 0.2560023945233697 ], ... ],
    	[ [ 1.564002279124653 ], [ -0.408 ], [ -1.112 ], [ 1.8959878831347556 ], [ -0.716 ], [ 0.5160008366406954 ], [ -0.12 ], [ -0.636 ], ... ],
    	[ [ 1.507993566521549 ], [ 1.432003202314386 ], [ -0.084 ], [ -0.404 ], [ -0.952 ], [ 0.3080081933088795 ], [ -1.0 ], [ -0.564 ], ... ],
    	[ [ -0.444 ], [ 3.865857006527651E-6 ], [ 6.346929413702124E-6 ], [ -1.708 ], [ -0.224 ], [ -0.772 ], [ -1.248 ], [ -1.288 ], ... ],
    	[ [ -0.708 ], [ -1.624 ], [ 6.462328130314878E-6 ], [ 0.08000447170026875 ], [ 1.8399921817369496 ], [ -1.556 ], [ -0.904 ], [ -0.812 ], ... ],
    	[ [ -0.672 ], [ -1.332 ], [ -1.228 ], [ -0.86 ], [ 0.735999423006417 ], [ -1.972 ], [ -0.904 ], [ -1.1 ], ... ],
    	[ [ -1.88 ], [ -1.504 ], [ -1.772 ], [ -0.292 ], [ 1.0719993076077003 ], [ 1.257846011079155E-5 ], [ 1.3126604014702156E-5 ], [ -0.508 ], ... ],
    	...
    ]
```



Code from [LearningTester.java:95](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L95) executed in 0.01 seconds: 
```java
    return TestUtil.compare(runs);
```

Returns: 

![Result](etc/test.17.png)



Code from [LearningTester.java:98](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L98) executed in 0.00 seconds: 
```java
    return TestUtil.compareTime(runs);
```

Returns: 

![Result](etc/test.18.png)



### Performance
Now we execute larger-scale runs to benchmark performance:

Code from [PerformanceTester.java:66](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/PerformanceTester.java#L66) executed in 0.25 seconds: 
```java
    test(component, inputPrototype);
```
Logging: 
```
    100 batches
    Input Dimensions:
    	[100, 100, 1]
    Performance:
    	Evaluation performance: 0.011895s +- 0.001914s [0.009277s - 0.014378s]
    	Learning performance: 0.028226s +- 0.002583s [0.025364s - 0.032040s]
    
```

### Function Plots
Code from [ActivationLayerTest.java:90](../../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/cudnn/ActivationLayerTest.java#L90) executed in 0.05 seconds: 
```java
    return ActivationLayerTestBase.plot("Value Plot", plotData, x -> new double[]{x[0], x[1]});
```

Returns: 

![Result](etc/test.19.png)



Code from [ActivationLayerTest.java:94](../../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/cudnn/ActivationLayerTest.java#L94) executed in 0.00 seconds: 
```java
    return ActivationLayerTestBase.plot("Derivative Plot", plotData, x -> new double[]{x[0], x[2]});
```

Returns: 

![Result](etc/test.20.png)



