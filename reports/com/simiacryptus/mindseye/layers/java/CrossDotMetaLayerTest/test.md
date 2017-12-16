# CrossDotMetaLayer
## CrossDotMetaLayerTest
Code from [SingleDerivativeTester.java:77](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/SingleDerivativeTester.java#L77) executed in 0.00 seconds: 
```java
    return test(component, inputPrototype);
```
Logging: 
```
    Inputs: [ 0.376, 1.528, -1.372 ]
    Inputs Statistics: {meanExponent=-0.03444489648731163, negative=1, min=-1.372, max=-1.372, mean=0.17733333333333326, count=3.0, positive=2, stdDev=1.1922251838008158, zeros=0}
    Output: [ [ 0.0, 0.574528, -0.515872 ], [ 0.574528, 0.0, -2.096416 ], [ -0.515872, -2.096416, 0.0 ] ]
    Outputs Statistics: {meanExponent=-0.06888979297462328, negative=4, min=0.0, max=0.0, mean=-0.45283555555555555, count=9.0, positive=2, stdDev=0.9508354462769525, zeros=3}
    Feedback for input 0
    Inputs Values: [ 0.376, 1.528, -1.372 ]
    Value Statistics: {meanExponent=-0.03444489648731163, negative=1, min=-1.372, max=-1.372, mean=0.17733333333333326, count=3.0, positive=2, stdDev=1.1922251838008158, zeros=0}
    Implemented Feedback: [ [ 0.0, 1.528, -1.372, 1.528, 0.0, 0.0, -1.372, 0.0, 0.0 ], [ 0.0, 0.376, 0.0, 0.376, 0.0, -1.372, 0.0, -1.372, 0.0 ], [ 0.0, 0.0, 0.376, 0.0, 0.0, 1.528, 0.376, 1.528, 0.0 ] ]
    Implemented Statistics: {meanExponent=-0.034444896487311624, negative=4, min=0.0, max=0.0, mean=0
```
...[skipping 342 bytes](etc/212.txt)...
```
    99999997099, 0.0, 0.0, 1.5279999999995297, 0.3759999999997099, 1.5279999999995297, 0.0 ] ]
    Measured Statistics: {meanExponent=-0.03444489648734654, negative=4, min=0.0, max=0.0, mean=0.07881481481453179, count=27.0, positive=8, stdDev=0.7996864680536965, zeros=15}
    Feedback Error: [ [ 0.0, -4.702904732312163E-13, -5.948574965941589E-13, -4.702904732312163E-13, 0.0, 0.0, -5.948574965941589E-13, 0.0, 0.0 ], [ 0.0, -2.901012763345534E-13, 0.0, -2.901012763345534E-13, 0.0, -1.7050805212193154E-12, 0.0, -1.7050805212193154E-12, 0.0 ], [ 0.0, 0.0, -2.901012763345534E-13, 0.0, 0.0, -4.702904732312163E-13, -2.901012763345534E-13, -4.702904732312163E-13, 0.0 ] ]
    Error Statistics: {meanExponent=-12.287335087715888, negative=12, min=0.0, max=0.0, mean=-2.830164086625936E-13, count=27.0, positive=0, stdDev=4.546461303895904E-13, zeros=15}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 2.8302e-13 +- 4.5465e-13 [0.0000e+00 - 1.7051e-12] (27#)
    relativeTol: 3.1958e-13 +- 1.6599e-13 [1.5389e-13 - 6.2139e-13] (12#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=2.8302e-13 +- 4.5465e-13 [0.0000e+00 - 1.7051e-12] (27#), relativeTol=3.1958e-13 +- 1.6599e-13 [1.5389e-13 - 6.2139e-13] (12#)}
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
      "class": "com.simiacryptus.mindseye.layers.java.CrossDotMetaLayer",
      "id": "e391f4e4-b8aa-4a43-b281-35c2aeb16dbd",
      "isFrozen": false,
      "name": "CrossDotMetaLayer/e391f4e4-b8aa-4a43-b281-35c2aeb16dbd"
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
    [[ -1.544, 0.064, 0.084 ]]
    --------------------
    Output: 
    [ [ 0.0, -0.098816, -0.129696 ], [ -0.098816, 0.0, 0.005376000000000001 ], [ -0.129696, 0.005376000000000001, 0.0 ] ]
    --------------------
    Derivative: 
    [ 0.29600000000000004, -2.92, -2.96 ]
```



### Input Learning
In this test, we use a network to learn this target input, given it's pre-evaluated output:

Code from [LearningTester.java:127](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L127) executed in 0.00 seconds: 
```java
    return Arrays.stream(input_target).map(x -> x.prettyPrint()).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [ -0.016, 0.692, -0.996 ]
```



First, we use a conjugate gradient descent method, which converges the fastest for purely linear functions.

Code from [LearningTester.java:225](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L225) executed in 0.17 seconds: 
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
    F(0.0) = LineSearchPoint{point=PointSample{avg=0.21286496130844443}, derivative=-0.2385489772743836}
    New Minimum: 0.21286496130844443 > 0.21286496128458948
    F(1.0E-10) = LineSearchPoint{point=PointSample{avg=0.21286496128458948}, derivative=-0.23854897725020785}, delta = -2.3854945796486504E-11
    New Minimum: 0.21286496128458948 > 0.21286496114146017
    F(7.000000000000001E-10) = LineSearchPoint{point=PointSample{avg=0.21286496114146017}, derivative=-0.23854897710515355}, delta = -1.6698425975292253E-10
    New Minimum: 0.21286496114146017 > 0.21286496013955442
    F(4.900000000000001E-9) = LineSearchPoint{point=PointSample{avg=0.21286496013955442}, derivative=-0.2385489760897734}, delta = -1.168890012559487E-9
    New Minimum: 0.21286496013955442 > 0.2128649531262146
    F(3.430000000000001E-8) = LineSearchPoint{point=PointSample{avg=0.2128649531262146}, derivative=-0.2385489689821123}, delta = -8.182229838116228E-9
    New Minimum: 0.2128649531262146 > 0.21286490403284197
    F(2.401000
```
...[skipping 296445 bytes](etc/213.txt)...
```
    7}, derivative=-7.044350305276297E-10}, delta = -4.3702469815173984E-10
    F(3.6185541799033345) = LineSearchPoint{point=PointSample{avg=5.480617187604205E-7}, derivative=9.875359301727264E-10}, delta = 1.9014391564213255E-12
    F(0.27835032153102574) = LineSearchPoint{point=PointSample{avg=5.478063850887309E-7}, derivative=-8.345691329277002E-10}, delta = -2.5343223253317676E-10
    New Minimum: 5.473642091365659E-7 > 5.471732766297894E-7
    F(1.9484522507171802) = LineSearchPoint{point=PointSample{avg=5.471732766297894E-7}, derivative=7.64221127724876E-11}, delta = -8.865406914746572E-10
    5.471732766297894E-7 <= 5.480598173212641E-7
    New Minimum: 5.471732766297894E-7 > 5.471679234814897E-7
    F(1.8083475575153785) = LineSearchPoint{point=PointSample{avg=5.471679234814897E-7}, derivative=-5.566712168303468E-15}, delta = -8.918938397743923E-10
    Left bracket at 1.8083475575153785
    Converged to left
    Iteration 250 complete. Error: 5.471679234814897E-7 Total: 239636535914716.4700; Orientation: 0.0000; Line Search: 0.0007
    
```

Returns: 

```
    5.471679234814897E-7
```



Training Converged

Next, we run the same optimization using L-BFGS, which is nearly ideal for purely second-order or quadratic functions.

Code from [LearningTester.java:249](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L249) executed in 0.10 seconds: 
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
    th(0)=0.21286496130844443;dx=-0.2385489772743836
    New Minimum: 0.21286496130844443 > 0.02396891441426463
    END: th(2.154434690031884)=0.02396891441426463; dx=-0.006209390626725427 delta=0.1888960468941798
    Iteration 1 complete. Error: 0.02396891441426463 Total: 239636540281155.4700; Orientation: 0.0001; Line Search: 0.0003
    LBFGS Accumulation History: 1 points
    th(0)=0.02396891441426463;dx=-0.015712678605880902
    Armijo: th(4.641588833612779)=0.02520061984111789; dx=0.02673955707837567 delta=-0.0012317054268532607
    New Minimum: 0.02396891441426463 > 3.343131792653765E-4
    END: th(2.3207944168063896)=3.343131792653765E-4; dx=-0.002358675283196713 delta=0.023634601234999254
    Iteration 2 complete. Error: 3.343131792653765E-4 Total: 239636540714037.4700; Orientation: 0.0000; Line Search: 0.0003
    LBFGS Accumulation History: 1 points
    th(0)=3.343131792653765E-4;dx=-3.831010069246327E-4
    Armijo: th(5.000000000000001)=0.0013394061271137056; d
```
...[skipping 148611 bytes](etc/214.txt)...
```
    > 1.6831416813323175E-7
    END: th(1.4105386889864107)=1.6831416813323175E-7; dx=-7.485938924653194E-11 delta=3.5265441849292517E-10
    Iteration 248 complete. Error: 1.6831416813323175E-7 Total: 239636636464211.3800; Orientation: 0.0000; Line Search: 0.0005
    LBFGS Accumulation History: 1 points
    th(0)=1.6831416813323175E-7;dx=-4.0418015805685664E-11
    New Minimum: 1.6831416813323175E-7 > 1.6820256453871268E-7
    END: th(3.0389134831844173)=1.6820256453871268E-7; dx=-3.303163908932187E-11 delta=1.1160359451906881E-10
    Iteration 249 complete. Error: 1.6820256453871268E-7 Total: 239636636755744.3800; Orientation: 0.0000; Line Search: 0.0002
    LBFGS Accumulation History: 1 points
    th(0)=1.6820256453871268E-7;dx=-3.993426245477332E-11
    New Minimum: 1.6820256453871268E-7 > 1.6798696639169288E-7
    END: th(6.547140628178132)=1.6798696639169288E-7; dx=-2.5925776843872648E-11 delta=2.155981470198041E-10
    Iteration 250 complete. Error: 1.6798696639169288E-7 Total: 239636636993131.3800; Orientation: 0.0000; Line Search: 0.0002
    
```

Returns: 

```
    1.6798696639169288E-7
```



Training Converged

Code from [LearningTester.java:95](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L95) executed in 0.00 seconds: 
```java
    return TestUtil.compare(runs);
```

Returns: 

![Result](etc/test.126.png)



Code from [LearningTester.java:98](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L98) executed in 0.01 seconds: 
```java
    return TestUtil.compareTime(runs);
```

Returns: 

![Result](etc/test.127.png)



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
    	Evaluation performance: 0.000029s +- 0.000009s [0.000020s - 0.000043s]
    	Learning performance: 0.000019s +- 0.000015s [0.000003s - 0.000042s]
    
```

