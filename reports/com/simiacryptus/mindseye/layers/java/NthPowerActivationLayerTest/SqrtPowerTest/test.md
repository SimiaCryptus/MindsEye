# NthPowerActivationLayer
## SqrtPowerTest
### Batch Execution
Code from [BatchingTester.java:66](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/BatchingTester.java#L66) executed in 0.00 seconds: 
```java
    return test(reference, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (120#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (58#)}
```



Code from [SingleDerivativeTester.java:77](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/SingleDerivativeTester.java#L77) executed in 0.00 seconds: 
```java
    return test(component, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ -0.028 ], [ 1.788 ], [ -1.252 ] ],
    	[ [ 1.848 ], [ -0.816 ], [ 1.704 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.13216806820914012, negative=3, min=1.704, max=1.704, mean=0.5406666666666667, count=6.0, positive=3, stdDev=1.2907288208174825, zeros=0}
    Output: [
    	[ [ 0.0 ], [ 1.3371611720357424 ], [ 0.0 ] ],
    	[ [ 1.3594116374373144 ], [ 0.0 ], [ 1.3053735097664576 ] ]
    ]
    Outputs Statistics: {meanExponent=0.12508984529577802, negative=0, min=1.3053735097664576, max=1.3053735097664576, mean=0.6669910532065857, count=6.0, positive=3, stdDev=0.6671753404783256, zeros=3}
    Feedback for input 0
    Inputs Values: [
    	[ [ -0.028 ], [ 1.788 ], [ -1.252 ] ],
    	[ [ 1.848 ], [ -0.816 ], [ 1.704 ] ]
    ]
    Value Statistics: {meanExponent=-0.13216806820914012, negative=3, min=1.704, max=1.704, mean=0.5406666666666667, count=6.0, positive=3, stdDev=1.2907288208174825, zeros=0}
    Implemented Feedback: [ [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.36780617896031226, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.3739265022471315, 0.0, 
```
...[skipping 473 bytes](etc/306.txt)...
```
    95224, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.38302651370836216 ] ]
    Measured Statistics: {meanExponent=-0.42612594724417674, negative=0, min=0.38302651370836216, max=0.38302651370836216, mean=0.031243027532878997, count=36.0, positive=3, stdDev=0.10363713284199151, zeros=33}
    Feedback Error: [ [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, -4.975599982748324E-6, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, -5.228132179235789E-6, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, -5.619439072568344E-6 ] ]
    Error Statistics: {meanExponent=-5.278371672167725, negative=3, min=-5.619439072568344E-6, max=-5.619439072568344E-6, mean=-4.3953253429312385E-7, count=36.0, positive=0, stdDev=1.4597684196942255E-6, zeros=33}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 4.3953e-07 +- 1.4598e-06 [0.0000e+00 - 5.6194e-06] (36#)
    relativeTol: 7.0301e-06 +- 2.3499e-07 [6.7639e-06 - 7.3355e-06] (3#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=4.3953e-07 +- 1.4598e-06 [0.0000e+00 - 5.6194e-06] (36#), relativeTol=7.0301e-06 +- 2.3499e-07 [6.7639e-06 - 7.3355e-06] (3#)}
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
      "id": "d6488679-b0b3-483d-91f7-a76c6eea27ee",
      "isFrozen": false,
      "name": "NthPowerActivationLayer/d6488679-b0b3-483d-91f7-a76c6eea27ee",
      "power": 0.5
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
    	[ [ -1.012 ], [ -1.948 ], [ 0.088 ] ],
    	[ [ 1.568 ], [ 1.16 ], [ 0.22 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 0.0 ], [ 0.0 ], [ 0.2966479394838265 ] ],
    	[ [ 1.2521980673998823 ], [ 1.0770329614269007 ], [ 0.469041575982343 ] ]
    ]
    --------------------
    Derivative: 
    [
    	[ [ 0.0 ], [ 0.0 ], [ 1.6854996561581053 ] ],
    	[ [ 0.3992978531249624 ], [ 0.4642383454426297 ], [ 1.0660035817780522 ] ]
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
    	[ [ -0.64 ], [ 0.632 ], [ -0.812 ], [ -0.668 ], [ 0.268 ], [ -0.068 ], [ -0.856 ], [ -0.18 ], ... ],
    	[ [ 1.472 ], [ 0.244 ], [ 1.704 ], [ -0.356 ], [ -1.672 ], [ 1.152 ], [ 1.616 ], [ -0.36 ], ... ],
    	[ [ -0.98 ], [ 1.292 ], [ 0.108 ], [ -0.96 ], [ -0.02 ], [ -0.22 ], [ -0.868 ], [ -1.0 ], ... ],
    	[ [ -0.788 ], [ 1.412 ], [ -0.908 ], [ 1.968 ], [ -0.836 ], [ -0.956 ], [ -1.62 ], [ -1.848 ], ... ],
    	[ [ -0.612 ], [ -1.728 ], [ 1.66 ], [ -1.78 ], [ 1.86 ], [ 0.54 ], [ 1.792 ], [ 0.1 ], ... ],
    	[ [ 0.336 ], [ 1.928 ], [ -1.944 ], [ -0.372 ], [ -1.196 ], [ 1.688 ], [ -0.124 ], [ 1.028 ], ... ],
    	[ [ 0.544 ], [ -0.388 ], [ -1.06 ], [ -1.42 ], [ 0.664 ], [ 0.344 ], [ 1.516 ], [ -0.084 ], ... ],
    	[ [ -1.228 ], [ -1.828 ], [ -1.656 ], [ -0.672 ], [ 1.4 ], [ 1.692 ], [ -1.732 ], [ -0.38 ], ... ],
    	...
    ]
```



First, we use a conjugate gradient descent method, which converges the fastest for purely linear functions.

Code from [LearningTester.java:225](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L225) executed in 1.12 seconds: 
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
    F(0.0) = LineSearchPoint{point=PointSample{avg=0.5610816731832546}, derivative=-6.551140922876874E-5}
    New Minimum: 0.5610816731832546 > 0.5610816731832484
    F(1.0E-10) = LineSearchPoint{point=PointSample{avg=0.5610816731832484}, derivative=-6.551140922854935E-5}, delta = -6.217248937900877E-15
    New Minimum: 0.5610816731832484 > 0.5610816731832086
    F(7.000000000000001E-10) = LineSearchPoint{point=PointSample{avg=0.5610816731832086}, derivative=-6.5511409227233E-5}, delta = -4.6074255521943996E-14
    New Minimum: 0.5610816731832086 > 0.561081673182933
    F(4.900000000000001E-9) = LineSearchPoint{point=PointSample{avg=0.561081673182933}, derivative=-6.551140921801854E-5}, delta = -3.2163161023390785E-13
    New Minimum: 0.561081673182933 > 0.5610816731810081
    F(3.430000000000001E-8) = LineSearchPoint{point=PointSample{avg=0.5610816731810081}, derivative=-6.551140915351732E-5}, delta = -2.2465362903290043E-12
    New Minimum: 0.5610816731810081 > 0.5610816731675237
    F(2.401000000000
```
...[skipping 93750 bytes](etc/307.txt)...
```
    02
    New Minimum: 0.2641892000000001 > 0.26418920000000007
    F(12893.393937389548) = LineSearchPoint{point=PointSample{avg=0.26418920000000007}, derivative=-4.4271097995586855E-27}, delta = -1.1102230246251565E-16
    Left bracket at 12893.393937389548
    Converged to left
    Iteration 58 complete. Error: 0.26418920000000007 Total: 239701153849538.8400; Orientation: 0.0003; Line Search: 0.0093
    Zero gradient: 7.800012263318808E-11
    F(0.0) = LineSearchPoint{point=PointSample{avg=0.26418920000000007}, derivative=-6.084019130792378E-21}
    F(12893.393937389548) = LineSearchPoint{point=PointSample{avg=0.26418920000000007}, derivative=5.865008093944579E-21}, delta = 0.0
    0.26418920000000007 <= 0.26418920000000007
    F(6564.857029827992) = LineSearchPoint{point=PointSample{avg=0.26418920000000007}, derivative=5.515137725040429E-27}, delta = 0.0
    Right bracket at 6564.857029827992
    Converged to right
    Iteration 59 failed, aborting. Error: 0.26418920000000007 Total: 239701160428834.8400; Orientation: 0.0004; Line Search: 0.0051
    
```

Returns: 

```
    0.26418920000000007
```



This training run resulted in the following regressed input:

Code from [LearningTester.java:144](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L144) executed in 0.00 seconds: 
```java
    return Arrays.stream(input_gd).map(x -> x.prettyPrint()).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [
    	[ [ -1.0104428851926788 ], [ 0.6320000000000001 ], [ -1.664 ], [ -1.2344428851926785 ], [ -1.744 ], [ -0.13844288519267867 ], [ -0.13844288519267867 ], [ -0.46644288519267874 ], ... ],
    	[ [ -1.0 ], [ -0.6565223849279084 ], [ -0.22 ], [ -1.54 ], [ -0.884 ], [ 1.1520000000000004 ], [ -1.924 ], [ -0.316 ], ... ],
    	[ [ -0.3132103489467689 ], [ -1.568 ], [ -1.376 ], [ -0.8504428851926785 ], [ -0.4504428851926787 ], [ -1.54 ], [ -1.376 ], [ -0.3092103489467691 ], ... ],
    	[ [ -0.26521034894676904 ], [ -1.584 ], [ -1.208 ], [ -1.916 ], [ -0.272 ], [ -0.722442885192679 ], [ -1.0704428851926786 ], [ -0.29321034894676884 ], ... ],
    	[ [ -0.3732103489467687 ], [ -0.376 ], [ 1.6599999998856072 ], [ -1.676 ], [ 1.8600000012907145 ], [ -0.424 ], [ -1.908 ], [ -0.33683057003712724 ], ... ],
    	[ [ -0.11818678822110074 ], [ 1.9279999996537054 ], [ -1.1184428851926786 ], [ -1.012 ], [ -0.11444288519267865 ], [ 1.688000000042311 ], [ -1.428 ], [ -0.476 ], ... ],
    	[ [ 0.544 ], [ -0.9064428851926787 ], [ -1.692 ], [ -0.5344428851926788 ], [ -1.46 ], [ -1.0732352209810476 ], [ 1.515999999998044 ], [ -0.016 ], ... ],
    	[ [ -1.872 ], [ -0.432 ], [ -1.4224428851926785 ], [ -0.252 ], [ -1.668 ], [ -1.232 ], [ -1.248 ], [ -0.616 ], ... ],
    	...
    ]
```



Next, we run the same optimization using L-BFGS, which is nearly ideal for purely second-order or quadratic functions.

Code from [LearningTester.java:249](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L249) executed in 0.07 seconds: 
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
    th(0)=0.5610816731832546;dx=-6.551140922876874E-5
    New Minimum: 0.5610816731832546 > 0.560944391053986
    WOLFE (weak): th(2.154434690031884)=0.560944391053986; dx=-6.231345934190313E-5 delta=1.3728212926866767E-4
    New Minimum: 0.560944391053986 > 0.5608122824883698
    WOLFE (weak): th(4.308869380063768)=0.5608122824883698; dx=-6.044510460489932E-5 delta=2.6939069488485234E-4
    New Minimum: 0.5608122824883698 > 0.5603113209663446
    END: th(12.926608140191302)=0.5603113209663446; dx=-5.633299082873258E-5 delta=7.703522169100196E-4
    Iteration 1 complete. Error: 0.5603113209663446 Total: 239701193881022.8400; Orientation: 0.0010; Line Search: 0.0097
    LBFGS Accumulation History: 1 points
    th(0)=0.5603113209663446;dx=-5.1339316157333336E-5
    New Minimum: 0.5603113209663446 > 0.558929538509686
    WOLFE (weak): th(27.849533001676672)=0.558929538509686; dx=-4.812068083464882E-5 delta=0.0013817824566586534
    New Minimum: 0.558929538509686 > 0.5576184
```
...[skipping 1221 bytes](etc/308.txt)...
```
    0.0005; Line Search: 0.0061
    LBFGS Accumulation History: 1 points
    th(0)=0.514280098199108;dx=-2.869128529464828E-5
    New Minimum: 0.514280098199108 > 0.4548399862170738
    END: th(2227.962640134134)=0.4548399862170738; dx=-2.4807640574643847E-5 delta=0.05944011198203425
    Iteration 5 complete. Error: 0.4548399862170738 Total: 239701224984704.7800; Orientation: 0.0006; Line Search: 0.0044
    LBFGS Accumulation History: 1 points
    th(0)=0.4548399862170738;dx=-2.3998515465123953E-5
    New Minimum: 0.4548399862170738 > 0.3567258887744459
    END: th(4800.0)=0.3567258887744459; dx=-1.70434527143155E-5 delta=0.09811409744262789
    Iteration 6 complete. Error: 0.3567258887744459 Total: 239701231156766.7800; Orientation: 0.0005; Line Search: 0.0041
    LBFGS Accumulation History: 1 points
    th(0)=0.3567258887744459;dx=-1.6633400143865936E-5
    MAX ALPHA: th(0)=0.3567258887744459;th'(0)=-1.6633400143865936E-5;
    Iteration 7 failed, aborting. Error: 0.3567258887744459 Total: 239701236513220.7800; Orientation: 0.0005; Line Search: 0.0036
    
```

Returns: 

```
    0.3567258887744459
```



This training run resulted in the following regressed input:

Code from [LearningTester.java:154](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L154) executed in 0.00 seconds: 
```java
    return Arrays.stream(input_lbgfs).map(x -> x.prettyPrint()).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [
    	[ [ -0.1410716965492984 ], [ 0.705734349058125 ], [ -1.664 ], [ -0.3650716965492984 ], [ -1.744 ], [ 0.7309283034507017 ], [ 0.7309283034507017 ], [ 0.4029283034507015 ], ... ],
    	[ [ -1.0 ], [ 0.8976754642218328 ], [ -0.22 ], [ -1.54 ], [ -0.884 ], [ 0.985364289438018 ], [ -1.924 ], [ -0.316 ], ... ],
    	[ [ 1.0509283034507015 ], [ -1.568 ], [ -1.376 ], [ 0.01892830345070151 ], [ 0.4189283034507016 ], [ -1.54 ], [ -1.376 ], [ 1.0549283034507015 ], ... ],
    	[ [ 1.0989283034507016 ], [ -1.584 ], [ -1.208 ], [ -1.916 ], [ -0.272 ], [ 0.14692830345070163 ], [ -0.2010716965492983 ], [ 1.0709283034507013 ], ... ],
    	[ [ 0.9909283034507017 ], [ -0.376 ], [ 0.958507326699601 ], [ -1.676 ], [ 1.226829737033112 ], [ -0.424 ], [ -1.908 ], [ 0.25324707266381474 ], ... ],
    	[ [ 1.4181915954650635 ], [ 1.9599651171431254 ], [ -0.24907169654929837 ], [ -1.012 ], [ 0.7549283034507017 ], [ 1.1834640993652035 ], [ -1.428 ], [ -0.476 ], ... ],
    	[ [ 0.6647917531822368 ], [ -0.037071696549298316 ], [ -1.692 ], [ 0.3349283034507015 ], [ -1.46 ], [ 1.088944897365709 ], [ 1.064961022848793 ], [ -0.016 ], ... ],
    	[ [ -1.872 ], [ -0.432 ], [ -0.07307169654929835 ], [ -0.252 ], [ -1.668 ], [ -1.232 ], [ -1.248 ], [ -0.616 ], ... ],
    	...
    ]
```



Code from [LearningTester.java:95](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L95) executed in 0.00 seconds: 
```java
    return TestUtil.compare(runs);
```

Returns: 

![Result](etc/test.213.png)



Code from [LearningTester.java:98](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L98) executed in 0.00 seconds: 
```java
    return TestUtil.compareTime(runs);
```

Returns: 

![Result](etc/test.214.png)



### Performance
Now we execute larger-scale runs to benchmark performance:

Code from [PerformanceTester.java:66](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/PerformanceTester.java#L66) executed in 0.24 seconds: 
```java
    test(component, inputPrototype);
```
Logging: 
```
    100 batches
    Input Dimensions:
    	[100, 100, 1]
    Performance:
    	Evaluation performance: 0.013005s +- 0.001471s [0.011627s - 0.015025s]
    	Learning performance: 0.022782s +- 0.018541s [0.011152s - 0.059623s]
    
```

### Function Plots
Code from [ActivationLayerTestBase.java:110](../../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/java/ActivationLayerTestBase.java#L110) executed in 0.00 seconds: 
```java
    return plot("Value Plot", plotData, x -> new double[]{x[0], x[1]});
```

Returns: 

![Result](etc/test.215.png)



Code from [ActivationLayerTestBase.java:114](../../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/java/ActivationLayerTestBase.java#L114) executed in 0.00 seconds: 
```java
    return plot("Derivative Plot", plotData, x -> new double[]{x[0], x[2]});
```

Returns: 

![Result](etc/test.216.png)



