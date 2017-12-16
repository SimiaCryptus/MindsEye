# MaxDropoutNoiseLayer
## MaxDropoutNoiseLayerTest
### Batch Execution
Code from [BatchingTester.java:66](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/BatchingTester.java#L66) executed in 0.00 seconds: 
```java
    return test(reference, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (320#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (20#)}
```



Code from [SingleDerivativeTester.java:77](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/SingleDerivativeTester.java#L77) executed in 0.00 seconds: 
```java
    return test(component, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ 0.9 ], [ -1.204 ], [ -0.684 ], [ 1.316 ] ],
    	[ [ -0.276 ], [ -1.772 ], [ -1.0 ], [ 0.8 ] ],
    	[ [ -0.416 ], [ -0.928 ], [ -1.172 ], [ 0.4 ] ],
    	[ [ 1.312 ], [ -1.64 ], [ 0.744 ], [ 1.124 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.056600649144420236, negative=9, min=1.124, max=1.124, mean=-0.15600000000000003, count=16.0, positive=7, stdDev=1.0524314704530646, zeros=0}
    Output: [
    	[ [ 0.0 ], [ -0.0 ], [ -0.0 ], [ 0.0 ] ],
    	[ [ -0.0 ], [ -0.0 ], [ -0.0 ], [ 0.0 ] ],
    	[ [ -0.0 ], [ -0.0 ], [ -0.0 ], [ 0.0 ] ],
    	[ [ 0.0 ], [ -0.0 ], [ 0.0 ], [ 1.124 ] ]
    ]
    Outputs Statistics: {meanExponent=0.05076631123304232, negative=0, min=1.124, max=1.124, mean=0.07025, count=16.0, positive=1, stdDev=0.27207708007107106, zeros=15}
    Feedback for input 0
    Inputs Values: [
    	[ [ 0.9 ], [ -1.204 ], [ -0.684 ], [ 1.316 ] ],
    	[ [ -0.276 ], [ -1.772 ], [ -1.0 ], [ 0.8 ] ],
    	[ [ -0.416 ], [ -0.928 ], [ -1.172 ], [ 0.4 ] ],
    	[ [ 1.312 ], [ -1.64 ], [ 0.744 ], [ 1.124 ] ]
    ]
    Value Statistics: {meanExponent=-0.0566006491444
```
...[skipping 1133 bytes](etc/279.txt)...
```
    ed Statistics: {meanExponent=-4.7830642341045674E-14, negative=0, min=0.9999999999998899, max=0.9999999999998899, mean=0.00390624999999957, count=256.0, positive=1, stdDev=0.06237781024480294, zeros=255}
    Feedback Error: [ [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], ... ]
    Error Statistics: {meanExponent=-12.958078098036825, negative=1, min=-1.1013412404281553E-13, max=-1.1013412404281553E-13, mean=-4.3021142204224816E-16, count=256.0, positive=0, stdDev=6.869925491021093E-15, zeros=255}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 4.3021e-16 +- 6.8699e-15 [0.0000e+00 - 1.1013e-13] (256#)
    relativeTol: 5.5067e-14 +- 0.0000e+00 [5.5067e-14 - 5.5067e-14] (1#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=4.3021e-16 +- 6.8699e-15 [0.0000e+00 - 1.1013e-13] (256#), relativeTol=5.5067e-14 +- 0.0000e+00 [5.5067e-14 - 5.5067e-14] (1#)}
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
      "class": "com.simiacryptus.mindseye.layers.java.MaxDropoutNoiseLayer",
      "id": "e47a2248-7cf3-4f3a-88b2-bf689a898857",
      "isFrozen": false,
      "name": "MaxDropoutNoiseLayer/e47a2248-7cf3-4f3a-88b2-bf689a898857",
      "kernelSize": [
        2,
        2,
        1
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
    [[
    	[ [ -0.436 ], [ -1.592 ], [ -1.12 ], [ -0.22 ] ],
    	[ [ 0.36 ], [ -0.948 ], [ -0.628 ], [ 0.68 ] ],
    	[ [ 0.82 ], [ 0.072 ], [ -0.652 ], [ -0.68 ] ],
    	[ [ -1.54 ], [ -1.908 ], [ 0.988 ], [ 0.416 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ -0.0 ], [ -0.0 ], [ -0.0 ], [ -0.0 ] ],
    	[ [ 0.0 ], [ -0.0 ], [ -0.0 ], [ 0.0 ] ],
    	[ [ 0.0 ], [ 0.0 ], [ -0.0 ], [ -0.0 ] ],
    	[ [ -0.0 ], [ -0.0 ], [ 0.0 ], [ 0.416 ] ]
    ]
    --------------------
    Derivative: 
    [
    	[ [ 0.0 ], [ 0.0 ], [ 0.0 ], [ 0.0 ] ],
    	[ [ 0.0 ], [ 0.0 ], [ 0.0 ], [ 0.0 ] ],
    	[ [ 0.0 ], [ 0.0 ], [ 0.0 ], [ 0.0 ] ],
    	[ [ 0.0 ], [ 0.0 ], [ 0.0 ], [ 1.0 ] ]
    ]
```



### Input Learning
In this test, we use a network to learn this target input, given it's pre-evaluated output:

Code from [LearningTester.java:127](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L127) executed in 0.01 seconds: 
```java
    return Arrays.stream(input_target).map(x -> x.prettyPrint()).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [
    	[ [ -0.704 ], [ -1.56 ], [ -1.08 ], [ -1.772 ], [ 0.212 ], [ 0.14 ], [ 1.552 ], [ -1.952 ], ... ],
    	[ [ 0.108 ], [ -1.632 ], [ 0.66 ], [ 1.392 ], [ -1.764 ], [ -1.336 ], [ -1.776 ], [ -0.136 ], ... ],
    	[ [ -0.02 ], [ 1.484 ], [ 1.448 ], [ 1.02 ], [ -1.712 ], [ 1.748 ], [ 0.612 ], [ 1.66 ], ... ],
    	[ [ 0.364 ], [ 1.78 ], [ -0.104 ], [ 0.604 ], [ 0.904 ], [ 0.984 ], [ -0.84 ], [ -1.076 ], ... ],
    	[ [ 0.584 ], [ -1.5 ], [ 0.516 ], [ 1.452 ], [ 0.572 ], [ -1.044 ], [ 0.408 ], [ 0.516 ], ... ],
    	[ [ 1.26 ], [ -1.272 ], [ -0.48 ], [ -0.256 ], [ 1.604 ], [ 0.008 ], [ 1.7 ], [ 0.04 ], ... ],
    	[ [ -1.98 ], [ 1.792 ], [ 1.028 ], [ -1.792 ], [ -1.844 ], [ 1.396 ], [ -1.788 ], [ 0.788 ], ... ],
    	[ [ -1.944 ], [ 1.516 ], [ -0.748 ], [ -1.464 ], [ 1.184 ], [ 1.272 ], [ 0.22 ], [ -1.22 ], ... ],
    	...
    ]
```



First, we use a conjugate gradient descent method, which converges the fastest for purely linear functions.

Code from [LearningTester.java:225](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L225) executed in 0.05 seconds: 
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
    F(0.0) = LineSearchPoint{point=PointSample{avg=7.744000000000014E-7}, derivative=-3.0976000000000064E-10}
    New Minimum: 7.744000000000014E-7 > 7.743999999999701E-7
    F(1.0E-10) = LineSearchPoint{point=PointSample{avg=7.743999999999701E-7}, derivative=-3.097599999999944E-10}, delta = -3.134021904840911E-20
    New Minimum: 7.743999999999701E-7 > 7.743999999997845E-7
    F(7.000000000000001E-10) = LineSearchPoint{point=PointSample{avg=7.743999999997845E-7}, derivative=-3.097599999999572E-10}, delta = -2.1694631361550767E-19
    New Minimum: 7.743999999997845E-7 > 7.743999999984831E-7
    F(4.900000000000001E-9) = LineSearchPoint{point=PointSample{avg=7.743999999984831E-7}, derivative=-3.097599999996969E-10}, delta = -1.5183065579533334E-18
    New Minimum: 7.743999999984831E-7 > 7.743999999893775E-7
    F(3.430000000000001E-8) = LineSearchPoint{point=PointSample{avg=7.743999999893775E-7}, derivative=-3.097599999978759E-10}, delta = -1.0623910740937062E-17
    New Minimum: 7.743999999893775E-7
```
...[skipping 1718 bytes](etc/280.txt)...
```
    971311631866E-7
    F(1.3841287201) = LineSearchPoint{point=PointSample{avg=7.73971311631866E-7}, derivative=-3.0967425045753307E-10}, delta = -4.2868836813546425E-10
    Loops = 12
    New Minimum: 7.73971311631866E-7 > 4.378375239202921E-31
    F(5000.000000003764) = LineSearchPoint{point=PointSample{avg=4.378375239202921E-31}, derivative=2.3291590878216107E-22}, delta = -7.744000000000014E-7
    Right bracket at 5000.000000003764
    Converged to right
    Iteration 1 complete. Error: 4.378375239202921E-31 Total: 239672457361527.6000; Orientation: 0.0006; Line Search: 0.0384
    Zero gradient: 1.3233858453531866E-17
    F(0.0) = LineSearchPoint{point=PointSample{avg=4.378375239202921E-31}, derivative=-1.7513500956811683E-34}
    New Minimum: 4.378375239202921E-31 > 0.0
    F(5000.000000003764) = LineSearchPoint{point=PointSample{avg=0.0}, derivative=0.0}, delta = -4.378375239202921E-31
    0.0 <= 4.378375239202921E-31
    Converged to right
    Iteration 2 complete. Error: 0.0 Total: 239672466193862.5300; Orientation: 0.0004; Line Search: 0.0071
    
```

Returns: 

```
    0.0
```



Training Converged

Next, we run the same optimization using L-BFGS, which is nearly ideal for purely second-order or quadratic functions.

Code from [LearningTester.java:249](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L249) executed in 0.06 seconds: 
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
    th(0)=7.744000000000014E-7;dx=-3.0976000000000064E-10
    New Minimum: 7.744000000000014E-7 > 7.737327860882731E-7
    WOLFE (weak): th(2.154434690031884)=7.737327860882731E-7; dx=-3.096265284620839E-10 delta=6.672139117282852E-10
    New Minimum: 7.737327860882731E-7 > 7.730658597322565E-7
    WOLFE (weak): th(4.308869380063768)=7.730658597322565E-7; dx=-3.09493056924167E-10 delta=1.334140267744953E-9
    New Minimum: 7.730658597322565E-7 > 7.704010298653017E-7
    WOLFE (weak): th(12.926608140191302)=7.704010298653017E-7; dx=-3.089591707724996E-10 delta=3.998970134699684E-9
    New Minimum: 7.704010298653017E-7 > 7.584662314948674E-7
    WOLFE (weak): th(51.70643256076521)=7.584662314948674E-7; dx=-3.06556683089996E-10 delta=1.5933768505134048E-8
    New Minimum: 7.584662314948674E-7 > 6.963874783721125E-7
    WOLFE (weak): th(258.53216280382605)=6.963874783721125E-7; dx=-2.937434154499781E-10 delta=7.801252162788894E-8
    New Minimum: 6.963874783721125E-7 > 3.
```
...[skipping 2681 bytes](etc/281.txt)...
```
    6601118108582E-16 delta=4.285858776717875E-12
    Iteration 6 complete. Error: 4.329150279541005E-14 Total: 239672519923403.5000; Orientation: 0.0005; Line Search: 0.0047
    LBFGS Accumulation History: 1 points
    th(0)=4.329150279541005E-14;dx=-1.731660111816402E-17
    New Minimum: 4.329150279541005E-14 > 3.817031339616296E-14
    WOLF (strong): th(9694.956105143481)=3.817031339616296E-14; dx=1.6260136427922944E-17 delta=5.121189399247094E-15
    New Minimum: 3.817031339616296E-14 > 4.0283512975482725E-17
    END: th(4847.478052571741)=4.0283512975482725E-17; dx=-5.282323450281397E-19 delta=4.325121928243457E-14
    Iteration 7 complete. Error: 4.0283512975482725E-17 Total: 239672526392412.4700; Orientation: 0.0005; Line Search: 0.0049
    LBFGS Accumulation History: 1 points
    th(0)=4.0283512975482725E-17;dx=-1.611340519019309E-20
    MAX ALPHA: th(0)=4.0283512975482725E-17;th'(0)=-1.611340519019309E-20;
    Iteration 8 failed, aborting. Error: 4.0283512975482725E-17 Total: 239672530862583.4700; Orientation: 0.0005; Line Search: 0.0028
    
```

Returns: 

```
    4.0283512975482725E-17
```



Training Converged

Code from [LearningTester.java:95](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L95) executed in 0.00 seconds: 
```java
    return TestUtil.compare(runs);
```

Returns: 

![Result](etc/test.188.png)



Code from [LearningTester.java:98](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L98) executed in 0.00 seconds: 
```java
    return TestUtil.compareTime(runs);
```

Returns: 

![Result](etc/test.189.png)



### Performance
Now we execute larger-scale runs to benchmark performance:

Code from [PerformanceTester.java:66](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/PerformanceTester.java#L66) executed in 0.74 seconds: 
```java
    test(component, inputPrototype);
```
Logging: 
```
    100 batches
    Input Dimensions:
    	[100, 100, 1]
    Performance:
    	Evaluation performance: 0.067475s +- 0.015842s [0.053731s - 0.098176s]
    	Learning performance: 0.011238s +- 0.001103s [0.010542s - 0.013428s]
    
```

### Function Plots
Code from [ActivationLayerTestBase.java:110](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/java/ActivationLayerTestBase.java#L110) executed in 0.00 seconds: 
```java
    return plot("Value Plot", plotData, x -> new double[]{x[0], x[1]});
```

Returns: 

![Result](etc/test.190.png)



Code from [ActivationLayerTestBase.java:114](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/java/ActivationLayerTestBase.java#L114) executed in 0.00 seconds: 
```java
    return plot("Derivative Plot", plotData, x -> new double[]{x[0], x[2]});
```

Returns: 

![Result](etc/test.191.png)



