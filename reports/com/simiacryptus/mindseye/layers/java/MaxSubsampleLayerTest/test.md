# MaxSubsampleLayer
## MaxSubsampleLayerTest
### Batch Execution
Code from [BatchingTester.java:66](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/BatchingTester.java#L66) executed in 0.00 seconds: 
```java
    return test(reference, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (150#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (60#)}
```



Code from [SingleDerivativeTester.java:77](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/SingleDerivativeTester.java#L77) executed in 0.00 seconds: 
```java
    return test(component, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ -1.432, -0.352, 0.164 ], [ 1.168, -0.068, 0.368 ] ],
    	[ [ 0.672, -1.652, 0.38 ], [ -0.124, 0.236, -0.768 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.38666775629394556, negative=6, min=-0.768, max=-0.768, mean=-0.11733333333333333, count=12.0, positive=6, stdDev=0.7919635232901968, zeros=0}
    Output: [
    	[ [ 1.168, 0.236, 0.38 ] ]
    ]
    Outputs Statistics: {meanExponent=-0.32662051921223423, negative=0, min=0.38, max=0.38, mean=0.5946666666666666, count=3.0, positive=3, stdDev=0.40964808745502, zeros=0}
    Feedback for input 0
    Inputs Values: [
    	[ [ -1.432, -0.352, 0.164 ], [ 1.168, -0.068, 0.368 ] ],
    	[ [ 0.672, -1.652, 0.38 ], [ -0.124, 0.236, -0.768 ] ]
    ]
    Value Statistics: {meanExponent=-0.38666775629394556, negative=6, min=-0.768, max=-0.768, mean=-0.11733333333333333, count=12.0, positive=6, stdDev=0.7919635232901968, zeros=0}
    Implemented Feedback: [ [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 1.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 1.0, 0.0
```
...[skipping 112 bytes](etc/287.txt)...
```
    unt=36.0, positive=3, stdDev=0.2763853991962833, zeros=33}
    Measured Feedback: [ [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.9999999999998899, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.9999999999998899, 0.0 ], ... ]
    Measured Statistics: {meanExponent=-4.7830642341045674E-14, negative=0, min=0.0, max=0.0, mean=0.08333333333332416, count=36.0, positive=3, stdDev=0.2763853991962529, zeros=33}
    Feedback Error: [ [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ -1.1013412404281553E-13, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, -1.1013412404281553E-13, 0.0 ], ... ]
    Error Statistics: {meanExponent=-12.958078098036827, negative=3, min=0.0, max=0.0, mean=-9.177843670234628E-15, count=36.0, positive=0, stdDev=3.0439463838706555E-14, zeros=33}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 9.1778e-15 +- 3.0439e-14 [0.0000e+00 - 1.1013e-13] (36#)
    relativeTol: 5.5067e-14 +- 0.0000e+00 [5.5067e-14 - 5.5067e-14] (3#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=9.1778e-15 +- 3.0439e-14 [0.0000e+00 - 1.1013e-13] (36#), relativeTol=5.5067e-14 +- 0.0000e+00 [5.5067e-14 - 5.5067e-14] (3#)}
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
      "class": "com.simiacryptus.mindseye.layers.java.MaxSubsampleLayer",
      "id": "cbdd362a-0ad8-4458-9e83-29b4a67515de",
      "isFrozen": false,
      "name": "MaxSubsampleLayer/cbdd362a-0ad8-4458-9e83-29b4a67515de",
      "inner": [
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
    	[ [ -1.056, -0.46, 0.62 ], [ -1.668, 0.24, -0.384 ] ],
    	[ [ 1.0, -0.82, 1.904 ], [ -0.504, 1.1, 0.064 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 1.0, 1.1, 1.904 ] ]
    ]
    --------------------
    Derivative: 
    [
    	[ [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ] ],
    	[ [ 1.0, 0.0, 1.0 ], [ 0.0, 1.0, 0.0 ] ]
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
    	[ [ -0.048, -1.692, 0.352 ], [ -0.22, 1.192, 0.544 ] ],
    	[ [ -1.428, 0.556, -0.844 ], [ 1.54, -0.404, 1.744 ] ]
    ]
```



First, we use a conjugate gradient descent method, which converges the fastest for purely linear functions.

Code from [LearningTester.java:225](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L225) executed in 0.00 seconds: 
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
    F(0.0) = LineSearchPoint{point=PointSample{avg=0.1752}, derivative=-0.23359999999999995}
    New Minimum: 0.1752 > 0.17519999997663996
    F(1.0E-10) = LineSearchPoint{point=PointSample{avg=0.17519999997663996}, derivative=-0.23359999998442663}, delta = -2.3360036127684225E-11
    New Minimum: 0.17519999997663996 > 0.17519999983647994
    F(7.000000000000001E-10) = LineSearchPoint{point=PointSample{avg=0.17519999983647994}, derivative=-0.2335999998909866}, delta = -1.6352005860476027E-10
    New Minimum: 0.17519999983647994 > 0.17519999885535997
    F(4.900000000000001E-9) = LineSearchPoint{point=PointSample{avg=0.17519999885535997}, derivative=-0.23359999923690664}, delta = -1.1446400216552632E-9
    New Minimum: 0.17519999885535997 > 0.1751999919875201
    F(3.430000000000001E-8) = LineSearchPoint{point=PointSample{avg=0.1751999919875201}, derivative=-0.23359999465834663}, delta = -8.012479901786662E-9
    New Minimum: 0.1751999919875201 > 0.17519994391264446
    F(2.4010000000000004E-7) = LineSe
```
...[skipping 1029 bytes](etc/288.txt)...
```
     = LineSearchPoint{point=PointSample{avg=0.1742586077318682}, derivative=-0.2329715598269866}, delta = -9.413922681318077E-4
    New Minimum: 0.1742586077318682 > 0.16866350976138375
    F(0.028247524900000005) = LineSearchPoint{point=PointSample{avg=0.16866350976138375}, derivative=-0.22920091878890664}, delta = -0.006536490238616244
    New Minimum: 0.16866350976138375 > 0.13205409460668327
    F(0.19773267430000002) = LineSearchPoint{point=PointSample{avg=0.13205409460668327}, derivative=-0.20280643152234662}, delta = -0.043145905393316725
    New Minimum: 0.13205409460668327 > 0.001045449819641048
    F(1.3841287201) = LineSearchPoint{point=PointSample{avg=0.001045449819641048}, derivative=-0.01804502065642666}, delta = -0.17415455018035894
    Loops = 12
    New Minimum: 0.001045449819641048 > 0.0
    F(1.5) = LineSearchPoint{point=PointSample{avg=0.0}, derivative=0.0}, delta = -0.1752
    Right bracket at 1.5
    Converged to right
    Iteration 1 complete. Error: 0.0 Total: 239673770098806.2200; Orientation: 0.0000; Line Search: 0.0018
    
```

Returns: 

```
    0.0
```



Training Converged

Next, we run the same optimization using L-BFGS, which is nearly ideal for purely second-order or quadratic functions.

Code from [LearningTester.java:249](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L249) executed in 0.01 seconds: 
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
    th(0)=0.1752;dx=-0.23359999999999995
    New Minimum: 0.1752 > 0.033349106919200344
    WOLF (strong): th(2.154434690031884)=0.033349106919200344; dx=0.10191729572763202 delta=0.14185089308079965
    New Minimum: 0.033349106919200344 > 0.013918290831938074
    END: th(1.077217345015942)=0.013918290831938074; dx=-0.06584135213618397 delta=0.16128170916806192
    Iteration 1 complete. Error: 0.013918290831938074 Total: 239673774672709.2200; Orientation: 0.0001; Line Search: 0.0004
    LBFGS Accumulation History: 1 points
    th(0)=0.013918290831938074;dx=-0.018557721109250764
    New Minimum: 0.013918290831938074 > 0.004167467064361166
    WOLF (strong): th(2.3207944168063896)=0.004167467064361166; dx=0.01015471591674873 delta=0.009750823767576907
    New Minimum: 0.004167467064361166 > 7.134210052940326E-4
    END: th(1.1603972084031948)=7.134210052940326E-4; dx=-0.004201502596251008 delta=0.01320486982664404
    Iteration 2 complete. Error: 7.134210052940326E-4 Total
```
...[skipping 9107 bytes](etc/289.txt)...
```
    73784692549.2200; Orientation: 0.0000; Line Search: 0.0003
    LBFGS Accumulation History: 1 points
    th(0)=4.2729965699471475E-30;dx=-5.6973287599295284E-30
    New Minimum: 4.2729965699471475E-30 > 2.2351058981262002E-30
    WOLF (strong): th(2.623149071368624)=2.2351058981262002E-30; dx=4.1196069494875055E-30 delta=2.0378906718209473E-30
    New Minimum: 2.2351058981262002E-30 > 8.217301096052207E-32
    END: th(1.311574535684312)=8.217301096052207E-32; dx=-7.888609052210117E-31 delta=4.190823558986625E-30
    Iteration 20 complete. Error: 8.217301096052207E-32 Total: 239673785164473.2200; Orientation: 0.0000; Line Search: 0.0004
    LBFGS Accumulation History: 1 points
    th(0)=8.217301096052207E-32;dx=-1.0956401461402942E-31
    Armijo: th(2.8257016782407427)=8.217301096052207E-32; dx=1.0956401461402942E-31 delta=0.0
    New Minimum: 8.217301096052207E-32 > 0.0
    END: th(1.4128508391203713)=0.0; dx=0.0 delta=8.217301096052207E-32
    Iteration 21 complete. Error: 0.0 Total: 239673785524971.2200; Orientation: 0.0000; Line Search: 0.0002
    
```

Returns: 

```
    0.0
```



Training Converged

Code from [LearningTester.java:95](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L95) executed in 0.00 seconds: 
```java
    return TestUtil.compare(runs);
```

Returns: 

![Result](etc/test.194.png)



Code from [LearningTester.java:98](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L98) executed in 0.00 seconds: 
```java
    return TestUtil.compareTime(runs);
```

Returns: 

![Result](etc/test.195.png)



### Performance
Now we execute larger-scale runs to benchmark performance:

Code from [PerformanceTester.java:66](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/PerformanceTester.java#L66) executed in 0.01 seconds: 
```java
    test(component, inputPrototype);
```
Logging: 
```
    100 batches
    Input Dimensions:
    	[2, 2, 3]
    Performance:
    	Evaluation performance: 0.001776s +- 0.001368s [0.000940s - 0.004501s]
    	Learning performance: 0.000045s +- 0.000011s [0.000036s - 0.000066s]
    
```

