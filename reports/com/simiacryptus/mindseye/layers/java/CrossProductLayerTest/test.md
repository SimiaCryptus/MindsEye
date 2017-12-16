# CrossProductLayer
## CrossProductLayerTest
### Batch Execution
Code from [BatchingTester.java:66](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/BatchingTester.java#L66) executed in 0.01 seconds: 
```java
    return test(reference, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (100#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (100#)}
```



Code from [SingleDerivativeTester.java:77](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/SingleDerivativeTester.java#L77) executed in 0.00 seconds: 
```java
    return test(component, inputPrototype);
```
Logging: 
```
    Inputs: [ 0.596, -0.092, 0.416, -1.748 ]
    Inputs Statistics: {meanExponent=-0.34983278849727034, negative=2, min=-1.748, max=-1.748, mean=-0.20700000000000002, count=4.0, positive=2, stdDev=0.9247761891398372, zeros=0}
    Output: [ -0.054832, 0.247936, -1.0418079999999998, -0.038272, 0.160816, -0.7271679999999999 ]
    Outputs Statistics: {meanExponent=-0.6996655769945407, negative=4, min=-0.7271679999999999, max=-0.7271679999999999, mean=-0.24222133333333332, count=6.0, positive=2, stdDev=0.4750290347823196, zeros=0}
    Feedback for input 0
    Inputs Values: [ 0.596, -0.092, 0.416, -1.748 ]
    Value Statistics: {meanExponent=-0.34983278849727034, negative=2, min=-1.748, max=-1.748, mean=-0.20700000000000002, count=4.0, positive=2, stdDev=0.9247761891398372, zeros=0}
    Implemented Feedback: [ [ -0.092, 0.416, -1.748, 0.0, 0.0, 0.0 ], [ 0.596, 0.0, 0.0, 0.416, -1.748, 0.0 ], [ 0.0, 0.596, 0.0, -0.092, 0.0, -1.748 ], [ 0.0, 0.0, 0.596, 0.0, -0.092, 0.416 ] ]
    Implemented Statistics: {meanExponent=-0.3498327884972703, negat
```
...[skipping 426 bytes](etc/215.txt)...
```
    99999999986996, 0.4159999999997499 ] ]
    Measured Statistics: {meanExponent=-0.34983278849741223, negative=6, min=0.4159999999997499, max=0.4159999999997499, mean=-0.10350000000010813, count=24.0, positive=6, stdDev=0.6620557000737798, zeros=12}
    Feedback Error: [ [ 6.064593272014918E-14, -2.500777362968165E-13, -1.4150902671872245E-12, 0.0, 0.0, 0.0 ], [ 6.872280522429719E-14, 0.0, 0.0, 2.7478019859472624E-14, -2.731148640577885E-14, 0.0 ], [ 0.0, -7.005507285384738E-14, 0.0, 6.064593272014918E-14, 0.0, -3.04867242562068E-13 ], [ 0.0, 0.0, -6.251665851664256E-13, 0.0, 1.3003487175922146E-13, -2.500777362968165E-13 ] ]
    Error Statistics: {meanExponent=-12.87795207050952, negative=7, min=-2.500777362968165E-13, max=-2.500777362968165E-13, mean=-1.0812994018690365E-13, count=24.0, positive=5, stdDev=3.139879187266443E-13, zeros=12}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 1.3709e-13 +- 3.0247e-13 [0.0000e+00 - 1.4151e-12] (24#)
    relativeTol: 2.6173e-13 +- 2.1012e-13 [7.8122e-15 - 7.0671e-13] (12#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=1.3709e-13 +- 3.0247e-13 [0.0000e+00 - 1.4151e-12] (24#), relativeTol=2.6173e-13 +- 2.1012e-13 [7.8122e-15 - 7.0671e-13] (12#)}
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
      "class": "com.simiacryptus.mindseye.layers.java.CrossProductLayer",
      "id": "367933c7-4070-4b20-a1da-dc9570dc3c49",
      "isFrozen": false,
      "name": "CrossProductLayer/367933c7-4070-4b20-a1da-dc9570dc3c49"
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
    [[ -1.92, 0.732, 1.264, -0.28 ]]
    --------------------
    Output: 
    [ -1.40544, -2.42688, 0.5376000000000001, 0.925248, -0.20496, -0.35392 ]
    --------------------
    Derivative: 
    [ 1.716, -0.9359999999999999, -1.468, 0.07600000000000007 ]
```



### Input Learning
In this test, we use a network to learn this target input, given it's pre-evaluated output:

Code from [LearningTester.java:127](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L127) executed in 0.00 seconds: 
```java
    return Arrays.stream(input_target).map(x -> x.prettyPrint()).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [ 0.548, 0.816, 0.02, -0.536 ]
```



First, we use a conjugate gradient descent method, which converges the fastest for purely linear functions.

Code from [LearningTester.java:225](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L225) executed in 0.08 seconds: 
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
    F(0.0) = LineSearchPoint{point=PointSample{avg=0.2028986515626667}, derivative=-0.14139175358454079}
    New Minimum: 0.2028986515626667 > 0.2028986515485275
    F(1.0E-10) = LineSearchPoint{point=PointSample{avg=0.2028986515485275}, derivative=-0.14139175357246608}, delta = -1.4139217574538065E-11
    New Minimum: 0.2028986515485275 > 0.20289865146369243
    F(7.000000000000001E-10) = LineSearchPoint{point=PointSample{avg=0.20289865146369243}, derivative=-0.14139175350001787}, delta = -9.897427322158592E-11
    New Minimum: 0.20289865146369243 > 0.20289865086984712
    F(4.900000000000001E-9) = LineSearchPoint{point=PointSample{avg=0.20289865086984712}, derivative=-0.14139175299288048}, delta = -6.92819579484194E-10
    New Minimum: 0.20289865086984712 > 0.20289864671292956
    F(3.430000000000001E-8) = LineSearchPoint{point=PointSample{avg=0.20289864671292956}, derivative=-0.14139174944291882}, delta = -4.849737139656085E-9
    New Minimum: 0.20289864671292956 > 0.2028986176145101
    F(2.4010000
```
...[skipping 161280 bytes](etc/216.txt)...
```
    13
    Zero gradient: 1.4888073469034735E-17
    F(0.0) = LineSearchPoint{point=PointSample{avg=1.0276641812628766E-33}, derivative=-2.21654731619376E-34}
    New Minimum: 1.0276641812628766E-33 > 1.0030885127016853E-36
    F(6.112495796772966) = LineSearchPoint{point=PointSample{avg=1.0030885127016853E-36}, derivative=2.561058841602939E-37}, delta = -1.026661092750175E-33
    1.0030885127016853E-36 <= 1.0276641812628766E-33
    Converged to right
    Iteration 127 complete. Error: 1.0030885127016853E-36 Total: 239636989509690.0000; Orientation: 0.0000; Line Search: 0.0001
    Zero gradient: 6.270267528028219E-19
    F(0.0) = LineSearchPoint{point=PointSample{avg=1.0030885127016853E-36}, derivative=-3.9316254873045105E-37}
    New Minimum: 1.0030885127016853E-36 > 0.0
    F(6.112495796772966) = LineSearchPoint{point=PointSample{avg=0.0}, derivative=0.0}, delta = -1.0030885127016853E-36
    0.0 <= 1.0030885127016853E-36
    Converged to right
    Iteration 128 complete. Error: 0.0 Total: 239636989699200.0000; Orientation: 0.0000; Line Search: 0.0001
    
```

Returns: 

```
    0.0
```



Training Converged

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
    th(0)=0.2028986515626667;dx=-0.14139175358454079
    New Minimum: 0.2028986515626667 > 0.07606226337729484
    END: th(2.154434690031884)=0.07606226337729484; dx=-0.013234942988025113 delta=0.12683638818537185
    Iteration 1 complete. Error: 0.07606226337729484 Total: 239636993445526.0000; Orientation: 0.0001; Line Search: 0.0002
    LBFGS Accumulation History: 1 points
    th(0)=0.07606226337729484;dx=-0.0046834538466270885
    New Minimum: 0.07606226337729484 > 0.04866679811394911
    WOLFE (weak): th(4.641588833612779)=0.04866679811394911; dx=-0.006805225055301483 delta=0.027395465263345728
    New Minimum: 0.04866679811394911 > 0.017620459272103622
    WOLFE (weak): th(9.283177667225559)=0.017620459272103622; dx=-0.005868360965848489 delta=0.05844180410519122
    Armijo: th(27.849533001676676)=0.4111492425703384; dx=0.07505024254124623 delta=-0.3350869791930436
    WOLF (strong): th(18.566355334451117)=0.0337969897003983; dx=0.014498231941041446 delta=0.04226
```
...[skipping 49566 bytes](etc/217.txt)...
```
    667E-35 delta=-3.510809794455902E-36
    New Minimum: 3.05941996374014E-35 > 3.009265538105056E-36
    END: th(1.5470237481473932)=3.009265538105056E-36; dx=-7.888737447539741E-36 delta=2.7584934099296345E-35
    Iteration 102 complete. Error: 3.009265538105056E-36 Total: 239637039671870.9700; Orientation: 0.0001; Line Search: 0.0005
    LBFGS Accumulation History: 1 points
    th(0)=3.009265538105056E-36;dx=-2.467282102727482E-36
    New Minimum: 3.009265538105056E-36 > 1.0030885127016853E-36
    END: th(3.332961629311892)=1.0030885127016853E-36; dx=9.84679832316584E-37 delta=2.006177025403371E-36
    Iteration 103 complete. Error: 1.0030885127016853E-36 Total: 239637039950579.9700; Orientation: 0.0000; Line Search: 0.0002
    LBFGS Accumulation History: 1 points
    th(0)=1.0030885127016853E-36;dx=-3.9316254873045105E-37
    New Minimum: 1.0030885127016853E-36 > 0.0
    END: th(7.180648154734729)=0.0; dx=0.0 delta=1.0030885127016853E-36
    Iteration 104 complete. Error: 0.0 Total: 239637040188536.9700; Orientation: 0.0000; Line Search: 0.0001
    
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

![Result](etc/test.128.png)



Code from [LearningTester.java:98](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L98) executed in 0.01 seconds: 
```java
    return TestUtil.compareTime(runs);
```

Returns: 

![Result](etc/test.129.png)



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
    	[4]
    Performance:
    	Evaluation performance: 0.000149s +- 0.000015s [0.000129s - 0.000172s]
    	Learning performance: 0.000046s +- 0.000005s [0.000041s - 0.000056s]
    
```

