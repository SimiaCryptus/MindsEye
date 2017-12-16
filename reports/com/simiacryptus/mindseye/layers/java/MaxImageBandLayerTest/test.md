# MaxImageBandLayer
## MaxImageBandLayerTest
### Batch Execution
Code from [BatchingTester.java:66](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/BatchingTester.java#L66) executed in 0.01 seconds: 
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
    	[ [ 0.76, -1.412, -1.316 ], [ 1.78, -0.088, 0.596 ] ],
    	[ [ 1.508, 1.824, 1.032 ], [ -1.496, -0.172, -1.76 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.06423902553822511, negative=6, min=-1.76, max=-1.76, mean=0.10466666666666667, count=12.0, positive=6, stdDev=1.2823923823160452, zeros=0}
    Output: [
    	[ [ -1.496, -0.172, -1.76 ] ]
    ]
    Outputs Statistics: {meanExponent=-0.11467576391661956, negative=3, min=-1.76, max=-1.76, mean=-1.1426666666666667, count=3.0, positive=0, stdDev=0.6947754233483571, zeros=0}
    Feedback for input 0
    Inputs Values: [
    	[ [ 0.76, -1.412, -1.316 ], [ 1.78, -0.088, 0.596 ] ],
    	[ [ 1.508, 1.824, 1.032 ], [ -1.496, -0.172, -1.76 ] ]
    ]
    Value Statistics: {meanExponent=-0.06423902553822511, negative=6, min=-1.76, max=-1.76, mean=0.10466666666666667, count=12.0, positive=6, stdDev=1.2823923823160452, zeros=0}
    Implemented Feedback: [ [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 1.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 1.0, 0.0 ]
```
...[skipping 180 bytes](etc/282.txt)...
```
    eedback: [ [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.9999999999998899, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.9999999999998899, 0.0 ], ... ]
    Measured Statistics: {meanExponent=-4.7830642341045674E-14, negative=0, min=0.9999999999998899, max=0.9999999999998899, mean=0.08333333333332416, count=36.0, positive=3, stdDev=0.2763853991962529, zeros=33}
    Feedback Error: [ [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ -1.1013412404281553E-13, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, -1.1013412404281553E-13, 0.0 ], ... ]
    Error Statistics: {meanExponent=-12.958078098036827, negative=3, min=-1.1013412404281553E-13, max=-1.1013412404281553E-13, mean=-9.177843670234628E-15, count=36.0, positive=0, stdDev=3.0439463838706555E-14, zeros=33}
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
      "class": "com.simiacryptus.mindseye.layers.java.MaxImageBandLayer",
      "id": "f9190f75-8993-4874-920d-1dde9151e28f",
      "isFrozen": false,
      "name": "MaxImageBandLayer/f9190f75-8993-4874-920d-1dde9151e28f"
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
    	[ [ 1.756, -0.776, -1.216 ], [ 0.708, 1.176, 0.888 ] ],
    	[ [ 0.852, 0.716, -1.388 ], [ -1.668, -0.096, 1.792 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ -1.668, -0.096, 1.792 ] ]
    ]
    --------------------
    Derivative: 
    [
    	[ [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ] ],
    	[ [ 0.0, 0.0, 0.0 ], [ 1.0, 1.0, 1.0 ] ]
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
    	[ [ 0.496, -0.92, -1.94 ], [ -0.952, 0.076, -1.692 ] ],
    	[ [ 1.952, -1.208, -1.132 ], [ -0.116, 0.612, 1.524 ] ]
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
    F(0.0) = LineSearchPoint{point=PointSample{avg=2.5653866666666665}, derivative=-3.4205155555555544}
    New Minimum: 2.5653866666666665 > 2.5653866663246148
    F(1.0E-10) = LineSearchPoint{point=PointSample{avg=2.5653866663246148}, derivative=-3.42051555532752}, delta = -3.4205172028123343E-10
    New Minimum: 2.5653866663246148 > 2.5653866642723058
    F(7.000000000000001E-10) = LineSearchPoint{point=PointSample{avg=2.5653866642723058}, derivative=-3.4205155539593144}, delta = -2.3943607097010045E-9
    New Minimum: 2.5653866642723058 > 2.56538664990614
    F(4.900000000000001E-9) = LineSearchPoint{point=PointSample{avg=2.56538664990614}, derivative=-3.4205155443818707}, delta = -1.676052630017466E-8
    New Minimum: 2.56538664990614 > 2.5653865493429846
    F(3.430000000000001E-8) = LineSearchPoint{point=PointSample{avg=2.5653865493429846}, derivative=-3.420515477339766}, delta = -1.1732368188077658E-7
    New Minimum: 2.5653865493429846 > 2.565385845400947
    F(2.4010000000000004E-7) = LineSea
```
...[skipping 1559 bytes](etc/283.txt)...
```
    83 > 0.01530812230580035
    F(1.3841287201) = LineSearchPoint{point=PointSample{avg=0.01530812230580035}, derivative=-0.26422634356005475}, delta = -2.5500785443608662
    Loops = 12
    New Minimum: 0.01530812230580035 > 5.861246797418487E-32
    F(1.5000000000000004) = LineSearchPoint{point=PointSample{avg=5.861246797418487E-32}, derivative=4.721161733161555E-16}, delta = -2.5653866666666665
    Right bracket at 1.5000000000000004
    Converged to right
    Iteration 1 complete. Error: 5.861246797418487E-32 Total: 239673603894279.4000; Orientation: 0.0000; Line Search: 0.0021
    Zero gradient: 2.795531385960694E-16
    F(0.0) = LineSearchPoint{point=PointSample{avg=5.861246797418487E-32}, derivative=-7.814995729891317E-32}
    New Minimum: 5.861246797418487E-32 > 0.0
    F(1.5000000000000004) = LineSearchPoint{point=PointSample{avg=0.0}, derivative=0.0}, delta = -5.861246797418487E-32
    0.0 <= 5.861246797418487E-32
    Converged to right
    Iteration 2 complete. Error: 0.0 Total: 239673604214880.4000; Orientation: 0.0000; Line Search: 0.0002
    
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
    th(0)=2.5653866666666665;dx=-3.4205155555555544
    New Minimum: 2.5653866666666665 > 0.48831823193925566
    WOLF (strong): th(2.154434690031884)=0.48831823193925566; dx=1.4923360248994901 delta=2.0770684347274107
    New Minimum: 0.48831823193925566 > 0.2038002153141716
    END: th(1.077217345015942)=0.2038002153141716; dx=-0.9640897653280324 delta=2.361586451352495
    Iteration 1 complete. Error: 0.2038002153141716 Total: 239673608245899.4000; Orientation: 0.0001; Line Search: 0.0005
    LBFGS Accumulation History: 1 points
    th(0)=0.2038002153141716;dx=-0.2717336204188955
    New Minimum: 0.2038002153141716 > 0.06102262808609949
    WOLF (strong): th(2.3207944168063896)=0.06102262808609949; dx=0.14869162566561078 delta=0.14277758722807213
    New Minimum: 0.06102262808609949 > 0.010446351225463732
    END: th(1.1603972084031948)=0.010446351225463732; dx=-0.06152099737664235 delta=0.1933538640887079
    Iteration 2 complete. Error: 0.010446351225463732 Total: 2
```
...[skipping 9630 bytes](etc/284.txt)...
```
    0628908E-31 Total: 239673616464391.3800; Orientation: 0.0000; Line Search: 0.0003
    LBFGS Accumulation History: 1 points
    th(0)=9.59434100628908E-31;dx=-1.2792454675052104E-30
    New Minimum: 9.59434100628908E-31 > 7.952806717023025E-31
    WOLF (strong): th(2.8257016782407427)=7.952806717023025E-31; dx=1.1636040739555592E-30 delta=1.6415342892660552E-31
    New Minimum: 7.952806717023025E-31 > 5.7135921683488E-33
    END: th(1.4128508391203713)=5.7135921683488E-33; dx=-8.859277744181284E-32 delta=9.537205084605592E-31
    Iteration 21 complete. Error: 5.7135921683488E-33 Total: 239673616863646.3800; Orientation: 0.0000; Line Search: 0.0003
    LBFGS Accumulation History: 1 points
    th(0)=5.7135921683488E-33;dx=-7.618122891131732E-33
    Armijo: th(3.043894859641584)=5.7135921683488E-33; dx=7.618122891131732E-33 delta=0.0
    New Minimum: 5.7135921683488E-33 > 0.0
    END: th(1.521947429820792)=0.0; dx=0.0 delta=5.7135921683488E-33
    Iteration 22 complete. Error: 0.0 Total: 239673617211889.3800; Orientation: 0.0000; Line Search: 0.0003
    
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

![Result](etc/test.192.png)



Code from [LearningTester.java:98](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L98) executed in 0.00 seconds: 
```java
    return TestUtil.compareTime(runs);
```

Returns: 

![Result](etc/test.193.png)



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
    	Evaluation performance: 0.000612s +- 0.000116s [0.000513s - 0.000834s]
    	Learning performance: 0.000032s +- 0.000010s [0.000025s - 0.000052s]
    
```

