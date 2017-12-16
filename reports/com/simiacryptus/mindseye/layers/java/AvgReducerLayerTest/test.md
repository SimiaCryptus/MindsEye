# AvgReducerLayer
## AvgReducerLayerTest
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
      "class": "com.simiacryptus.mindseye.layers.java.AvgReducerLayer",
      "id": "7e840654-84dd-430c-bb04-71b513431070",
      "isFrozen": false,
      "name": "AvgReducerLayer/7e840654-84dd-430c-bb04-71b513431070"
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
    [[ -1.984, -1.244, 0.468 ]]
    --------------------
    Output: 
    [ -0.92 ]
    --------------------
    Derivative: 
    [ 0.3333333333333333, 0.3333333333333333, 0.3333333333333333 ]
```



### Batch Execution
Code from [BatchingTester.java:66](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/BatchingTester.java#L66) executed in 0.00 seconds: 
```java
    return test(reference, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (40#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (40#)}
```



Code from [SingleDerivativeTester.java:77](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/SingleDerivativeTester.java#L77) executed in 0.00 seconds: 
```java
    return test(component, inputPrototype);
```
Logging: 
```
    Inputs: [ 0.044, 0.896, 1.836 ]
    Inputs Statistics: {meanExponent=-0.38012221232882126, negative=0, min=1.836, max=1.836, mean=0.9253333333333335, count=3.0, positive=3, stdDev=0.7318749134168275, zeros=0}
    Output: [ 0.9253333333333333 ]
    Outputs Statistics: {meanExponent=-0.03370179293684513, negative=0, min=0.9253333333333333, max=0.9253333333333333, mean=0.9253333333333333, count=1.0, positive=1, stdDev=0.0, zeros=0}
    Feedback for input 0
    Inputs Values: [ 0.044, 0.896, 1.836 ]
    Value Statistics: {meanExponent=-0.38012221232882126, negative=0, min=1.836, max=1.836, mean=0.9253333333333335, count=3.0, positive=3, stdDev=0.7318749134168275, zeros=0}
    Implemented Feedback: [ [ 0.3333333333333333 ], [ 0.3333333333333333 ], [ 0.3333333333333333 ] ]
    Implemented Statistics: {meanExponent=-0.47712125471966244, negative=0, min=0.3333333333333333, max=0.3333333333333333, mean=0.3333333333333333, count=3.0, positive=3, stdDev=0.0, zeros=0}
    Measured Feedback: [ [ 0.3333333333332966 ], [ 0.3333333333332966 ], [ 0.3333333333332966 ] ]
    Measured Statistics: {meanExponent=-0.4771212547197103, negative=0, min=0.3333333333332966, max=0.3333333333332966, mean=0.3333333333332966, count=3.0, positive=3, stdDev=0.0, zeros=0}
    Feedback Error: [ [ -3.6692870963861424E-14 ], [ -3.6692870963861424E-14 ], [ -3.6692870963861424E-14 ] ]
    Error Statistics: {meanExponent=-13.435418306369344, negative=3, min=-3.6692870963861424E-14, max=-3.6692870963861424E-14, mean=-3.6692870963861424E-14, count=3.0, positive=0, stdDev=0.0, zeros=0}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 3.6693e-14 +- 0.0000e+00 [3.6693e-14 - 3.6693e-14] (3#)
    relativeTol: 5.5039e-14 +- 0.0000e+00 [5.5039e-14 - 5.5039e-14] (3#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=3.6693e-14 +- 0.0000e+00 [3.6693e-14 - 3.6693e-14] (3#), relativeTol=5.5039e-14 +- 0.0000e+00 [5.5039e-14 - 5.5039e-14] (3#)}
```



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
    	[100, 100, 1]
    Performance:
    	Evaluation performance: 0.001462s +- 0.002277s [0.000280s - 0.006014s]
    	Learning performance: 0.000033s +- 0.000004s [0.000028s - 0.000038s]
    
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
    	[ [ -0.256 ], [ 1.176 ], [ -0.548 ], [ 0.452 ], [ 0.768 ], [ -1.012 ], [ -1.008 ], [ -1.004 ], ... ],
    	[ [ 1.192 ], [ 0.976 ], [ -0.02 ], [ 0.828 ], [ -1.948 ], [ -0.264 ], [ -1.912 ], [ -1.432 ], ... ],
    	[ [ -1.716 ], [ -1.88 ], [ 1.408 ], [ -1.456 ], [ 1.448 ], [ -0.28 ], [ 0.264 ], [ 0.376 ], ... ],
    	[ [ 1.156 ], [ 0.764 ], [ 1.24 ], [ -1.28 ], [ 1.412 ], [ 1.544 ], [ -0.356 ], [ 0.372 ], ... ],
    	[ [ 0.616 ], [ -1.388 ], [ -0.048 ], [ -1.144 ], [ -1.208 ], [ -1.66 ], [ -0.468 ], [ -1.632 ], ... ],
    	[ [ 0.288 ], [ 0.04 ], [ 0.212 ], [ 0.048 ], [ 1.7 ], [ -0.604 ], [ -1.372 ], [ -1.8 ], ... ],
    	[ [ -0.68 ], [ -0.256 ], [ 1.288 ], [ -0.556 ], [ -0.668 ], [ -1.616 ], [ 0.972 ], [ -0.692 ], ... ],
    	[ [ 1.136 ], [ 1.14 ], [ 0.656 ], [ 1.932 ], [ 0.54 ], [ -0.288 ], [ 0.384 ], [ -0.54 ], ... ],
    	...
    ]
```



First, we use a conjugate gradient descent method, which converges the fastest for purely linear functions.

Code from [LearningTester.java:300](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L300) executed in 0.03 seconds: 
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
    Zero gradient: 2.7755575615628914E-19
    Constructing line search parameters: GD
    F(0.0) = LineSearchPoint{point=PointSample{avg=1.9259299443872359E-34}, derivative=-7.703719777548944E-38}
    F(1.0E-10) = LineSearchPoint{point=PointSample{avg=1.9259299443872359E-34}, derivative=-7.703719777548944E-38}, delta = 0.0
    F(7.000000000000001E-10) = LineSearchPoint{point=PointSample{avg=1.9259299443872359E-34}, derivative=-7.703719777548944E-38}, delta = 0.0
    F(4.900000000000001E-9) = LineSearchPoint{point=PointSample{avg=1.9259299443872359E-34}, derivative=-7.703719777548944E-38}, delta = 0.0
    F(3.430000000000001E-8) = LineSearchPoint{point=PointSample{avg=1.9259299443872359E-34}, derivative=-7.703719777548944E-38}, delta = 0.0
    F(2.4010000000000004E-7) = LineSearchPoint{point=PointSample{avg=1.9259299443872359E-34}, derivative=-7.703719777548944E-38}, delta = 0.0
    F(1.6807000000000003E-6) = LineSearchPoint{point=PointSample{avg=1.9259299443872359E-34}, derivative=-7.703719777548944E-38}, delta = 0.0
    F(1.17649000000000
```
...[skipping 2104 bytes](etc/232.txt)...
```
    racket at 0.01081350562578125
    F(0.005406752812890625) = LineSearchPoint{point=PointSample{avg=1.9259299443872359E-34}, derivative=-7.703719777548944E-38}, delta = 0.0
    Right bracket at 0.005406752812890625
    F(0.0027033764064453127) = LineSearchPoint{point=PointSample{avg=1.9259299443872359E-34}, derivative=-7.703719777548944E-38}, delta = 0.0
    Right bracket at 0.0027033764064453127
    F(0.0013516882032226563) = LineSearchPoint{point=PointSample{avg=1.9259299443872359E-34}, derivative=-7.703719777548944E-38}, delta = 0.0
    Right bracket at 0.0013516882032226563
    F(6.758441016113282E-4) = LineSearchPoint{point=PointSample{avg=1.9259299443872359E-34}, derivative=-7.703719777548944E-38}, delta = 0.0
    Right bracket at 6.758441016113282E-4
    F(3.379220508056641E-4) = LineSearchPoint{point=PointSample{avg=1.9259299443872359E-34}, derivative=-7.703719777548944E-38}, delta = 0.0
    Loops = 12
    Iteration 1 failed, aborting. Error: 1.9259299443872359E-34 Total: 249768014607295.0000; Orientation: 0.0003; Line Search: 0.0257
    
```

Returns: 

```
    1.9259299443872359E-34
```



This training run resulted in the following regressed input:

Code from [LearningTester.java:144](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L144) executed in 0.00 seconds: 
```java
    return Arrays.stream(input_gd).map(x -> x.prettyPrint()).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [
    	[ [ -1.916 ], [ 1.66 ], [ -0.884 ], [ -0.7 ], [ 0.448 ], [ -0.548 ], [ -0.652 ], [ -1.528 ], ... ],
    	[ [ -1.08 ], [ 0.62 ], [ -1.964 ], [ -1.14 ], [ 0.004 ], [ -1.284 ], [ -1.832 ], [ -0.708 ], ... ],
    	[ [ -0.056 ], [ -0.728 ], [ 1.388 ], [ -1.136 ], [ -0.884 ], [ 0.84 ], [ -1.712 ], [ 0.552 ], ... ],
    	[ [ -1.736 ], [ 1.556 ], [ -1.3 ], [ 0.948 ], [ 1.684 ], [ 1.14 ], [ -0.692 ], [ 1.86 ], ... ],
    	[ [ 1.18 ], [ -1.696 ], [ 1.8 ], [ 0.384 ], [ -1.7 ], [ 0.396 ], [ 0.096 ], [ -0.64 ], ... ],
    	[ [ -0.648 ], [ 1.76 ], [ 1.864 ], [ 1.096 ], [ 1.22 ], [ -0.516 ], [ -0.884 ], [ -1.308 ], ... ],
    	[ [ -0.172 ], [ -0.316 ], [ -1.844 ], [ 0.124 ], [ 1.46 ], [ 1.468 ], [ -1.092 ], [ 1.788 ], ... ],
    	[ [ 0.904 ], [ -1.36 ], [ 1.24 ], [ -1.092 ], [ -1.452 ], [ 1.384 ], [ -1.088 ], [ 1.372 ], ... ],
    	...
    ]
```



Next, we run the same optimization using L-BFGS, which is nearly ideal for purely second-order or quadratic functions.

Code from [LearningTester.java:324](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L324) executed in 0.02 seconds: 
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
    th(0)=1.9259299443872359E-34;dx=-7.703719777548944E-38
    Armijo: th(2.154434690031884)=1.9259299443872359E-34; dx=-7.703719777548944E-38 delta=0.0
    Armijo: th(1.077217345015942)=1.9259299443872359E-34; dx=-7.703719777548944E-38 delta=0.0
    Armijo: th(0.3590724483386473)=1.9259299443872359E-34; dx=-7.703719777548944E-38 delta=0.0
    Armijo: th(0.08976811208466183)=1.9259299443872359E-34; dx=-7.703719777548944E-38 delta=0.0
    Armijo: th(0.017953622416932366)=1.9259299443872359E-34; dx=-7.703719777548944E-38 delta=0.0
    Armijo: th(0.002992270402822061)=1.9259299443872359E-34; dx=-7.703719777548944E-38 delta=0.0
    Armijo: th(4.2746720040315154E-4)=1.9259299443872359E-34; dx=-7.703719777548944E-38 delta=0.0
    Armijo: th(5.343340005039394E-5)=1.9259299443872359E-34; dx=-7.703719777548944E-38 delta=0.0
    Armijo: th(5.9370444500437714E-6)=1.9259299443872359E-34; dx=-7.703719777548944E-38 delta=0.0
    Armijo: th(5.937044450043771E-7)=1.925929944387235
```
...[skipping 184 bytes](etc/233.txt)...
```
    99443872359E-34; dx=-7.703719777548944E-38 delta=0.0
    Armijo: th(1.8890595977412E-7)=1.9259299443872359E-34; dx=-7.703719777548944E-38 delta=0.0
    WOLFE (weak): th(1.2143954556907715E-7)=1.9259299443872359E-34; dx=-7.703719777548944E-38 delta=0.0
    Armijo: th(1.5517275267159856E-7)=1.9259299443872359E-34; dx=-7.703719777548944E-38 delta=0.0
    WOLFE (weak): th(1.3830614912033784E-7)=1.9259299443872359E-34; dx=-7.703719777548944E-38 delta=0.0
    Armijo: th(1.467394508959682E-7)=1.9259299443872359E-34; dx=-7.703719777548944E-38 delta=0.0
    Armijo: th(1.4252280000815304E-7)=1.9259299443872359E-34; dx=-7.703719777548944E-38 delta=0.0
    Armijo: th(1.4041447456424544E-7)=1.9259299443872359E-34; dx=-7.703719777548944E-38 delta=0.0
    Armijo: th(1.3936031184229164E-7)=1.9259299443872359E-34; dx=-7.703719777548944E-38 delta=0.0
    mu /= nu: th(0)=1.9259299443872359E-34;th'(0)=-7.703719777548944E-38;
    Iteration 1 failed, aborting. Error: 1.9259299443872359E-34 Total: 249768041717858.9700; Orientation: 0.0005; Line Search: 0.0183
    
```

Returns: 

```
    1.9259299443872359E-34
```



This training run resulted in the following regressed input:

Code from [LearningTester.java:154](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L154) executed in 0.00 seconds: 
```java
    return Arrays.stream(input_lbgfs).map(x -> x.prettyPrint()).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [
    	[ [ -1.916 ], [ 1.66 ], [ -0.884 ], [ -0.7 ], [ 0.448 ], [ -0.548 ], [ -0.652 ], [ -1.528 ], ... ],
    	[ [ -1.08 ], [ 0.62 ], [ -1.964 ], [ -1.14 ], [ 0.004 ], [ -1.284 ], [ -1.832 ], [ -0.708 ], ... ],
    	[ [ -0.056 ], [ -0.728 ], [ 1.388 ], [ -1.136 ], [ -0.884 ], [ 0.84 ], [ -1.712 ], [ 0.552 ], ... ],
    	[ [ -1.736 ], [ 1.556 ], [ -1.3 ], [ 0.948 ], [ 1.684 ], [ 1.14 ], [ -0.692 ], [ 1.86 ], ... ],
    	[ [ 1.18 ], [ -1.696 ], [ 1.8 ], [ 0.384 ], [ -1.7 ], [ 0.396 ], [ 0.096 ], [ -0.64 ], ... ],
    	[ [ -0.648 ], [ 1.76 ], [ 1.864 ], [ 1.096 ], [ 1.22 ], [ -0.516 ], [ -0.884 ], [ -1.308 ], ... ],
    	[ [ -0.172 ], [ -0.316 ], [ -1.844 ], [ 0.124 ], [ 1.46 ], [ 1.468 ], [ -1.092 ], [ 1.788 ], ... ],
    	[ [ 0.904 ], [ -1.36 ], [ 1.24 ], [ -1.092 ], [ -1.452 ], [ 1.384 ], [ -1.088 ], [ 1.372 ], ... ],
    	...
    ]
```



Code from [LearningTester.java:96](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L96) executed in 0.00 seconds: 
```java
    return TestUtil.compare(runs);
```

Code from [LearningTester.java:99](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L99) executed in 0.00 seconds: 
```java
    return TestUtil.compareTime(runs);
```

