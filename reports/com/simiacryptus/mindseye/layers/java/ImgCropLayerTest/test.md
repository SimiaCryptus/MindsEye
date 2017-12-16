# ImgCropLayer
## ImgCropLayerTest
Code from [SingleDerivativeTester.java:77](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/SingleDerivativeTester.java#L77) executed in 0.00 seconds: 
```java
    return test(component, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ -1.928 ], [ 0.044 ], [ -0.908 ] ],
    	[ [ -0.028 ], [ 0.592 ], [ 0.868 ] ],
    	[ [ -1.352 ], [ 1.232 ], [ -1.996 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.2704007828665454, negative=5, min=-1.996, max=-1.996, mean=-0.38622222222222224, count=9.0, positive=4, stdDev=1.1370278778735938, zeros=0}
    Output: [
    	[ [ 0.592 ] ]
    ]
    Outputs Statistics: {meanExponent=-0.22767829327708025, negative=0, min=0.592, max=0.592, mean=0.592, count=1.0, positive=1, stdDev=0.0, zeros=0}
    Feedback for input 0
    Inputs Values: [
    	[ [ -1.928 ], [ 0.044 ], [ -0.908 ] ],
    	[ [ -0.028 ], [ 0.592 ], [ 0.868 ] ],
    	[ [ -1.352 ], [ 1.232 ], [ -1.996 ] ]
    ]
    Value Statistics: {meanExponent=-0.2704007828665454, negative=5, min=-1.996, max=-1.996, mean=-0.38622222222222224, count=9.0, positive=4, stdDev=1.1370278778735938, zeros=0}
    Implemented Feedback: [ [ 0.0 ], [ 0.0 ], [ 0.0 ], [ 0.0 ], [ 1.0 ], [ 0.0 ], [ 0.0 ], [ 0.0 ], [ 0.0 ] ]
    Implemented Statistics: {meanExponent=0.0, negative=0, min=0.0, max=0.0, mean=0.1111111111111111, count=9.0, positive=1, stdDev=0.31426968052735443, zeros=8}
    Measured Feedback: [ [ 0.0 ], [ 0.0 ], [ 0.0 ], [ 0.0 ], [ 0.9999999999998899 ], [ 0.0 ], [ 0.0 ], [ 0.0 ], [ 0.0 ] ]
    Measured Statistics: {meanExponent=-4.7830642341045674E-14, negative=0, min=0.0, max=0.0, mean=0.11111111111109888, count=9.0, positive=1, stdDev=0.31426968052731985, zeros=8}
    Feedback Error: [ [ 0.0 ], [ 0.0 ], [ 0.0 ], [ 0.0 ], [ -1.1013412404281553E-13 ], [ 0.0 ], [ 0.0 ], [ 0.0 ], [ 0.0 ] ]
    Error Statistics: {meanExponent=-12.958078098036825, negative=1, min=0.0, max=0.0, mean=-1.223712489364617E-14, count=9.0, positive=0, stdDev=3.461181597809566E-14, zeros=8}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 1.2237e-14 +- 3.4612e-14 [0.0000e+00 - 1.1013e-13] (9#)
    relativeTol: 5.5067e-14 +- 0.0000e+00 [5.5067e-14 - 5.5067e-14] (1#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=1.2237e-14 +- 3.4612e-14 [0.0000e+00 - 1.1013e-13] (9#), relativeTol=5.5067e-14 +- 0.0000e+00 [5.5067e-14 - 5.5067e-14] (1#)}
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
      "class": "com.simiacryptus.mindseye.layers.java.ImgCropLayer",
      "id": "ff985210-21e2-469c-88b7-18bd4000e75b",
      "isFrozen": false,
      "name": "ImgCropLayer/ff985210-21e2-469c-88b7-18bd4000e75b",
      "sizeX": 1,
      "sizeY": 1
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
    	[ [ 0.976 ], [ -0.98 ], [ 0.972 ] ],
    	[ [ 1.388 ], [ 1.052 ], [ -1.112 ] ],
    	[ [ 0.716 ], [ 1.324 ], [ -1.168 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 1.052 ] ]
    ]
    --------------------
    Derivative: 
    [
    	[ [ 0.0 ], [ 0.0 ], [ 0.0 ] ],
    	[ [ 0.0 ], [ 1.0 ], [ 0.0 ] ],
    	[ [ 0.0 ], [ 0.0 ], [ 0.0 ] ]
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
    	[ [ -1.62 ], [ 1.12 ], [ -1.436 ] ],
    	[ [ 0.552 ], [ 1.188 ], [ 0.196 ] ],
    	[ [ 0.216 ], [ -0.336 ], [ -0.136 ] ]
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
    F(0.0) = LineSearchPoint{point=PointSample{avg=0.944784}, derivative=-3.779136}
    New Minimum: 0.944784 > 0.9447839996220863
    F(1.0E-10) = LineSearchPoint{point=PointSample{avg=0.9447839996220863}, derivative=-3.7791359992441724}, delta = -3.779137003334654E-10
    New Minimum: 0.9447839996220863 > 0.9447839973546047
    F(7.000000000000001E-10) = LineSearchPoint{point=PointSample{avg=0.9447839973546047}, derivative=-3.7791359947092094}, delta = -2.645395236200443E-9
    New Minimum: 0.9447839973546047 > 0.9447839814822335
    F(4.900000000000001E-9) = LineSearchPoint{point=PointSample{avg=0.9447839814822335}, derivative=-3.7791359629644665}, delta = -1.8517766431358496E-8
    New Minimum: 0.9447839814822335 > 0.9447838703756395
    F(3.430000000000001E-8) = LineSearchPoint{point=PointSample{avg=0.9447838703756395}, derivative=-3.77913574075127}, delta = -1.2962436046759507E-7
    New Minimum: 0.9447838703756395 > 0.9447830926296642
    F(2.4010000000000004E-7) = LineSearchPoint{point=PointSam
```
...[skipping 909 bytes](etc/258.txt)...
```
    w Minimum: 0.9426066592184493 > 0.9295953630701522
    F(0.004035360700000001) = LineSearchPoint{point=PointSample{avg=0.9295953630701522}, derivative=-3.748635646211289}, delta = -0.015188636929847732
    New Minimum: 0.9295953630701522 > 0.8410482200003825
    F(0.028247524900000005) = LineSearchPoint{point=PointSample{avg=0.8410482200003825}, derivative=-3.565633523479027}, delta = -0.10373577999961747
    New Minimum: 0.8410482200003825 > 0.34528278695917225
    F(0.19773267430000002) = LineSearchPoint{point=PointSample{avg=0.34528278695917225}, derivative=-2.28461866435319}, delta = -0.5995012130408277
    F(1.3841287201) = LineSearchPoint{point=PointSample{avg=2.9540886095824495}, derivative=6.682485349527668}, delta = 2.0093046095824496
    Loops = 12
    New Minimum: 0.34528278695917225 > 0.0
    F(0.5) = LineSearchPoint{point=PointSample{avg=0.0}, derivative=0.0}, delta = -0.944784
    Right bracket at 0.5
    Converged to right
    Iteration 1 complete. Error: 0.0 Total: 239662177530426.8000; Orientation: 0.0000; Line Search: 0.0015
    
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
    th(0)=0.944784;dx=-3.779136
    Armijo: th(2.154434690031884)=10.344077761555736; dx=12.50466739349667 delta=-9.399293761555736
    Armijo: th(1.077217345015942)=1.2591320162018502; dx=4.362765696748334 delta=-0.31434801620185027
    New Minimum: 0.944784 > 0.07505581327261289
    END: th(0.3590724483386473)=0.07505581327261289; dx=-1.0651687677505555 delta=0.869728186727387
    Iteration 1 complete. Error: 0.07505581327261289 Total: 239662181860388.8000; Orientation: 0.0001; Line Search: 0.0005
    LBFGS Accumulation History: 1 points
    th(0)=0.07505581327261289;dx=-0.30022325309045156
    New Minimum: 0.07505581327261289 > 0.022473494308992033
    WOLF (strong): th(0.7735981389354633)=0.022473494308992033; dx=0.1642810466213962 delta=0.05258231896362085
    New Minimum: 0.022473494308992033 > 0.003847196067726713
    END: th(0.3867990694677316)=0.003847196067726713; dx=-0.06797110323452775 delta=0.07120861720488618
    Iteration 2 complete. Error: 0.0038471960677
```
...[skipping 9155 bytes](etc/259.txt)...
```
    189133321.8000; Orientation: 0.0000; Line Search: 0.0002
    LBFGS Accumulation History: 1 points
    th(0)=2.1742978700154138E-29;dx=-8.697191480061655E-29
    New Minimum: 2.1742978700154138E-29 > 1.2621774483536189E-29
    WOLF (strong): th(0.8743830237895417)=1.2621774483536189E-29; dx=6.626431603856499E-29 delta=9.121204216617949E-30
    New Minimum: 1.2621774483536189E-29 > 4.437342591868191E-31
    END: th(0.43719151189477085)=4.437342591868191E-31; dx=-1.2424559257230936E-29 delta=2.129924444096732E-29
    Iteration 20 complete. Error: 4.437342591868191E-31 Total: 239662189451357.8000; Orientation: 0.0000; Line Search: 0.0002
    LBFGS Accumulation History: 1 points
    th(0)=4.437342591868191E-31;dx=-1.7749370367472766E-30
    Armijo: th(0.9419005594135813)=4.437342591868191E-31; dx=1.7749370367472766E-30 delta=0.0
    New Minimum: 4.437342591868191E-31 > 0.0
    END: th(0.47095027970679065)=0.0; dx=0.0 delta=4.437342591868191E-31
    Iteration 21 complete. Error: 0.0 Total: 239662189714677.8000; Orientation: 0.0000; Line Search: 0.0002
    
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

![Result](etc/test.166.png)



Code from [LearningTester.java:98](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L98) executed in 0.00 seconds: 
```java
    return TestUtil.compareTime(runs);
```

Returns: 

![Result](etc/test.167.png)



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
    	[3, 3, 1]
    Performance:
    	Evaluation performance: 0.000281s +- 0.000026s [0.000243s - 0.000311s]
    	Learning performance: 0.000076s +- 0.000009s [0.000068s - 0.000091s]
    
```

