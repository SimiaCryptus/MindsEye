# DropoutNoiseLayer
## DropoutNoiseLayerTest
### Batch Execution
Code from [BatchingTester.java:66](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/BatchingTester.java#L66) executed in 0.00 seconds: 
```java
    return test(reference, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (60#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (20#)}
```



Code from [SingleDerivativeTester.java:77](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/SingleDerivativeTester.java#L77) executed in 0.00 seconds: 
```java
    return test(component, inputPrototype);
```
Logging: 
```
    Inputs: [ 0.032, 0.608, 1.772 ]
    Inputs Statistics: {meanExponent=-0.4874942416187757, negative=0, min=1.772, max=1.772, mean=0.8039999999999999, count=3.0, positive=3, stdDev=0.7237458117322684, zeros=0}
    Output: [ 0.0, 0.608, 0.0 ]
    Outputs Statistics: {meanExponent=-0.21609642072726507, negative=0, min=0.0, max=0.0, mean=0.20266666666666666, count=3.0, positive=1, stdDev=0.28661394864094725, zeros=2}
    Feedback for input 0
    Inputs Values: [ 0.032, 0.608, 1.772 ]
    Value Statistics: {meanExponent=-0.4874942416187757, negative=0, min=1.772, max=1.772, mean=0.8039999999999999, count=3.0, positive=3, stdDev=0.7237458117322684, zeros=0}
    Implemented Feedback: [ [ 0.0, 0.0, 0.0 ], [ 0.0, 1.0, 0.0 ], [ 0.0, 0.0, 0.0 ] ]
    Implemented Statistics: {meanExponent=0.0, negative=0, min=0.0, max=0.0, mean=0.1111111111111111, count=9.0, positive=1, stdDev=0.31426968052735443, zeros=8}
    Measured Feedback: [ [ 0.0, 0.0, 0.0 ], [ 0.0, 0.9999999999998899, 0.0 ], [ 0.0, 0.0, 0.0 ] ]
    Measured Statistics: {meanExponent=-4.7830642341045674E-14, negative=0, min=0.0, max=0.0, mean=0.11111111111109888, count=9.0, positive=1, stdDev=0.31426968052731985, zeros=8}
    Feedback Error: [ [ 0.0, 0.0, 0.0 ], [ 0.0, -1.1013412404281553E-13, 0.0 ], [ 0.0, 0.0, 0.0 ] ]
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
      "class": "com.simiacryptus.mindseye.layers.java.DropoutNoiseLayer",
      "id": "7c4a571f-d864-4b80-9b5e-caf790320eb8",
      "isFrozen": false,
      "name": "DropoutNoiseLayer/7c4a571f-d864-4b80-9b5e-caf790320eb8",
      "value": 0.5
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
    [[ 1.832, -1.384, -0.036 ]]
    --------------------
    Output: 
    [ 0.0, -1.384, -0.036 ]
    --------------------
    Derivative: 
    [ 0.0, 1.0, 1.0 ]
```



### Input Learning
In this test, we use a network to learn this target input, given it's pre-evaluated output:

Code from [LearningTester.java:127](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L127) executed in 0.00 seconds: 
```java
    return Arrays.stream(input_target).map(x -> x.prettyPrint()).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [ 0.896, 1.86, 0.032 ]
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
    F(0.0) = LineSearchPoint{point=PointSample{avg=0.49766399999999994}, derivative=-0.6635519999999999}
    New Minimum: 0.49766399999999994 > 0.4976639999336448
    F(1.0E-10) = LineSearchPoint{point=PointSample{avg=0.4976639999336448}, derivative=-0.6635519999557631}, delta = -6.635514360198158E-11
    New Minimum: 0.4976639999336448 > 0.4976639995355136
    F(7.000000000000001E-10) = LineSearchPoint{point=PointSample{avg=0.4976639995355136}, derivative=-0.6635519996903423}, delta = -4.6448633828077845E-10
    New Minimum: 0.4976639995355136 > 0.4976639967485952
    F(4.900000000000001E-9) = LineSearchPoint{point=PointSample{avg=0.4976639967485952}, derivative=-0.6635519978323967}, delta = -3.251404756543508E-9
    New Minimum: 0.4976639967485952 > 0.4976639772401667
    F(3.430000000000001E-8) = LineSearchPoint{point=PointSample{avg=0.4976639772401667}, derivative=-0.6635519848267775}, delta = -2.2759833240293403E-8
    New Minimum: 0.4976639772401667 > 0.4976638406811776
    F(2.4010000000000004E-
```
...[skipping 1606 bytes](etc/218.txt)...
```
    29696503369968197
    F(1.3841287201) = LineSearchPoint{point=PointSample{avg=0.0029696503369968197}, derivative=-0.051257746346803204}, delta = -0.49469434966300313
    Loops = 12
    New Minimum: 0.0029696503369968197 > 6.419766481290787E-33
    F(1.5000000000000002) = LineSearchPoint{point=PointSample{avg=6.419766481290787E-33}, derivative=7.460698725481051E-17}, delta = -0.49766399999999994
    Right bracket at 1.5000000000000002
    Converged to right
    Iteration 1 complete. Error: 6.419766481290787E-33 Total: 239637210263924.7800; Orientation: 0.0000; Line Search: 0.0013
    Zero gradient: 9.251858538542972E-17
    F(0.0) = LineSearchPoint{point=PointSample{avg=6.419766481290787E-33}, derivative=-8.559688641721048E-33}
    New Minimum: 6.419766481290787E-33 > 0.0
    F(1.5000000000000002) = LineSearchPoint{point=PointSample{avg=0.0}, derivative=0.0}, delta = -6.419766481290787E-33
    0.0 <= 6.419766481290787E-33
    Converged to right
    Iteration 2 complete. Error: 0.0 Total: 239637210483357.7800; Orientation: 0.0000; Line Search: 0.0001
    
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
    th(0)=0.49766399999999994;dx=-0.6635519999999999
    New Minimum: 0.49766399999999994 > 0.09472973713377238
    WOLF (strong): th(2.154434690031884)=0.09472973713377238; dx=0.2895009649600243 delta=0.4029342628662276
    New Minimum: 0.09472973713377238 > 0.03953557242343397
    END: th(1.077217345015942)=0.03953557242343397; dx=-0.1870255175199878 delta=0.45812842757656597
    Iteration 1 complete. Error: 0.03953557242343397 Total: 239637213431744.7800; Orientation: 0.0001; Line Search: 0.0002
    LBFGS Accumulation History: 1 points
    th(0)=0.03953557242343397;dx=-0.05271409656457862
    New Minimum: 0.03953557242343397 > 0.011837890006382653
    WOLF (strong): th(2.3207944168063896)=0.011837890006382653; dx=0.028844957431466027 delta=0.02769768241705132
    New Minimum: 0.011837890006382653 > 0.0020265065706543974
    END: th(1.1603972084031948)=0.0020265065706543974; dx=-0.011934569566556308 delta=0.03750906585277958
    Iteration 2 complete. Error: 0.002026506
```
...[skipping 9690 bytes](etc/219.txt)...
```
    791300.7800; Orientation: 0.0000; Line Search: 0.0003
    LBFGS Accumulation History: 1 points
    th(0)=1.9299422984380427E-31;dx=-2.5732563979173896E-31
    New Minimum: 1.9299422984380427E-31 > 1.3564966574967432E-31
    WOLF (strong): th(2.8257016782407427)=1.3564966574967432E-31; dx=2.1549016155532735E-31 delta=5.734456409412995E-32
    New Minimum: 1.3564966574967432E-31 > 2.5679065925163143E-34
    END: th(1.4128508391203713)=2.5679065925163143E-34; dx=-6.419766481290785E-33 delta=1.9273743918455263E-31
    Iteration 21 complete. Error: 2.5679065925163143E-34 Total: 239637222167756.7800; Orientation: 0.0000; Line Search: 0.0003
    LBFGS Accumulation History: 1 points
    th(0)=2.5679065925163143E-34;dx=-3.423875456688419E-34
    Armijo: th(3.043894859641584)=2.5679065925163143E-34; dx=3.423875456688419E-34 delta=0.0
    New Minimum: 2.5679065925163143E-34 > 0.0
    END: th(1.521947429820792)=0.0; dx=0.0 delta=2.5679065925163143E-34
    Iteration 22 complete. Error: 0.0 Total: 239637222490922.7800; Orientation: 0.0000; Line Search: 0.0002
    
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

![Result](etc/test.130.png)



Code from [LearningTester.java:98](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L98) executed in 0.00 seconds: 
```java
    return TestUtil.compareTime(runs);
```

Returns: 

![Result](etc/test.131.png)



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
    	[3]
    Performance:
    	Evaluation performance: 0.000766s +- 0.000421s [0.000341s - 0.001539s]
    	Learning performance: 0.000036s +- 0.000007s [0.000027s - 0.000046s]
    
```

