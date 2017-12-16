# EntropyLossLayer
## EntropyLossLayerTest
### Batch Execution
Code from [BatchingTester.java:66](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/BatchingTester.java#L66) executed in 0.00 seconds: 
```java
    return test(reference, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (90#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (90#)}
```



Code from [SingleDerivativeTester.java:77](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/SingleDerivativeTester.java#L77) executed in 0.00 seconds: 
```java
    return test(component, inputPrototype);
```
Logging: 
```
    Inputs: [ 0.07183034489573148, 0.8875597801359891, 0.5083415144325021, 0.2724862697675823 ],
    [ 0.6754930264699268, 0.6302018261052557, 0.13627979634936238, 0.1687119614641338 ]
    Inputs Statistics: {meanExponent=-0.5134985580812935, negative=0, min=0.2724862697675823, max=0.2724862697675823, mean=0.4350544773079512, count=4.0, positive=4, stdDev=0.303517804818551, zeros=0},
    {meanExponent=-0.5023305305799719, negative=0, min=0.1687119614641338, max=0.1687119614641338, mean=0.40267165259716964, count=4.0, positive=4, stdDev=0.25094981572450714, zeros=0}
    Output: [ 2.16560692013699 ]
    Outputs Statistics: {meanExponent=0.33557963054066714, negative=0, min=2.16560692013699, max=2.16560692013699, mean=2.16560692013699, count=1.0, positive=1, stdDev=0.0, zeros=0}
    Feedback for input 0
    Inputs Values: [ 0.07183034489573148, 0.8875597801359891, 0.5083415144325021, 0.2724862697675823 ]
    Value Statistics: {meanExponent=-0.5134985580812935, negative=0, min=0.2724862697675823, max=0.2724862697675823, mean=0.435054477307951
```
...[skipping 1508 bytes](etc/223.txt)...
```
    ed Statistics: {meanExponent=-0.1396443652871546, negative=0, min=1.3001670523222741, max=1.3001670523222741, mean=1.1823741251119237, count=4.0, positive=4, stdDev=0.936144670059333, zeros=0}
    Measured Feedback: [ [ 2.6334483038681356 ], [ 0.11927943077694181 ], [ 0.6766017968118376 ], [ 1.300167085105386 ] ]
    Measured Statistics: {meanExponent=-0.13964433267178597, negative=0, min=1.300167085105386, max=1.300167085105386, mean=1.1823741541405752, count=4.0, positive=4, stdDev=0.9361446777707909, zeros=0}
    Feedback Error: [ [ 4.251667995447406E-8 ], [ 2.877564021785961E-8 ], [ 1.2039174324129931E-8 ], [ 3.278311178078752E-8 ] ]
    Error Statistics: {meanExponent=-7.579042195922408, negative=0, min=3.278311178078752E-8, max=3.278311178078752E-8, mean=2.902865156931278E-8, count=4.0, positive=4, stdDev=1.1008283619699911E-8, zeros=0}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 1.1042e-07 +- 2.2134e-07 [1.0083e-09 - 6.9484e-07] (8#)
    relativeTol: 3.3918e-08 +- 3.6278e-08 [7.1003e-10 - 1.2062e-07] (8#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=1.1042e-07 +- 2.2134e-07 [1.0083e-09 - 6.9484e-07] (8#), relativeTol=3.3918e-08 +- 3.6278e-08 [7.1003e-10 - 1.2062e-07] (8#)}
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
      "class": "com.simiacryptus.mindseye.layers.java.EntropyLossLayer",
      "id": "3d413494-7da5-4c32-968e-7221366319c3",
      "isFrozen": false,
      "name": "EntropyLossLayer/3d413494-7da5-4c32-968e-7221366319c3"
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
    [[ 0.36744383545134107, 0.8636529618111257, 0.5091088650002741, 0.5635183266958179 ],
    [ 0.13196220076158693, 0.5007433703718884, 0.3215189677351382, 0.6878052146694149 ]]
    --------------------
    Output: 
    [ 0.817069389409297 ]
    --------------------
    Derivative: 
    [ -0.35913570464311706, -0.5797969699794733, -0.6315328406920692, -1.2205551835418582 ],
    [ 1.0011848007125759, 0.14658425543735865, 0.6750934051430051, 0.5735554230572216 ]
```



### Input Learning
In this test, we use a network to learn this target input, given it's pre-evaluated output:

Code from [LearningTester.java:127](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L127) executed in 0.00 seconds: 
```java
    return Arrays.stream(input_target).map(x -> x.prettyPrint()).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [ 0.33766735858148234, 0.9397890718484745, 0.44620604824392485, 0.0014884140009894242 ]
    [ 0.08402489416555492, 0.19671421931821953, 0.8943761808352251, 0.36707166708575123 ]
```



First, we use a conjugate gradient descent method, which converges the fastest for purely linear functions.

Code from [LearningTester.java:225](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L225) executed in 0.01 seconds: 
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
    F(0.0) = LineSearchPoint{point=PointSample{avg=5.8568938944289695}, derivative=-732.9280862431061}
    New Minimum: 5.8568938944289695 > 5.856893821136172
    F(1.0E-10) = LineSearchPoint{point=PointSample{avg=5.856893821136172}, derivative=-732.9278503446409}, delta = -7.329279760881491E-8
    New Minimum: 5.856893821136172 > 5.856893381379884
    F(7.000000000000001E-10) = LineSearchPoint{point=PointSample{avg=5.856893381379884}, derivative=-732.9264349626447}, delta = -5.130490858817893E-7
    New Minimum: 5.856893381379884 > 5.8568903031096635
    F(4.900000000000001E-9) = LineSearchPoint{point=PointSample{avg=5.8568903031096635}, derivative=-732.916527710786}, delta = -3.591319305940033E-6
    New Minimum: 5.8568903031096635 > 5.85686875638299
    F(3.430000000000001E-8) = LineSearchPoint{point=PointSample{avg=5.85686875638299}, derivative=-732.8471976229317}, delta = -2.5138045979034018E-5
    New Minimum: 5.85686875638299 > 5.856717986294502
    F(2.4010000000000004E-7) = LineSearchPoint{poi
```
...[skipping 9516 bytes](etc/224.txt)...
```
    042491866 Total: 239645030710147.9700; Orientation: 0.0000; Line Search: 0.0006
    F(0.0) = LineSearchPoint{point=PointSample{avg=3.039147042491866}, derivative=-1.338823533117947E-10}
    New Minimum: 3.039147042491866 > 3.039147042484804
    F(0.10551540044824143) = LineSearchPoint{point=PointSample{avg=3.039147042484804}, derivative=4.835450765724894E-15}, delta = -7.061906615035696E-12
    3.039147042484804 <= 3.039147042491866
    Converged to right
    Iteration 5 complete. Error: 3.039147042484804 Total: 239645030917896.9700; Orientation: 0.0000; Line Search: 0.0001
    Low gradient: 4.1792044773868164E-10
    F(0.0) = LineSearchPoint{point=PointSample{avg=3.039147042484804}, derivative=-1.7465750063810014E-19}
    F(0.10551540044824143) = LineSearchPoint{point=PointSample{avg=3.039147042484804}, derivative=6.4536548939645405E-24}, delta = 0.0
    3.039147042484804 <= 3.039147042484804
    Converged to right
    Iteration 6 failed, aborting. Error: 3.039147042484804 Total: 239645031333679.9700; Orientation: 0.0000; Line Search: 0.0003
    
```

Returns: 

```
    3.039147042484804
```



This training run resulted in the following regressed input:

Code from [LearningTester.java:144](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L144) executed in 0.00 seconds: 
```java
    return Arrays.stream(input_gd).map(x -> x.prettyPrint()).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [ 0.36787944117144233, 0.36787944117144367, 0.3678794411714429, 0.367879441171443 ]
    [ 0.19671421931821953, 0.08402489416555492, 0.8943761808352251, 0.36707166708575123 ]
```



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
    th(0)=5.8568938944289695;dx=-732.9280862431061
    Armijo: th(2.154434690031884)=10.335149434294765; dx=10828.813988525593 delta=-4.4782555398657955
    Armijo: th(1.077217345015942)=9.147034076542727; dx=5551.607932982503 delta=-3.290140182113758
    Armijo: th(0.3590724483386473)=6.867801646134567; dx=2002.7590171544475 delta=-1.0109077517055978
    New Minimum: 5.8568938944289695 > 4.594731613963157
    WOLF (strong): th(0.08976811208466183)=4.594731613963157; dx=266.8036273421695 delta=1.2621622804658124
    New Minimum: 4.594731613963157 > 4.005172224850727
    WOLF (strong): th(0.017953622416932366)=4.005172224850727; dx=12.388335860891189 delta=1.8517216695782421
    END: th(0.002992270402822061)=4.895142008707; dx=-197.63699174236646 delta=0.9617518857219691
    Iteration 1 complete. Error: 4.005172224850727 Total: 239645036953738.9700; Orientation: 0.0001; Line Search: 0.0005
    LBFGS Accumulation History: 1 points
    th(0)=4.895142008707;dx=-62.1368670
```
...[skipping 6273 bytes](etc/225.txt)...
```
    . Error: 3.039147042484814 Total: 239645042625093.9700; Orientation: 0.0000; Line Search: 0.0010
    LBFGS Accumulation History: 1 points
    th(0)=3.039147042484814;dx=-1.8980320753955185E-13
    New Minimum: 3.039147042484814 > 3.039147042484811
    WOLF (strong): th(0.19480927101706144)=3.039147042484811; dx=1.6063665280806318E-13 delta=3.1086244689504383E-15
    New Minimum: 3.039147042484811 > 3.039147042484804
    END: th(0.09740463550853072)=3.039147042484804; dx=-1.4583287413040383E-14 delta=1.021405182655144E-14
    Iteration 16 complete. Error: 3.039147042484804 Total: 239645043035177.9700; Orientation: 0.0000; Line Search: 0.0003
    LBFGS Accumulation History: 1 points
    th(0)=3.039147042484804;dx=-1.12048829167982E-15
    WOLF (strong): th(0.20985192570949002)=3.039147042484804; dx=1.1080526440671697E-15 delta=0.0
    END: th(0.10492596285474501)=3.039147042484804; dx=-6.217824834743824E-18 delta=0.0
    Iteration 17 failed, aborting. Error: 3.039147042484804 Total: 239645043332124.9700; Orientation: 0.0000; Line Search: 0.0002
    
```

Returns: 

```
    3.039147042484804
```



This training run resulted in the following regressed input:

Code from [LearningTester.java:154](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L154) executed in 0.00 seconds: 
```java
    return Arrays.stream(input_lbgfs).map(x -> x.prettyPrint()).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [ 0.36787944117213733, 0.36787944119101007, 0.367879441170603, 0.3678794411712483 ]
    [ 0.19671421931821953, 0.08402489416555492, 0.8943761808352251, 0.36707166708575123 ]
```



Code from [LearningTester.java:95](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L95) executed in 0.00 seconds: 
```java
    return TestUtil.compare(runs);
```

Returns: 

![Result](etc/test.136.png)



Code from [LearningTester.java:98](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L98) executed in 0.00 seconds: 
```java
    return TestUtil.compareTime(runs);
```

Returns: 

![Result](etc/test.137.png)



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
    	[4]
    Performance:
    	Evaluation performance: 0.000216s +- 0.000020s [0.000194s - 0.000247s]
    	Learning performance: 0.000025s +- 0.000002s [0.000024s - 0.000027s]
    
```

