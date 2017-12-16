# GaussianActivationLayer
## GaussianActivationLayerTest
### Batch Execution
Code from [BatchingTester.java:66](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/BatchingTester.java#L66) executed in 0.00 seconds: 
```java
    return test(reference, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (120#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (120#)}
```



Code from [SingleDerivativeTester.java:77](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/SingleDerivativeTester.java#L77) executed in 0.00 seconds: 
```java
    return test(component, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ 1.76 ], [ 0.26 ], [ 0.956 ] ],
    	[ [ 1.184 ], [ 0.72 ], [ 0.74 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.09319002889829779, negative=0, min=0.74, max=0.74, mean=0.9366666666666665, count=6.0, positive=6, stdDev=0.4624077085093727, zeros=0}
    Output: [
    	[ [ 0.08477636130802224 ], [ 0.3856833691918161 ], [ 0.25261049217640646 ] ],
    	[ [ 0.19792511375148217 ], [ 0.30785126046985295 ], [ 0.30338928375630014 ] ]
    ]
    Outputs Statistics: {meanExponent=-0.636033475734452, negative=0, min=0.30338928375630014, max=0.30338928375630014, mean=0.2553726467756467, count=6.0, positive=6, stdDev=0.09526404346408747, zeros=0}
    Feedback for input 0
    Inputs Values: [
    	[ [ 1.76 ], [ 0.26 ], [ 0.956 ] ],
    	[ [ 1.184 ], [ 0.72 ], [ 0.74 ] ]
    ]
    Value Statistics: {meanExponent=-0.09319002889829779, negative=0, min=0.74, max=0.74, mean=0.9366666666666665, count=6.0, positive=6, stdDev=0.4624077085093727, zeros=0}
    Implemented Feedback: [ [ -0.14920639590211912, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, -0.2343433346817549, 0.0, 0.0
```
...[skipping 665 bytes](etc/231.txt)...
```
     0.0, 0.0, -0.2414967167141402, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, -0.22451493172759296 ] ]
    Measured Statistics: {meanExponent=-0.7292111111954633, negative=6, min=-0.22451493172759296, max=-0.22451493172759296, mean=-0.03254179127638251, count=36.0, positive=0, stdDev=0.07582364588977823, zeros=30}
    Feedback Error: [ [ 8.891320419562865E-6, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 3.97749378117096E-6, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, -1.7980068752129763E-5, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, -7.412141445423126E-6, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, -1.086193495591914E-6, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, -6.861747930853035E-6 ] ]
    Error Statistics: {meanExponent=-5.242391204189829, negative=4, min=-6.861747930853035E-6, max=-6.861747930853035E-6, mean=-5.686482617573337E-7, count=36.0, positive=2, stdDev=3.7628341832236373E-6, zeros=30}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 1.2836e-06 +- 3.5826e-06 [0.0000e+00 - 1.7980e-05] (36#)
    relativeTol: 2.7029e-05 +- 2.9241e-05 [2.2489e-06 - 8.9643e-05] (6#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=1.2836e-06 +- 3.5826e-06 [0.0000e+00 - 1.7980e-05] (36#), relativeTol=2.7029e-05 +- 2.9241e-05 [2.2489e-06 - 8.9643e-05] (6#)}
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
      "class": "com.simiacryptus.mindseye.layers.java.GaussianActivationLayer",
      "id": "80c0bf87-0bbe-456c-adf8-8103ccb574f2",
      "isFrozen": true,
      "name": "GaussianActivationLayer/80c0bf87-0bbe-456c-adf8-8103ccb574f2",
      "mean": 0.0,
      "stddev": 1.0
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
    	[ [ -1.888 ], [ -0.056 ], [ -0.036 ] ],
    	[ [ 0.92 ], [ 1.36 ], [ -0.14 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 0.06712420745634257 ], [ 0.39831722907406775 ], [ 0.3986838495443733 ] ],
    	[ [ 0.26128630124955315 ], [ 0.15822479037038303 ], [ 0.39505174083461125 ] ]
    ]
    --------------------
    Derivative: 
    [
    	[ [ 0.12673050367757474 ], [ 0.022305764828147796 ], [ 0.014352618583597439 ] ],
    	[ [ -0.24038339714958892 ], [ -0.21518571490372093 ], [ 0.05530724371684558 ] ]
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
    	[ [ 1.736 ], [ 0.188 ], [ 0.028 ], [ 1.048 ], [ -0.184 ], [ 0.256 ], [ 0.22 ], [ 1.764 ], ... ],
    	[ [ -1.796 ], [ 1.216 ], [ 0.232 ], [ -1.408 ], [ -0.324 ], [ 1.004 ], [ -1.152 ], [ 0.956 ], ... ],
    	[ [ 0.944 ], [ -0.324 ], [ -1.84 ], [ 0.152 ], [ -1.868 ], [ -0.064 ], [ -1.936 ], [ 0.76 ], ... ],
    	[ [ 0.424 ], [ -0.016 ], [ -1.884 ], [ -1.08 ], [ 1.428 ], [ 1.316 ], [ 1.0 ], [ -0.716 ], ... ],
    	[ [ -0.72 ], [ 1.248 ], [ 1.112 ], [ -1.0 ], [ -0.328 ], [ -0.58 ], [ 0.692 ], [ 0.916 ], ... ],
    	[ [ 1.488 ], [ -0.828 ], [ -0.304 ], [ -0.796 ], [ 0.44 ], [ 1.712 ], [ -1.456 ], [ 0.332 ], ... ],
    	[ [ 0.128 ], [ 1.948 ], [ -0.176 ], [ 0.732 ], [ 0.404 ], [ -0.404 ], [ -0.684 ], [ 1.548 ], ... ],
    	[ [ -0.58 ], [ 0.636 ], [ 0.308 ], [ 0.328 ], [ -0.528 ], [ 1.876 ], [ -0.728 ], [ -1.264 ], ... ],
    	...
    ]
```



First, we use a conjugate gradient descent method, which converges the fastest for purely linear functions.

Code from [LearningTester.java:225](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L225) executed in 4.42 seconds: 
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
    F(0.0) = LineSearchPoint{point=PointSample{avg=0.0267390704662899}, derivative=-2.874154638123844E-7}
    New Minimum: 0.0267390704662899 > 0.02673907046628987
    F(1.0E-10) = LineSearchPoint{point=PointSample{avg=0.02673907046628987}, derivative=-2.874154638123844E-7}, delta = -2.7755575615628914E-17
    New Minimum: 0.02673907046628987 > 0.026739070466289705
    F(7.000000000000001E-10) = LineSearchPoint{point=PointSample{avg=0.026739070466289705}, derivative=-2.874154638123843E-7}, delta = -1.942890293094024E-16
    New Minimum: 0.026739070466289705 > 0.026739070466288463
    F(4.900000000000001E-9) = LineSearchPoint{point=PointSample{avg=0.026739070466288463}, derivative=-2.8741546381238334E-7}, delta = -1.4363510381087963E-15
    New Minimum: 0.026739070466288463 > 0.02673907046627997
    F(3.430000000000001E-8) = LineSearchPoint{point=PointSample{avg=0.02673907046627997}, derivative=-2.8741546381237683E-7}, delta = -9.929557176491244E-15
    New Minimum: 0.02673907046627997 > 0.0267390704
```
...[skipping 374990 bytes](etc/232.txt)...
```
    radient: 6.67732042307762E-6
    F(0.0) = LineSearchPoint{point=PointSample{avg=0.037986368404507496}, derivative=-4.458660803244949E-11}
    New Minimum: 0.037986368404507496 > 0.037983254694418885
    F(151778.19114938975) = LineSearchPoint{point=PointSample{avg=0.037983254694418885}, derivative=1.3134080037251852E-11}, delta = -3.113710088611943E-6
    0.037983254694418885 <= 0.037986368404507496
    New Minimum: 0.037983254694418885 > 0.03798309392960819
    F(117241.75409136024) = LineSearchPoint{point=PointSample{avg=0.03798309392960819}, derivative=-3.9601543794117455E-12}, delta = -3.2744748993049355E-6
    Left bracket at 117241.75409136024
    New Minimum: 0.03798309392960819 > 0.03798307825103771
    F(125242.67528683126) = LineSearchPoint{point=PointSample{avg=0.03798307825103771}, derivative=4.526603205165098E-14}, delta = -3.2901534697885038E-6
    Right bracket at 125242.67528683126
    Converged to right
    Iteration 250 complete. Error: 0.03798307825103771 Total: 239650208644379.8000; Orientation: 0.0004; Line Search: 0.0101
    
```

Returns: 

```
    0.03798307825103771
```



This training run resulted in the following regressed input:

Code from [LearningTester.java:144](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L144) executed in 0.00 seconds: 
```java
    return Arrays.stream(input_gd).map(x -> x.prettyPrint()).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [
    	[ [ -13.204127262539755 ], [ 10.514392216722278 ], [ -8.976820056350814 ], [ 1.0491376349592325 ], [ -9.903818646128057 ], [ -7.0004275761492725 ], [ -9.818683622514223 ], [ -1.764 ], ... ],
    	[ [ -12.924418003246716 ], [ -7.687803301110344 ], [ 0.2320000000000003 ], [ 1.408 ], [ 0.3239999999999999 ], [ -1.011498430917878 ], [ -1.1520000000000195 ], [ 0.9561439219117411 ], ... ],
    	[ [ 0.9467964219974927 ], [ 9.56309662475643 ], [ 9.404102910769613 ], [ 0.1519999999999993 ], [ 1.868 ], [ 11.286402280965543 ], [ -1.936 ], [ 5.477749403623311 ], ... ],
    	[ [ -6.1363019636451 ], [ -12.103917403626419 ], [ 4.947038741873857 ], [ 1.0800129535707572 ], [ -8.148593827260381 ], [ -7.6258550884011935 ], [ -1.0067520728531938 ], [ 0.716 ], ... ],
    	[ [ -0.7199999999999999 ], [ 1.248 ], [ 1.1120000113685569 ], [ -0.9943061506313084 ], [ -0.3280000000000005 ], [ -0.58 ], [ -7.385544760084913 ], [ 0.9159971900203904 ], ... ],
    	[ [ 7.696759200832131 ], [ 6.340875195747955 ], [ -0.30400000000000005 ], [ -0.796 ], [ -0.44000000000000006 ], [ 13.248111009984598 ], [ 1.4560000000000002 ], [ -8.20916009734799 ], ... ],
    	[ [ 8.609220993922523 ], [ -11.689807198352064 ], [ -4.684867178733881 ], [ -0.732 ], [ -9.15693685215063 ], [ 0.4039999999999998 ], [ -6.872052477121015 ], [ 5.788828846688051 ], ... ],
    	[ [ -4.445804440278231 ], [ 0.6360000000000001 ], [ -10.871910041685993 ], [ -0.32800000000000007 ], [ 9.423146247670624 ], [ -1.876 ], [ -5.376737551054142 ], [ -1.264 ], ... ],
    	...
    ]
```



Next, we run the same optimization using L-BFGS, which is nearly ideal for purely second-order or quadratic functions.

Code from [LearningTester.java:249](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L249) executed in 4.82 seconds: 
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
    th(0)=0.0267390704662899;dx=-2.874154638123844E-7
    New Minimum: 0.0267390704662899 > 0.026738451248957226
    WOLFE (weak): th(2.154434690031884)=0.026738451248957226; dx=-2.8741498734838093E-7 delta=6.192173326730055E-7
    New Minimum: 0.026738451248957226 > 0.026737832032651697
    WOLFE (weak): th(4.308869380063768)=0.026737832032651697; dx=-2.8741451059703774E-7 delta=1.238433638202363E-6
    New Minimum: 0.026737832032651697 > 0.02673535517770948
    WOLFE (weak): th(12.926608140191302)=0.02673535517770948; dx=-2.874126007183092E-7 delta=3.7152885804178637E-6
    New Minimum: 0.02673535517770948 > 0.02672420953496528
    WOLFE (weak): th(51.70643256076521)=0.02672420953496528; dx=-2.8740394937472634E-7 delta=1.4860931324620075E-5
    New Minimum: 0.02672420953496528 > 0.02666477189170491
    WOLFE (weak): th(258.53216280382605)=0.02666477189170491; dx=-2.8735623711159713E-7 delta=7.429857458498765E-5
    New Minimum: 0.02666477189170491 > 0.02629353806696
```
...[skipping 325654 bytes](etc/233.txt)...
```
     > 1.4239658744564227E-4
    WOLFE (weak): th(4.308869380063768)=1.4239658744564227E-4; dx=-1.1226874629481323E-9 delta=4.837574686292851E-9
    New Minimum: 1.4239658744564227E-4 > 1.4238691266259281E-4
    WOLFE (weak): th(12.926608140191302)=1.4238691266259281E-4; dx=-1.1226307854189121E-9 delta=1.4512357735750039E-8
    New Minimum: 1.4238691266259281E-4 > 1.4234338218335207E-4
    WOLFE (weak): th(51.70643256076521)=1.4234338218335207E-4; dx=-1.122375727167785E-9 delta=5.804283697649778E-8
    New Minimum: 1.4234338218335207E-4 > 1.4211138669685249E-4
    WOLFE (weak): th(258.53216280382605)=1.4211138669685249E-4; dx=-1.1210151576822756E-9 delta=2.9003832347607823E-7
    New Minimum: 1.4211138669685249E-4 > 1.40667794977533E-4
    WOLFE (weak): th(1551.1929768229563)=1.40667794977533E-4; dx=-1.1125017528950784E-9 delta=1.733630042795569E-6
    MAX ALPHA: th(0)=1.4240142502032856E-4;th'(0)=-1.1227158014287853E-9;
    Iteration 250 complete. Error: 1.40667794977533E-4 Total: 239655047439508.9700; Orientation: 0.0011; Line Search: 0.0213
    
```

Returns: 

```
    1.40667794977533E-4
```



This training run resulted in the following regressed input:

Code from [LearningTester.java:154](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L154) executed in 0.01 seconds: 
```java
    return Arrays.stream(input_lbgfs).map(x -> x.prettyPrint()).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [
    	[ [ -1.665340659538596 ], [ -0.36226463121152186 ], [ 0.29840228585171197 ], [ 1.0179531750975908 ], [ 0.3338911474419736 ], [ 0.35486965174200935 ], [ 0.3476733637754483 ], [ -1.753507150870043 ], ... ],
    	[ [ -1.7044071370036742 ], [ -1.2052191181684078 ], [ -0.31768202920263 ], [ -1.4571136614793199 ], [ 0.3782492492392642 ], [ -0.9888128131605612 ], [ -1.151316133331393 ], [ -0.9588707452465268 ], ... ],
    	[ [ -0.9601956940296558 ], [ -0.42547820891725124 ], [ 1.7618632765181679 ], [ 0.0882207002407698 ], [ 1.8278687641010358 ], [ -0.311179540964244 ], [ -1.9119992832042596 ], [ -0.7819101508270785 ], ... ],
    	[ [ 0.4613635092089228 ], [ 0.3203393982658008 ], [ 1.814397527634399 ], [ -1.0813960090508004 ], [ -1.4014497779636201 ], [ -1.2984801102384067 ], [ -0.973266730450466 ], [ 0.6879767683680498 ], ... ],
    	[ [ -0.7156857530020899 ], [ 1.2477332010304998 ], [ 1.0487784782620269 ], [ 1.0059864123159752 ], [ -0.24758724675968238 ], [ 0.5835869761672792 ], [ 0.7110782925975067 ], [ -0.9177501961370952 ], ... ],
    	[ [ 1.4724467892058446 ], [ -0.8375035966295576 ], [ -0.2996156730500029 ], [ 0.8006009781476637 ], [ -0.4394407005953018 ], [ 1.6502734537730999 ], [ 1.3867976220268494 ], [ 0.4373313391986732 ], ... ],
    	[ [ -0.3577960390628962 ], [ -1.8251020843308965 ], [ 0.3121457429225856 ], [ -0.7024668365314078 ], [ 0.4542156682477593 ], [ 0.4110979009310585 ], [ 0.6948671651349304 ], [ 1.5315367898261985 ], ... ],
    	[ [ 0.592890478516086 ], [ -0.643010207008009 ], [ 0.39767253746281794 ], [ 0.3746835527614579 ], [ -0.557909316126273 ], [ -1.8473215386401385 ], [ 0.7534281202291204 ], [ -1.2607550481337555 ], ... ],
    	...
    ]
```



Code from [LearningTester.java:95](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L95) executed in 0.01 seconds: 
```java
    return TestUtil.compare(runs);
```

Returns: 

![Result](etc/test.142.png)



Code from [LearningTester.java:98](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L98) executed in 0.00 seconds: 
```java
    return TestUtil.compareTime(runs);
```

Returns: 

![Result](etc/test.143.png)



### Performance
Now we execute larger-scale runs to benchmark performance:

Code from [PerformanceTester.java:66](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/PerformanceTester.java#L66) executed in 0.29 seconds: 
```java
    test(component, inputPrototype);
```
Logging: 
```
    100 batches
    Input Dimensions:
    	[100, 100, 1]
    Performance:
    	Evaluation performance: 0.008654s +- 0.000600s [0.007757s - 0.009321s]
    	Learning performance: 0.037916s +- 0.041289s [0.012497s - 0.119479s]
    
```

### Function Plots
Code from [ActivationLayerTestBase.java:110](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/java/ActivationLayerTestBase.java#L110) executed in 0.00 seconds: 
```java
    return plot("Value Plot", plotData, x -> new double[]{x[0], x[1]});
```

Returns: 

![Result](etc/test.144.png)



Code from [ActivationLayerTestBase.java:114](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/java/ActivationLayerTestBase.java#L114) executed in 0.00 seconds: 
```java
    return plot("Derivative Plot", plotData, x -> new double[]{x[0], x[2]});
```

Returns: 

![Result](etc/test.145.png)



