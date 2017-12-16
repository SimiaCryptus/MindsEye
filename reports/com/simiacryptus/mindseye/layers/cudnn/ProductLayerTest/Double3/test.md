# ProductLayer
## Double3
### Batch Execution
Code from [BatchingTester.java:66](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/BatchingTester.java#L66) executed in 0.03 seconds: 
```java
    return test(reference, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (160#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (160#)}
```



Code from [SingleDerivativeTester.java:77](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/SingleDerivativeTester.java#L77) executed in 0.05 seconds: 
```java
    return test(component, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ -1.116 ], [ 0.588 ] ],
    	[ [ 0.448 ], [ 0.908 ] ]
    ],
    [
    	[ [ -1.664 ], [ 2.0 ] ],
    	[ [ 0.364 ], [ 0.968 ] ]
    ],
    [
    	[ [ -0.008 ], [ -1.364 ] ],
    	[ [ 1.804 ], [ -0.396 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.1433986542007681, negative=1, min=0.908, max=0.908, mean=0.20699999999999996, count=4.0, positive=3, stdDev=0.781820311836422, zeros=0},
    {meanExponent=0.017290014644033985, negative=1, min=0.968, max=0.968, mean=0.41700000000000004, count=4.0, positive=3, stdDev=1.336306476823337, zeros=0},
    {meanExponent=-0.5270409808890403, negative=3, min=-0.396, max=-0.396, mean=0.00899999999999998, count=4.0, positive=1, stdDev=1.1479838849043136, zeros=0}
    Output: [
    	[ [ -0.014856192 ], [ -1.604064 ] ],
    	[ [ 0.294181888 ], [ -0.348061824 ] ]
    ]
    Outputs Statistics: {meanExponent=-0.6531496204457743, negative=3, min=-0.348061824, max=-0.348061824, mean=-0.41820003199999994, count=4.0, positive=1, stdDev=0.7213471723156244, zeros=0}
    Feedback for input 0
    Inputs Values: [
    	[ [ -1.116 ], [ 0.588 ] ],
    	[ [ 0.
```
...[skipping 3314 bytes](etc/137.txt)...
```
    ositive=4, stdDev=0.5357691452071498, zeros=12}
    Measured Feedback: [ [ 1.8570239999999953, 0.0, 0.0, 0.0 ], [ 0.0, 0.16307200000009736, 0.0, 0.0 ], [ 0.0, 0.0, 1.1759999999982895, 0.0 ], [ 0.0, 0.0, 0.0, 0.8789440000001036 ] ]
    Measured Statistics: {meanExponent=-0.12610863955681467, negative=0, min=0.8789440000001036, max=0.8789440000001036, mean=0.2546899999999054, count=16.0, positive=4, stdDev=0.5357691452069716, zeros=12}
    Feedback Error: [ [ -4.6629367034256575E-15, 0.0, 0.0, 0.0 ], [ 0.0, 9.736655925962623E-14, 0.0, 0.0 ], [ 0.0, 0.0, -1.7104095917375162E-12, 0.0 ], [ 0.0, 0.0, 0.0, 1.0369483049998962E-13 ] ]
    Error Statistics: {meanExponent=-13.023518356811199, negative=2, min=1.0369483049998962E-13, max=1.0369483049998962E-13, mean=-9.462569616758287E-14, count=16.0, positive=2, stdDev=4.1851607057089824E-13, zeros=12}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 1.1969e-13 +- 4.2468e-13 [0.0000e+00 - 2.3951e-12] (48#)
    relativeTol: 3.3847e-13 +- 3.9628e-13 [1.2555e-15 - 1.4931e-12] (12#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=1.1969e-13 +- 4.2468e-13 [0.0000e+00 - 2.3951e-12] (48#), relativeTol=3.3847e-13 +- 3.9628e-13 [1.2555e-15 - 1.4931e-12] (12#)}
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
      "class": "com.simiacryptus.mindseye.layers.cudnn.ProductLayer",
      "id": "3bec3976-7b4b-4bd8-9a3d-fb98e27400b3",
      "isFrozen": false,
      "name": "ProductLayer/3bec3976-7b4b-4bd8-9a3d-fb98e27400b3"
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
    	[ [ -0.412 ], [ -0.312 ] ],
    	[ [ 0.328 ], [ 1.516 ] ]
    ],
    [
    	[ [ 1.972 ], [ 1.348 ] ],
    	[ [ -1.724 ], [ 1.048 ] ]
    ],
    [
    	[ [ -0.404 ], [ -1.888 ] ],
    	[ [ -0.736 ], [ 1.484 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 0.328235456 ], [ 0.794047488 ] ],
    	[ [ 0.416187392 ], [ 2.357731712 ] ]
    ]
    --------------------
    Derivative: 
    [
    	[ [ -0.7966880000000001 ], [ -2.545024 ] ],
    	[ [ 1.268864 ], [ 1.555232 ] ]
    ],
    [
    	[ [ 0.166448 ], [ 0.5890559999999999 ] ],
    	[ [ -0.241408 ], [ 2.249744 ] ]
    ],
    [
    	[ [ -0.812464 ], [ -0.420576 ] ],
    	[ [ -0.565472 ], [ 1.5887680000000002 ] ]
    ]
```



[GPU Log](etc/cuda.log)

### Input Learning
In this test, we use a network to learn this target input, given it's pre-evaluated output:

Code from [LearningTester.java:127](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L127) executed in 0.00 seconds: 
```java
    return Arrays.stream(input_target).map(x -> x.prettyPrint()).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [
    	[ [ 0.928 ], [ -0.876 ] ],
    	[ [ 1.072 ], [ 0.052 ] ]
    ]
    [
    	[ [ -1.6 ], [ 1.416 ] ],
    	[ [ -0.9 ], [ -0.752 ] ]
    ]
    [
    	[ [ -0.936 ], [ -1.604 ] ],
    	[ [ -1.436 ], [ 1.86 ] ]
    ]
```



First, we use a conjugate gradient descent method, which converges the fastest for purely linear functions.

Code from [LearningTester.java:225](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L225) executed in 5.49 seconds: 
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
    F(0.0) = LineSearchPoint{point=PointSample{avg=1.293740053284649}, derivative=-7.912917341375033}
    New Minimum: 1.293740053284649 > 1.2937400524933578
    F(1.0E-10) = LineSearchPoint{point=PointSample{avg=1.2937400524933578}, derivative=-7.912917336133611}, delta = -7.912912547425321E-10
    New Minimum: 1.2937400524933578 > 1.293740047745607
    F(7.000000000000001E-10) = LineSearchPoint{point=PointSample{avg=1.293740047745607}, derivative=-7.912917304685059}, delta = -5.539042113866799E-9
    New Minimum: 1.293740047745607 > 1.2937400145113551
    F(4.900000000000001E-9) = LineSearchPoint{point=PointSample{avg=1.2937400145113551}, derivative=-7.912917084545216}, delta = -3.877329390888917E-8
    New Minimum: 1.2937400145113551 > 1.2937397818716154
    F(3.430000000000001E-8) = LineSearchPoint{point=PointSample{avg=1.2937397818716154}, derivative=-7.912915543566429}, delta = -2.714130336034515E-7
    New Minimum: 1.2937397818716154 > 1.2937381533947065
    F(2.4010000000000004E-7) = LineSearch
```
...[skipping 278526 bytes](etc/138.txt)...
```
    int=PointSample{avg=0.4841930171818337}, derivative=-1.2226262349390176E-11}, delta = -1.5400364650197673E-8
    Left bracket at 0.19675114163622992
    Converged to left
    Iteration 249 complete. Error: 0.4841930171818337 Total: 239587961982536.0600; Orientation: 0.0000; Line Search: 0.0211
    F(0.0) = LineSearchPoint{point=PointSample{avg=0.4841930171818337}, derivative=-1.8918581271747954E-7}
    New Minimum: 0.4841930171818337 > 0.48419300268925886
    F(0.19675114163622992) = LineSearchPoint{point=PointSample{avg=0.48419300268925886}, derivative=4.185502993960602E-8}, delta = -1.449257486241251E-8
    0.48419300268925886 <= 0.4841930171818337
    New Minimum: 0.48419300268925886 > 0.48419300194323434
    F(0.1611079850881099) = LineSearchPoint{point=PointSample{avg=0.48419300194323434}, derivative=5.314946358783391E-12}, delta = -1.523859938235006E-8
    Right bracket at 0.1611079850881099
    Converged to right
    Iteration 250 complete. Error: 0.48419300194323434 Total: 239587973796030.0300; Orientation: 0.0000; Line Search: 0.0096
    
```

Returns: 

```
    0.48419300194323434
```



This training run resulted in the following regressed input:

Code from [LearningTester.java:144](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L144) executed in 0.00 seconds: 
```java
    return Arrays.stream(input_gd).map(x -> x.prettyPrint()).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [
    	[ [ -0.010312868119419841 ], [ 1.2577158087247056 ] ],
    	[ [ 1.1148007312123673 ], [ 0.04144117020615379 ] ]
    ]
    [
    	[ [ -0.752 ], [ 1.416 ] ],
    	[ [ -0.9 ], [ -1.6 ] ]
    ]
    [
    	[ [ -0.936 ], [ -1.604 ] ],
    	[ [ -1.436 ], [ 1.86 ] ]
    ]
```



Next, we run the same optimization using L-BFGS, which is nearly ideal for purely second-order or quadratic functions.

Code from [LearningTester.java:249](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L249) executed in 3.44 seconds: 
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
    th(0)=1.293740053284649;dx=-7.912917341375033
    Armijo: th(2.154434690031884)=2257.6837373841554; dx=6567.271226031184 delta=-2256.3899973308708
    Armijo: th(1.077217345015942)=50.509620834450565; dx=206.7630809084248 delta=-49.215880781165914
    New Minimum: 1.293740053284649 > 1.2104404885356408
    WOLF (strong): th(0.3590724483386473)=1.2104404885356408; dx=8.324882862478539 delta=0.08329956474900824
    New Minimum: 1.2104404885356408 > 0.7710279363028762
    END: th(0.08976811208466183)=0.7710279363028762; dx=-3.930783992618444 delta=0.5227121169817728
    Iteration 1 complete. Error: 0.7710279363028762 Total: 239587995251768.0300; Orientation: 0.0001; Line Search: 0.0112
    LBFGS Accumulation History: 1 points
    th(0)=0.7710279363028762;dx=-2.003971649542225
    New Minimum: 0.7710279363028762 > 0.5695959113554253
    END: th(0.1933995347338658)=0.5695959113554253; dx=-0.027284304261027372 delta=0.20143202494745094
    Iteration 2 complete. Error: 0.56
```
...[skipping 140538 bytes](etc/139.txt)...
```
    E (weak): th(0.2163939858224583)=0.484190067204063; dx=-2.96700671569805E-9 delta=6.704443666905036E-10
    New Minimum: 0.484190067204063 > 0.48419006659042224
    END: th(0.4327879716449166)=0.48419006659042224; dx=-2.704507531636972E-9 delta=1.284085116015632E-9
    Iteration 249 complete. Error: 0.48419006659042224 Total: 239591394439804.6200; Orientation: 0.0000; Line Search: 0.0113
    LBFGS Accumulation History: 1 points
    th(0)=0.48419006659042224;dx=-4.705439011715035E-9
    Armijo: th(0.9324134195403436)=0.48419007002110126; dx=1.206463755529148E-8 delta=-3.430679018645577E-9
    New Minimum: 0.48419006659042224 > 0.4841900663511812
    WOLF (strong): th(0.4662067097701718)=0.4841900663511812; dx=3.6792313280483627E-9 delta=2.3924101588690405E-10
    New Minimum: 0.4841900663511812 > 0.4841900660763451
    END: th(0.15540223659005728)=0.4841900660763451; dx=-1.910630661510271E-9 delta=5.140771142109202E-10
    Iteration 250 complete. Error: 0.4841900660763451 Total: 239591413777298.6000; Orientation: 0.0000; Line Search: 0.0156
    
```

Returns: 

```
    0.4841900660763451
```



This training run resulted in the following regressed input:

Code from [LearningTester.java:154](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L154) executed in 0.00 seconds: 
```java
    return Arrays.stream(input_lbgfs).map(x -> x.prettyPrint()).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [
    	[ [ -0.0018786846496966093 ], [ 1.2577417957611565 ] ],
    	[ [ 1.1148007312123673 ], [ 0.022498701285598336 ] ]
    ]
    [
    	[ [ -0.752 ], [ 1.416 ] ],
    	[ [ -0.9 ], [ -1.6 ] ]
    ]
    [
    	[ [ -0.936 ], [ -1.604 ] ],
    	[ [ -1.436 ], [ 1.86 ] ]
    ]
```



Code from [LearningTester.java:95](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L95) executed in 0.00 seconds: 
```java
    return TestUtil.compare(runs);
```

Returns: 

![Result](etc/test.82.png)



Code from [LearningTester.java:98](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L98) executed in 0.00 seconds: 
```java
    return TestUtil.compareTime(runs);
```

Returns: 

![Result](etc/test.83.png)



### Performance
Now we execute larger-scale runs to benchmark performance:

Code from [PerformanceTester.java:66](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/PerformanceTester.java#L66) executed in 0.02 seconds: 
```java
    test(component, inputPrototype);
```
Logging: 
```
    100 batches
    Input Dimensions:
    	[2, 2, 1]
    	[2, 2, 1]
    	[2, 2, 1]
    Performance:
    	Evaluation performance: 0.000986s +- 0.000421s [0.000560s - 0.001686s]
    	Learning performance: 0.000250s +- 0.000048s [0.000186s - 0.000309s]
    
```

