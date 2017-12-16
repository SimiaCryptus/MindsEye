# PoolingLayer
## Double
### Batch Execution
Code from [BatchingTester.java:66](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/BatchingTester.java#L66) executed in 0.01 seconds: 
```java
    return test(reference, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (400#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (160#)}
```



Code from [SingleDerivativeTester.java:77](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/SingleDerivativeTester.java#L77) executed in 0.02 seconds: 
```java
    return test(component, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ 1.732, 1.632 ], [ 0.752, 1.104 ], [ 0.632, -1.1 ], [ -1.28, -0.952 ] ],
    	[ [ -1.572, -0.984 ], [ -0.58, -1.956 ], [ -1.412, -1.636 ], [ -1.072, 1.876 ] ],
    	[ [ 1.98, -1.924 ], [ 0.12, 1.456 ], [ -0.932, 0.624 ], [ -1.496, 1.78 ] ],
    	[ [ 0.42, 1.4 ], [ -1.508, -1.328 ], [ 1.528, 0.384 ], [ 1.04, 0.028 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.014797206065347297, negative=15, min=0.028, max=0.028, mean=-0.03887500000000003, count=32.0, positive=17, stdDev=1.306874988043998, zeros=0}
    Output: [
    	[ [ 1.732, 1.632 ], [ 0.632, 1.876 ] ],
    	[ [ 1.98, 1.456 ], [ 1.528, 1.78 ] ]
    ]
    Outputs Statistics: {meanExponent=0.17744848452646445, negative=0, min=1.78, max=1.78, mean=1.5769999999999997, count=8.0, positive=8, stdDev=0.39213135554301215, zeros=0}
    Feedback for input 0
    Inputs Values: [
    	[ [ 1.732, 1.632 ], [ 0.752, 1.104 ], [ 0.632, -1.1 ], [ -1.28, -0.952 ] ],
    	[ [ -1.572, -0.984 ], [ -0.58, -1.956 ], [ -1.412, -1.636 ], [ -1.072, 1.876 ] ],
    	[ [ 1.98, -1.924 ], [ 0.12, 1.456 ], [ -0.932, 0.624 ], 
```
...[skipping 1157 bytes](etc/129.txt)...
```
    0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], ... ]
    Measured Statistics: {meanExponent=-4.7830642341045674E-14, negative=0, min=0.0, max=0.0, mean=0.03124999999999656, count=256.0, positive=8, stdDev=0.17399263633841902, zeros=248}
    Feedback Error: [ [ -1.1013412404281553E-13, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, -1.1013412404281553E-13, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], ... ]
    Error Statistics: {meanExponent=-12.958078098036824, negative=8, min=0.0, max=0.0, mean=-3.4416913763379853E-15, count=256.0, positive=0, stdDev=1.9162526593034043E-14, zeros=248}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 3.4417e-15 +- 1.9163e-14 [0.0000e+00 - 1.1013e-13] (256#)
    relativeTol: 5.5067e-14 +- 0.0000e+00 [5.5067e-14 - 5.5067e-14] (8#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=3.4417e-15 +- 1.9163e-14 [0.0000e+00 - 1.1013e-13] (256#), relativeTol=5.5067e-14 +- 0.0000e+00 [5.5067e-14 - 5.5067e-14] (8#)}
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
      "class": "com.simiacryptus.mindseye.layers.cudnn.PoolingLayer",
      "id": "3380c05b-fcf5-4271-9daa-a8b97df9286f",
      "isFrozen": false,
      "name": "PoolingLayer/3380c05b-fcf5-4271-9daa-a8b97df9286f",
      "mode": 0,
      "windowX": 2,
      "windowY": 2,
      "paddingX": 0,
      "paddingY": 0,
      "strideX": 2,
      "strideY": 2
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
    	[ [ 1.188, -0.848 ], [ -0.864, 0.976 ], [ -1.764, -1.668 ], [ 1.824, -1.004 ] ],
    	[ [ -0.968, 1.356 ], [ 0.6, -0.072 ], [ 0.448, -0.636 ], [ -0.368, -1.94 ] ],
    	[ [ -1.68, -1.82 ], [ 1.028, 0.12 ], [ -1.172, -0.552 ], [ -0.844, -1.872 ] ],
    	[ [ 0.032, 0.504 ], [ 0.572, 1.476 ], [ -0.948, 1.92 ], [ 0.992, 0.644 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 1.188, 1.356 ], [ 1.824, -0.636 ] ],
    	[ [ 1.028, 1.476 ], [ 0.992, 1.92 ] ]
    ]
    --------------------
    Derivative: 
    [
    	[ [ 1.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 1.0, 0.0 ] ],
    	[ [ 0.0, 1.0 ], [ 0.0, 0.0 ], [ 0.0, 1.0 ], [ 0.0, 0.0 ] ],
    	[ [ 0.0, 0.0 ], [ 1.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ] ],
    	[ [ 0.0, 0.0 ], [ 0.0, 1.0 ], [ 0.0, 1.0 ], [ 1.0, 0.0 ] ]
    ]
```



[GPU Log](etc/cuda.log)

### Input Learning
In this test, we use a network to learn this target input, given it's pre-evaluated output:

Code from [LearningTester.java:127](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L127) executed in 0.01 seconds: 
```java
    return Arrays.stream(input_target).map(x -> x.prettyPrint()).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [
    	[ [ -0.888, 1.708 ], [ -1.484, -1.144 ], [ 1.672, 1.676 ], [ 1.78, 0.064 ], [ 0.512, 0.72 ], [ 1.296, -1.676 ], [ -1.248, 1.896 ], [ 1.308, -0.9 ], ... ],
    	[ [ 0.88, 1.332 ], [ -0.872, -0.86 ], [ 1.14, -1.092 ], [ 0.976, -0.828 ], [ -0.696, -1.82 ], [ 1.644, -1.276 ], [ -0.216, -1.588 ], [ 0.3, 0.868 ], ... ],
    	[ [ 0.796, -1.324 ], [ 1.788, -0.404 ], [ 0.688, 1.388 ], [ 1.552, 1.8 ], [ -1.048, 0.628 ], [ -1.28, 1.776 ], [ -1.44, -1.88 ], [ -1.52, -0.696 ], ... ],
    	[ [ -0.26, -0.248 ], [ 1.156, 0.632 ], [ -1.04, -1.772 ], [ 0.476, 0.9 ], [ 1.372, 0.168 ], [ -0.576, 1.728 ], [ 0.56, -0.216 ], [ -1.04, -1.572 ], ... ],
    	[ [ -1.788, -1.364 ], [ -0.308, -0.804 ], [ 1.276, -1.564 ], [ 1.32, 0.536 ], [ 0.256, 0.932 ], [ 1.672, -0.916 ], [ -0.444, -0.512 ], [ -0.436, 1.896 ], ... ],
    	[ [ -1.512, 1.508 ], [ 1.356, 1.52 ], [ -0.248, -0.056 ], [ 1.908, 1.496 ], [ 0.368, 0.952 ], [ 0.628, 1.808 ], [ 0.192, 0.056 ], [ 0.068, 1.416 ], ... ],
    	[ [ -0.096, -1.048 ], [ 0.612, 1.412 ], [ 0.648, -0.656 ], [ -1.784, 1.524 ], [ 0.324, -0.948 ], [ 1.74, 1.584 ], [ -1.808, -0.052 ], [ 0.14, 0.604 ], ... ],
    	[ [ 0.164, -1.98 ], [ 1.26, -0.076 ], [ 1.66, 0.148 ], [ -0.892, 0.54 ], [ -0.968, -0.996 ], [ 0.7, -0.34 ], [ -1.136, -0.24 ], [ -0.896, -1.668 ], ... ],
    	...
    ]
```



First, we use a conjugate gradient descent method, which converges the fastest for purely linear functions.

Code from [LearningTester.java:225](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L225) executed in 0.31 seconds: 
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
    F(0.0) = LineSearchPoint{point=PointSample{avg=0.8967213823999997}, derivative=-7.1737710592E-4}
    New Minimum: 0.8967213823999997 > 0.8967213823999269
    F(1.0E-10) = LineSearchPoint{point=PointSample{avg=0.8967213823999269}, derivative=-7.172716108799714E-4}, delta = -7.271960811294775E-14
    New Minimum: 0.8967213823999269 > 0.8967213823994985
    F(7.000000000000001E-10) = LineSearchPoint{point=PointSample{avg=0.8967213823994985}, derivative=-7.172716108797992E-4}, delta = -5.011546733157957E-13
    New Minimum: 0.8967213823994985 > 0.8967213823964879
    F(4.900000000000001E-9) = LineSearchPoint{point=PointSample{avg=0.8967213823964879}, derivative=-7.172716108785942E-4}, delta = -3.5117464491918327E-12
    New Minimum: 0.8967213823964879 > 0.8967213823753999
    F(3.430000000000001E-8) = LineSearchPoint{point=PointSample{avg=0.8967213823753999}, derivative=-7.17271610870159E-4}, delta = -2.4599766668131906E-11
    New Minimum: 0.8967213823753999 > 0.896721382227782
    F(2.401000000000000
```
...[skipping 13909 bytes](etc/130.txt)...
```
    nt=PointSample{avg=1.026134200324594E-292}, derivative=3.697038081771171E-280}, delta = -2.0812474159298974E-261
    1.026134200324594E-292 <= 2.0812474159298974E-261
    Converged to right
    Iteration 13 complete. Error: 1.026134200324594E-292 Total: 239578327097999.7000; Orientation: 0.0007; Line Search: 0.0111
    Zero gradient: 2.8651480943568612E-148
    F(0.0) = LineSearchPoint{point=PointSample{avg=1.026134200324594E-292}, derivative=-8.209073602596752E-296}
    New Minimum: 1.026134200324594E-292 > 4.9E-324
    F(2500.0000000000005) = LineSearchPoint{point=PointSample{avg=4.9E-324}, derivative=1.822780504889E-311}, delta = -1.026134200324594E-292
    4.9E-324 <= 1.026134200324594E-292
    Converged to right
    Iteration 14 complete. Error: 4.9E-324 Total: 239578350111998.7000; Orientation: 0.0013; Line Search: 0.0171
    Zero gradient: 0.0
    F(0.0) = LineSearchPoint{point=PointSample{avg=4.9E-324}, derivative=0.0}
    Iteration 15 failed, aborting. Error: 4.9E-324 Total: 239578355693585.6600; Orientation: 0.0007; Line Search: 0.0020
    
```

Returns: 

```
    4.9E-324
```



Training Converged

Next, we run the same optimization using L-BFGS, which is nearly ideal for purely second-order or quadratic functions.

Code from [LearningTester.java:249](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L249) executed in 0.41 seconds: 
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
    th(0)=0.8967213823999997;dx=-7.1737710592E-4
    New Minimum: 0.8967213823999997 > 0.8951767334153207
    WOLFE (weak): th(2.154434690031884)=0.8951767334153207; dx=-7.166534849437381E-4 delta=0.0015446489846789868
    New Minimum: 0.8951767334153207 > 0.8936334161426105
    WOLFE (weak): th(4.308869380063768)=0.8936334161426105; dx=-7.160353590074762E-4 delta=0.003087966257389163
    New Minimum: 0.8936334161426105 > 0.8874811586664295
    WOLFE (weak): th(12.926608140191302)=0.8874811586664295; dx=-7.117234937764303E-4 delta=0.009240223733570141
    New Minimum: 0.8874811586664295 > 0.8601620703226505
    WOLFE (weak): th(51.70643256076521)=0.8601620703226505; dx=-6.969930325921383E-4 delta=0.03655931207734919
    New Minimum: 0.8601620703226505 > 0.7254888632746489
    END: th(258.53216280382605)=0.7254888632746489; dx=-6.050020908823822E-4 delta=0.17123251912535076
    Iteration 1 complete. Error: 0.7254888632746489 Total: 239578385081127.6200; Orientation: 0.
```
...[skipping 13732 bytes](etc/131.txt)...
```
    11 > 5.2216783472341894E-11
    WOLF (strong): th(4918.404508816173)=5.2216783472341894E-11; dx=1.7333369499485122E-41 delta=4.665514956279824E-11
    END: th(2459.2022544080864)=5.2216783472341894E-11; dx=0.0 delta=4.665514956279824E-11
    Iteration 29 complete. Error: 5.2216783472341894E-11 Total: 239578756905541.2500; Orientation: 0.0010; Line Search: 0.0084
    LBFGS Accumulation History: 1 points
    th(0)=5.2216783472341894E-11;dx=-4.177342677787348E-14
    New Minimum: 5.2216783472341894E-11 > 1.5832314514979556E-11
    END: th(5298.190646701396)=1.5832314514979556E-11; dx=0.0 delta=3.638446895736234E-11
    Iteration 30 complete. Error: 1.5832314514979556E-11 Total: 239578765617616.2500; Orientation: 0.0011; Line Search: 0.0060
    LBFGS Accumulation History: 1 points
    th(0)=1.5832314514979556E-11;dx=-1.2665851611983682E-14
    MAX ALPHA: th(0)=1.5832314514979556E-11;th'(0)=-1.2665851611983682E-14;
    Iteration 31 failed, aborting. Error: 1.5832314514979556E-11 Total: 239578772136211.2200; Orientation: 0.0010; Line Search: 0.0043
    
```

Returns: 

```
    1.5832314514979556E-11
```



Training Converged

Code from [LearningTester.java:95](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L95) executed in 0.00 seconds: 
```java
    return TestUtil.compare(runs);
```

Returns: 

![Result](etc/test.78.png)



Code from [LearningTester.java:98](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L98) executed in 0.00 seconds: 
```java
    return TestUtil.compareTime(runs);
```

Returns: 

![Result](etc/test.79.png)



### Performance
Now we execute larger-scale runs to benchmark performance:

Code from [PerformanceTester.java:66](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/PerformanceTester.java#L66) executed in 0.21 seconds: 
```java
    test(component, inputPrototype);
```
Logging: 
```
    100 batches
    Input Dimensions:
    	[100, 100, 2]
    Performance:
    	Evaluation performance: 0.012397s +- 0.001104s [0.010651s - 0.013754s]
    	Learning performance: 0.014480s +- 0.000921s [0.012732s - 0.015386s]
    
```

