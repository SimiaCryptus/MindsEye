# BinarySumLayer
## Float_Add
### Batch Execution
Code from [BatchingTester.java:66](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/BatchingTester.java#L66) executed in 0.01 seconds: 
```java
    return test(reference, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (120#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (120#)}
```



Code from [SingleDerivativeTester.java:77](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/SingleDerivativeTester.java#L77) executed in 0.02 seconds: 
```java
    return test(component, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ 0.448 ], [ 0.488 ] ],
    	[ [ 1.956 ], [ 1.196 ] ]
    ],
    [
    	[ [ 0.832 ], [ -0.832 ] ],
    	[ [ -1.048 ], [ -1.036 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.07280053347379269, negative=0, min=1.196, max=1.196, mean=1.022, count=4.0, positive=4, stdDev=0.6158863531529175, zeros=0},
    {meanExponent=-0.031008077340407513, negative=3, min=-1.036, max=-1.036, mean=-0.521, count=4.0, positive=1, stdDev=0.7858568571947439, zeros=0}
    Output: [
    	[ [ 1.28 ], [ -0.344 ] ],
    	[ [ 0.9079999999999999 ], [ 0.15999999999999992 ] ]
    ]
    Outputs Statistics: {meanExponent=-0.298506439150898, negative=1, min=0.15999999999999992, max=0.15999999999999992, mean=0.5009999999999999, count=4.0, positive=3, stdDev=0.6330078988448722, zeros=0}
    Feedback for input 0
    Inputs Values: [
    	[ [ 0.448 ], [ 0.488 ] ],
    	[ [ 1.956 ], [ 1.196 ] ]
    ]
    Value Statistics: {meanExponent=-0.07280053347379269, negative=0, min=1.196, max=1.196, mean=1.022, count=4.0, positive=4, stdDev=0.6158863531529175, zeros=0}
    Implemented Feedback: [ [ 1.0, 0.0, 0.0, 
```
...[skipping 1508 bytes](etc/35.txt)...
```
    tive=4, stdDev=0.4330127018922193, zeros=12}
    Measured Feedback: [ [ 0.9999999999998899, 0.0, 0.0, 0.0 ], [ 0.0, 0.9999999999998899, 0.0, 0.0 ], [ 0.0, 0.0, 1.0000000000000009, 0.0 ], [ 0.0, 0.0, 0.0, 0.9999999999998899 ] ]
    Measured Statistics: {meanExponent=-3.577654900912893E-14, negative=0, min=0.9999999999998899, max=0.9999999999998899, mean=0.2499999999999794, count=16.0, positive=4, stdDev=0.43301270189218366, zeros=12}
    Feedback Error: [ [ -1.1013412404281553E-13, 0.0, 0.0, 0.0 ], [ 0.0, -1.1013412404281553E-13, 0.0, 0.0 ], [ 0.0, 0.0, 8.881784197001252E-16, 0.0 ], [ 0.0, 0.0, 0.0, -1.1013412404281553E-13 ] ]
    Error Statistics: {meanExponent=-13.481433519327382, negative=3, min=-1.1013412404281553E-13, max=-1.1013412404281553E-13, mean=-2.0594637106796654E-14, count=16.0, positive=1, stdDev=4.30139072393814E-14, zeros=12}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 2.0706e-14 +- 4.2961e-14 [0.0000e+00 - 1.1013e-13] (32#)
    relativeTol: 4.1411e-14 +- 2.3652e-14 [4.4409e-16 - 5.5067e-14] (8#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=2.0706e-14 +- 4.2961e-14 [0.0000e+00 - 1.1013e-13] (32#), relativeTol=4.1411e-14 +- 2.3652e-14 [4.4409e-16 - 5.5067e-14] (8#)}
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
      "class": "com.simiacryptus.mindseye.layers.cudnn.BinarySumLayer",
      "id": "6649973c-039d-4dcc-91ae-2bd7ed656500",
      "isFrozen": false,
      "name": "BinarySumLayer/6649973c-039d-4dcc-91ae-2bd7ed656500",
      "rightFactor": 1.0,
      "leftFactor": 1.0
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
    	[ [ -1.596 ], [ -0.964 ] ],
    	[ [ -0.588 ], [ -1.952 ] ]
    ],
    [
    	[ [ 1.784 ], [ 0.072 ] ],
    	[ [ 0.004 ], [ -0.396 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 0.18799999999999994 ], [ -0.892 ] ],
    	[ [ -0.584 ], [ -2.348 ] ]
    ]
    --------------------
    Derivative: 
    [
    	[ [ 1.0 ], [ 1.0 ] ],
    	[ [ 1.0 ], [ 1.0 ] ]
    ],
    [
    	[ [ 1.0 ], [ 1.0 ] ],
    	[ [ 1.0 ], [ 1.0 ] ]
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
    	[ [ 1.084 ], [ 0.436 ], [ -0.368 ], [ 1.58 ], [ 0.192 ], [ -1.124 ], [ 1.572 ], [ -0.932 ], ... ],
    	[ [ -1.92 ], [ -1.248 ], [ -0.52 ], [ 1.168 ], [ -1.056 ], [ 0.488 ], [ -0.88 ], [ 1.96 ], ... ],
    	[ [ 1.8 ], [ 0.928 ], [ -0.948 ], [ 1.412 ], [ 0.028 ], [ -1.2 ], [ 0.788 ], [ -0.44 ], ... ],
    	[ [ 1.516 ], [ 1.124 ], [ 1.284 ], [ -0.628 ], [ -0.54 ], [ -1.98 ], [ 0.38 ], [ 1.628 ], ... ],
    	[ [ -1.996 ], [ 0.58 ], [ 0.736 ], [ -1.328 ], [ -0.396 ], [ 0.764 ], [ 0.912 ], [ 1.716 ], ... ],
    	[ [ 1.628 ], [ 1.784 ], [ -0.428 ], [ 1.18 ], [ 1.408 ], [ -1.288 ], [ 0.332 ], [ -1.404 ], ... ],
    	[ [ -0.512 ], [ -1.04 ], [ 1.564 ], [ -0.712 ], [ 0.448 ], [ -0.712 ], [ 0.408 ], [ -0.028 ], ... ],
    	[ [ 1.688 ], [ 1.22 ], [ -1.604 ], [ -1.924 ], [ 0.456 ], [ -1.652 ], [ -0.028 ], [ -0.504 ], ... ],
    	...
    ]
    [
    	[ [ 0.14 ], [ -0.536 ], [ 0.932 ], [ 1.38 ], [ 1.484 ], [ 1.788 ], [ -1.296 ], [ -0.88 ], ... ],
    	[ [ -1.208 ], [ 1.688 ], [ -1.164 ], [ 1.268 ], [ -0.788 ], [ -0.572 ], [ 1.724 ], [ -0.112 ], ... ],
    	[ [ 1.644 ], [ -0.412 ], [ -0.02 ], [ 0.284 ], [ 0.18 ], [ -1.448 ], [ -1.068 ], [ 1.348 ], ... ],
    	[ [ -0.336 ], [ 1.048 ], [ -0.368 ], [ 0.616 ], [ 1.424 ], [ 1.42 ], [ 1.888 ], [ -1.18 ], ... ],
    	[ [ -1.036 ], [ -0.508 ], [ 0.972 ], [ -1.568 ], [ -0.576 ], [ -0.764 ], [ 1.652 ], [ 0.696 ], ... ],
    	[ [ 1.52 ], [ 0.628 ], [ 0.928 ], [ 1.532 ], [ 0.872 ], [ 1.644 ], [ -1.86 ], [ -0.568 ], ... ],
    	[ [ 1.988 ], [ -0.136 ], [ -1.708 ], [ -0.844 ], [ 1.464 ], [ -1.504 ], [ -0.352 ], [ 0.768 ], ... ],
    	[ [ -1.808 ], [ -1.396 ], [ -0.892 ], [ -0.104 ], [ -0.956 ], [ 1.04 ], [ -1.128 ], [ -0.972 ], ... ],
    	...
    ]
```



First, we use a conjugate gradient descent method, which converges the fastest for purely linear functions.

Code from [LearningTester.java:225](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L225) executed in 0.14 seconds: 
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
    F(0.0) = LineSearchPoint{point=PointSample{avg=8.148089718400009}, derivative=-0.013036943549440001}
    New Minimum: 8.148089718400009 > 8.148089718398671
    F(1.0E-10) = LineSearchPoint{point=PointSample{avg=8.148089718398671}, derivative=-0.013036943549438958}, delta = -1.3375967000683886E-12
    New Minimum: 8.148089718398671 > 8.14808971839089
    F(7.000000000000001E-10) = LineSearchPoint{point=PointSample{avg=8.14808971839089}, derivative=-0.013036943549432701}, delta = -9.118039656641486E-12
    New Minimum: 8.14808971839089 > 8.148089718336117
    F(4.900000000000001E-9) = LineSearchPoint{point=PointSample{avg=8.148089718336117}, derivative=-0.013036943549388896}, delta = -6.389200279954821E-11
    New Minimum: 8.148089718336117 > 8.148089717952857
    F(3.430000000000001E-8) = LineSearchPoint{point=PointSample{avg=8.148089717952857}, derivative=-0.013036943549082268}, delta = -4.4715164904118865E-10
    New Minimum: 8.148089717952857 > 8.148089715269796
    F(2.4010000000000004E-7) = Lin
```
...[skipping 6621 bytes](etc/36.txt)...
```
    ient: 8.383481632102247E-138
    F(0.0) = LineSearchPoint{point=PointSample{avg=4.392672767237235E-272}, derivative=-7.028276427579575E-275}
    New Minimum: 4.392672767237235E-272 > 5.597948499686933E-302
    F(1250.0000000000014) = LineSearchPoint{point=PointSample{avg=5.597948499686933E-302}, derivative=7.926944161022709E-290}, delta = -4.392672767237235E-272
    5.597948499686933E-302 <= 4.392672767237235E-272
    Converged to right
    Iteration 10 complete. Error: 5.597948499686933E-302 Total: 239461002885327.0000; Orientation: 0.0003; Line Search: 0.0047
    Zero gradient: 9.463993659919206E-153
    F(0.0) = LineSearchPoint{point=PointSample{avg=5.597948499686933E-302}, derivative=-8.956717599499092E-305}
    New Minimum: 5.597948499686933E-302 > 0.0
    F(1250.0000000000014) = LineSearchPoint{point=PointSample{avg=0.0}, derivative=9.9515E-320}, delta = -5.597948499686933E-302
    0.0 <= 5.597948499686933E-302
    Converged to right
    Iteration 11 complete. Error: 0.0 Total: 239461009802037.0000; Orientation: 0.0003; Line Search: 0.0043
    
```

Returns: 

```
    0.0
```



Training Converged

Next, we run the same optimization using L-BFGS, which is nearly ideal for purely second-order or quadratic functions.

Code from [LearningTester.java:249](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L249) executed in 2.36 seconds: 
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
    th(0)=8.148089718400009;dx=-0.013036943549440001
    New Minimum: 8.148089718400009 > 8.120026679817745
    WOLFE (weak): th(2.154434690031884)=8.120026679817745; dx=-0.01301447375469208 delta=0.028063038582264
    New Minimum: 8.120026679817745 > 8.09201205094079
    WOLFE (weak): th(4.308869380063768)=8.09201205094079; dx=-0.012992003959944159 delta=0.056077667459218716
    New Minimum: 8.09201205094079 > 7.980437632485693
    WOLFE (weak): th(12.926608140191302)=7.980437632485693; dx=-0.012902124780952477 delta=0.16765208591431513
    New Minimum: 7.980437632485693 > 7.4879378710838065
    WOLFE (weak): th(51.70643256076521)=7.4879378710838065; dx=-0.012497668475489903 delta=0.660151847316202
    New Minimum: 7.4879378710838065 > 5.126170384248183
    END: th(258.53216280382605)=5.126170384248183; dx=-0.010340568179689511 delta=3.021919334151826
    Iteration 1 complete. Error: 5.126170384248183 Total: 239461033330993.0000; Orientation: 0.0006; Line Search: 0.0
```
...[skipping 107106 bytes](etc/37.txt)...
```
    79
    LBFGS Accumulation History: 1 points
    th(0)=7.472009E-317;dx=-1.19554E-319
    New Minimum: 7.472009E-317 > 5.8613814E-317
    WOLF (strong): th(2357.111348962417)=5.8613814E-317; dx=1.05883E-319 delta=1.610628E-317
    New Minimum: 5.8613814E-317 > 2.44093E-319
    END: th(1178.5556744812086)=2.44093E-319; dx=-6.84E-321 delta=7.4476E-317
    Iteration 199 complete. Error: 2.44093E-319 Total: 239463357147375.6600; Orientation: 0.0005; Line Search: 0.0080
    LBFGS Accumulation History: 1 points
    th(0)=2.44093E-319;dx=-3.9E-322
    Armijo: th(2539.12122923624)=2.5961E-319; dx=4.05E-322 delta=-1.552E-320
    New Minimum: 2.44093E-319 > 5.9E-323
    END: th(1269.56061461812)=5.9E-323; dx=0.0 delta=2.44034E-319
    Iteration 200 complete. Error: 5.9E-323 Total: 239463366482982.6600; Orientation: 0.0005; Line Search: 0.0072
    LBFGS Accumulation History: 1 points
    th(0)=5.9E-323;dx=0.0 (ERROR: Starting derivative negative)
    Iteration 201 failed, aborting. Error: 5.9E-323 Total: 239463374370899.6200; Orientation: 0.0006; Line Search: 0.0056
    
```

Returns: 

```
    5.9E-323
```



Training Converged

Code from [LearningTester.java:95](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L95) executed in 0.00 seconds: 
```java
    return TestUtil.compare(runs);
```

Returns: 

![Result](etc/test.31.png)



Code from [LearningTester.java:98](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L98) executed in 0.00 seconds: 
```java
    return TestUtil.compareTime(runs);
```

Returns: 

![Result](etc/test.32.png)



### Performance
Now we execute larger-scale runs to benchmark performance:

Code from [PerformanceTester.java:66](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/PerformanceTester.java#L66) executed in 0.29 seconds: 
```java
    test(component, inputPrototype);
```
Logging: 
```
    100 batches
    Input Dimensions:
    	[100, 100, 1]
    	[100, 100, 1]
    Performance:
    	Evaluation performance: 0.011076s +- 0.000830s [0.010272s - 0.012380s]
    	Learning performance: 0.033145s +- 0.011886s [0.025448s - 0.056748s]
    
```

