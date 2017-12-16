# BinarySumLayer
## Double_Add
### Batch Execution
Code from [BatchingTester.java:66](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/BatchingTester.java#L66) executed in 0.01 seconds: 
```java
    return test(reference, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (120#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (120#)}
```



Code from [SingleDerivativeTester.java:77](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/SingleDerivativeTester.java#L77) executed in 0.01 seconds: 
```java
    return test(component, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ -1.812 ], [ 0.508 ] ],
    	[ [ 0.56 ], [ -1.808 ] ]
    ],
    [
    	[ [ -0.004 ], [ 1.128 ] ],
    	[ [ -1.308 ], [ 0.64 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.007647910307435377, negative=2, min=-1.808, max=-1.808, mean=-0.638, count=4.0, positive=2, stdDev=1.1721450422196051, zeros=0},
    {meanExponent=-0.6057107977631446, negative=2, min=0.64, max=0.64, mean=0.11399999999999996, count=4.0, positive=2, stdDev=0.9139037148409016, zeros=0}
    Output: [
    	[ [ -1.816 ], [ 1.636 ] ],
    	[ [ -0.748 ], [ -1.1680000000000001 ] ]
    ]
    Outputs Statistics: {meanExponent=0.10356089604030315, negative=3, min=-1.1680000000000001, max=-1.1680000000000001, mean=-0.524, count=4.0, positive=1, stdDev=1.303819005843986, zeros=0}
    Feedback for input 0
    Inputs Values: [
    	[ [ -1.812 ], [ 0.508 ] ],
    	[ [ 0.56 ], [ -1.808 ] ]
    ]
    Value Statistics: {meanExponent=-0.007647910307435377, negative=2, min=-1.808, max=-1.808, mean=-0.638, count=4.0, positive=2, stdDev=1.1721450422196051, zeros=0}
    Implemented Feedback: [ [ 1.0, 0.0, 0.0, 0.0 ], [
```
...[skipping 1497 bytes](etc/31.txt)...
```
    positive=4, stdDev=0.4330127018922193, zeros=12}
    Measured Feedback: [ [ 0.9999999999998899, 0.0, 0.0, 0.0 ], [ 0.0, 0.9999999999998899, 0.0, 0.0 ], [ 0.0, 0.0, 0.9999999999998899, 0.0 ], [ 0.0, 0.0, 0.0, 1.0000000000021103 ] ]
    Measured Statistics: {meanExponent=1.93251224297035E-13, negative=0, min=1.0000000000021103, max=1.0000000000021103, mean=0.25000000000011124, count=16.0, positive=4, stdDev=0.433012701892412, zeros=12}
    Feedback Error: [ [ -1.1013412404281553E-13, 0.0, 0.0, 0.0 ], [ 0.0, -1.1013412404281553E-13, 0.0, 0.0 ], [ 0.0, 0.0, -1.1013412404281553E-13, 0.0 ], [ 0.0, 0.0, 0.0, 2.1103119252074976E-12 ] ]
    Error Statistics: {meanExponent=-12.637471910250095, negative=3, min=2.1103119252074976E-12, max=2.1103119252074976E-12, mean=1.1124434706744069E-13, count=16.0, positive=1, stdDev=5.179165063886764E-13, zeros=12}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 1.5254e-13 +- 5.0729e-13 [0.0000e+00 - 2.1103e-12] (32#)
    relativeTol: 3.0509e-13 +- 4.3305e-13 [5.5067e-14 - 1.0552e-12] (8#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=1.5254e-13 +- 5.0729e-13 [0.0000e+00 - 2.1103e-12] (32#), relativeTol=3.0509e-13 +- 4.3305e-13 [5.5067e-14 - 1.0552e-12] (8#)}
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
      "id": "273834f2-5e43-4c3a-8c76-2a1860acabcc",
      "isFrozen": false,
      "name": "BinarySumLayer/273834f2-5e43-4c3a-8c76-2a1860acabcc",
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
    	[ [ -0.432 ], [ -0.204 ] ],
    	[ [ 1.708 ], [ 0.78 ] ]
    ],
    [
    	[ [ -0.692 ], [ -1.584 ] ],
    	[ [ -1.556 ], [ 0.76 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ -1.1239999999999999 ], [ -1.788 ] ],
    	[ [ 0.1519999999999999 ], [ 1.54 ] ]
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
    	[ [ -1.088 ], [ 1.688 ], [ -1.696 ], [ 0.672 ], [ -0.292 ], [ -0.668 ], [ 1.548 ], [ -0.304 ], ... ],
    	[ [ -0.548 ], [ -0.16 ], [ -1.808 ], [ -0.348 ], [ 0.868 ], [ -0.304 ], [ -0.86 ], [ -1.164 ], ... ],
    	[ [ -0.164 ], [ -1.684 ], [ 0.14 ], [ 0.28 ], [ 0.888 ], [ 1.976 ], [ -0.984 ], [ -1.496 ], ... ],
    	[ [ -1.296 ], [ -1.832 ], [ -0.816 ], [ -1.212 ], [ 0.576 ], [ -0.624 ], [ 1.044 ], [ 1.968 ], ... ],
    	[ [ 0.876 ], [ 1.476 ], [ -0.744 ], [ -0.24 ], [ -1.54 ], [ -0.372 ], [ -0.192 ], [ -0.668 ], ... ],
    	[ [ -1.188 ], [ -1.24 ], [ 0.328 ], [ -0.504 ], [ 0.536 ], [ 1.592 ], [ 1.372 ], [ 0.288 ], ... ],
    	[ [ 1.708 ], [ -0.484 ], [ -1.82 ], [ -0.208 ], [ 1.78 ], [ 0.432 ], [ 1.532 ], [ 0.94 ], ... ],
    	[ [ -1.092 ], [ 0.084 ], [ -1.204 ], [ -0.932 ], [ -0.544 ], [ -0.308 ], [ -1.996 ], [ -1.4 ], ... ],
    	...
    ]
    [
    	[ [ 1.96 ], [ 1.828 ], [ -1.932 ], [ -1.456 ], [ -1.304 ], [ -0.256 ], [ 0.308 ], [ -0.564 ], ... ],
    	[ [ 0.788 ], [ 0.032 ], [ -0.736 ], [ -1.68 ], [ 0.336 ], [ 1.984 ], [ -0.524 ], [ -1.22 ], ... ],
    	[ [ 0.868 ], [ -0.904 ], [ 0.936 ], [ 1.556 ], [ -0.512 ], [ -0.472 ], [ 1.24 ], [ -0.648 ], ... ],
    	[ [ 1.696 ], [ -1.432 ], [ -1.704 ], [ -0.364 ], [ 1.72 ], [ 1.404 ], [ -0.44 ], [ -1.36 ], ... ],
    	[ [ -0.92 ], [ 1.38 ], [ -0.888 ], [ 1.36 ], [ 0.244 ], [ 1.216 ], [ 1.088 ], [ -0.54 ], ... ],
    	[ [ -1.696 ], [ 1.696 ], [ 0.736 ], [ 1.1 ], [ 1.42 ], [ -0.604 ], [ -1.984 ], [ -0.744 ], ... ],
    	[ [ 1.132 ], [ -0.208 ], [ 1.228 ], [ 1.384 ], [ -0.348 ], [ 0.908 ], [ -1.34 ], [ 1.04 ], ... ],
    	[ [ 0.34 ], [ 0.12 ], [ 0.732 ], [ -0.192 ], [ 1.168 ], [ 1.16 ], [ 1.456 ], [ -1.84 ], ... ],
    	...
    ]
```



First, we use a conjugate gradient descent method, which converges the fastest for purely linear functions.

Code from [LearningTester.java:225](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L225) executed in 0.16 seconds: 
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
    F(0.0) = LineSearchPoint{point=PointSample{avg=8.030231740800046}, derivative=-0.012848370785280002}
    New Minimum: 8.030231740800046 > 8.03023174079868
    F(1.0E-10) = LineSearchPoint{point=PointSample{avg=8.03023174079868}, derivative=-0.012848370785278973}, delta = -1.3660184094987926E-12
    New Minimum: 8.03023174079868 > 8.030231740791038
    F(7.000000000000001E-10) = LineSearchPoint{point=PointSample{avg=8.030231740791038}, derivative=-0.012848370785272806}, delta = -9.00790553259867E-12
    New Minimum: 8.030231740791038 > 8.030231740737074
    F(4.900000000000001E-9) = LineSearchPoint{point=PointSample{avg=8.030231740737074}, derivative=-0.012848370785229636}, delta = -6.297184995673888E-11
    New Minimum: 8.030231740737074 > 8.030231740359307
    F(3.430000000000001E-8) = LineSearchPoint{point=PointSample{avg=8.030231740359307}, derivative=-0.012848370784927442}, delta = -4.4073900085095374E-10
    New Minimum: 8.030231740359307 > 8.030231737715088
    F(2.4010000000000004E-7) = Line
```
...[skipping 6359 bytes](etc/32.txt)...
```
    5.6317278643367) = LineSearchPoint{point=PointSample{avg=1.1242631868536057E-46}, derivative=6.310461233505683E-48}, delta = -1.3824875160248767E-43
    1.1242631868536057E-46 <= 1.3836117792117303E-43
    New Minimum: 1.1242631868536057E-46 > 1.6554913315980084E-75
    F(1250.0) = LineSearchPoint{point=PointSample{avg=1.6554913315980084E-75}, derivative=1.695269155427844E-62}, delta = -1.3836117792117303E-43
    Right bracket at 1250.0
    Converged to right
    Iteration 4 complete. Error: 1.6554913315980084E-75 Total: 239457496905913.5300; Orientation: 0.0007; Line Search: 0.0101
    Zero gradient: 1.627509179868677E-39
    F(0.0) = LineSearchPoint{point=PointSample{avg=1.6554913315980084E-75}, derivative=-2.6487861305568134E-78}
    New Minimum: 1.6554913315980084E-75 > 0.0
    F(1250.0) = LineSearchPoint{point=PointSample{avg=0.0}, derivative=0.0}, delta = -1.6554913315980084E-75
    0.0 <= 1.6554913315980084E-75
    Converged to right
    Iteration 5 complete. Error: 0.0 Total: 239457503694098.5000; Orientation: 0.0003; Line Search: 0.0042
    
```

Returns: 

```
    0.0
```



Training Converged

Next, we run the same optimization using L-BFGS, which is nearly ideal for purely second-order or quadratic functions.

Code from [LearningTester.java:249](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L249) executed in 2.51 seconds: 
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
    th(0)=8.030231740800046;dx=-0.012848370785280002
    New Minimum: 8.030231740800046 > 8.002574619811561
    WOLFE (weak): th(2.154434690031884)=8.002574619811561; dx=-0.012826226004695842 delta=0.027657120988484607
    New Minimum: 8.002574619811561 > 7.974965208306572
    WOLFE (weak): th(4.308869380063768)=7.974965208306572; dx=-0.012804081224111682 delta=0.0552665324934738
    New Minimum: 7.974965208306572 > 7.865004657121685
    WOLFE (weak): th(12.926608140191302)=7.865004657121685; dx=-0.012715502101775043 delta=0.16522708367836092
    New Minimum: 7.865004657121685 > 7.379628654521371
    WOLFE (weak): th(51.70643256076521)=7.379628654521371; dx=-0.012316896051260172 delta=0.6506030862786751
    New Minimum: 7.379628654521371 > 5.052022934330405
    END: th(258.53216280382605)=5.052022934330405; dx=-0.010190997115180856 delta=2.978208806469641
    Iteration 1 complete. Error: 5.052022934330405 Total: 239457593668615.5000; Orientation: 0.0007; Line Search: 
```
...[skipping 107171 bytes](etc/33.txt)...
```
    
    LBFGS Accumulation History: 1 points
    th(0)=1.2353127E-316;dx=-1.97656E-319
    New Minimum: 1.2353127E-316 > 9.69035E-317
    WOLF (strong): th(2357.111348962417)=9.69035E-317; dx=1.75057E-319 delta=2.6627767E-317
    New Minimum: 9.69035E-317 > 4.03543E-319
    END: th(1178.5556744812086)=4.03543E-319; dx=-1.13E-320 delta=1.23127725E-316
    Iteration 199 complete. Error: 4.03543E-319 Total: 239459997719809.0000; Orientation: 0.0005; Line Search: 0.0076
    LBFGS Accumulation History: 1 points
    th(0)=4.03543E-319;dx=-6.47E-322
    Armijo: th(2539.12122923624)=4.292E-319; dx=6.57E-322 delta=-2.5657E-320
    New Minimum: 4.03543E-319 > 1.0E-322
    END: th(1269.56061461812)=1.0E-322; dx=0.0 delta=4.03444E-319
    Iteration 200 complete. Error: 1.0E-322 Total: 239460008093593.0000; Orientation: 0.0005; Line Search: 0.0080
    LBFGS Accumulation History: 1 points
    th(0)=1.0E-322;dx=0.0 (ERROR: Starting derivative negative)
    Iteration 201 failed, aborting. Error: 1.0E-322 Total: 239460015685702.0000; Orientation: 0.0005; Line Search: 0.0052
    
```

Returns: 

```
    1.0E-322
```



Training Converged

Code from [LearningTester.java:95](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L95) executed in 0.00 seconds: 
```java
    return TestUtil.compare(runs);
```

Returns: 

![Result](etc/test.29.png)



Code from [LearningTester.java:98](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L98) executed in 0.00 seconds: 
```java
    return TestUtil.compareTime(runs);
```

Returns: 

![Result](etc/test.30.png)



### Performance
Now we execute larger-scale runs to benchmark performance:

Code from [PerformanceTester.java:66](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/PerformanceTester.java#L66) executed in 0.27 seconds: 
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
    	Evaluation performance: 0.011572s +- 0.000649s [0.010528s - 0.012445s]
    	Learning performance: 0.027467s +- 0.001622s [0.025451s - 0.029687s]
    
```

