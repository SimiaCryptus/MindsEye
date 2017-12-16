# SimpleConvolutionLayer
## SimpleConvolutionLayerTest
### Batch Execution
Code from [BatchingTester.java:66](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/BatchingTester.java#L66) executed in 0.01 seconds: 
```java
    return test(reference, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (20#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (20#)}
```



Code from [SingleDerivativeTester.java:77](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/SingleDerivativeTester.java#L77) executed in 0.01 seconds: 
```java
    return test(component, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ -0.52 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.28399665636520083, negative=1, min=-0.52, max=-0.52, mean=-0.52, count=1.0, positive=0, stdDev=0.0, zeros=0}
    Output: [
    	[ [ 0.51168 ] ]
    ]
    Outputs Statistics: {meanExponent=-0.2910015579338593, negative=0, min=0.51168, max=0.51168, mean=0.51168, count=1.0, positive=1, stdDev=0.0, zeros=0}
    Feedback for input 0
    Inputs Values: [
    	[ [ -0.52 ] ]
    ]
    Value Statistics: {meanExponent=-0.28399665636520083, negative=1, min=-0.52, max=-0.52, mean=-0.52, count=1.0, positive=0, stdDev=0.0, zeros=0}
    Implemented Feedback: [ [ -0.984 ] ]
    Implemented Statistics: {meanExponent=-0.007004901568658489, negative=1, min=-0.984, max=-0.984, mean=-0.984, count=1.0, positive=0, stdDev=0.0, zeros=0}
    Measured Feedback: [ [ -0.98400000000054 ] ]
    Measured Statistics: {meanExponent=-0.007004901568420151, negative=1, min=-0.98400000000054, max=-0.98400000000054, mean=-0.98400000000054, count=1.0, positive=0, stdDev=0.0, zeros=0}
    Feedback Error: [ [ -5.400124791776761E-13
```
...[skipping 74 bytes](etc/143.txt)...
```
    =-5.400124791776761E-13, max=-5.400124791776761E-13, mean=-5.400124791776761E-13, count=1.0, positive=0, stdDev=0.0, zeros=0}
    Learning Gradient for weight set 0
    Weights: [ -0.984 ]
    Implemented Gradient: [ [ -0.52 ] ]
    Implemented Statistics: {meanExponent=-0.28399665636520083, negative=1, min=-0.52, max=-0.52, mean=-0.52, count=1.0, positive=0, stdDev=0.0, zeros=0}
    Measured Gradient: [ [ -0.52000000000052 ] ]
    Measured Statistics: {meanExponent=-0.2839966563647665, negative=1, min=-0.52000000000052, max=-0.52000000000052, mean=-0.52000000000052, count=1.0, positive=0, stdDev=0.0, zeros=0}
    Gradient Error: [ [ -5.200284647344233E-13 ] ]
    Error Statistics: {meanExponent=-12.283972883790678, negative=1, min=-5.200284647344233E-13, max=-5.200284647344233E-13, mean=-5.200284647344233E-13, count=1.0, positive=0, stdDev=0.0, zeros=0}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 5.3002e-13 +- 9.9920e-15 [5.2003e-13 - 5.4001e-13] (2#)
    relativeTol: 3.8721e-13 +- 1.1282e-13 [2.7440e-13 - 5.0003e-13] (2#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=5.3002e-13 +- 9.9920e-15 [5.2003e-13 - 5.4001e-13] (2#), relativeTol=3.8721e-13 +- 1.1282e-13 [2.7440e-13 - 5.0003e-13] (2#)}
```



### Reference Implementation
Code from [EquivalencyTester.java:61](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/EquivalencyTester.java#L61) executed in 0.00 seconds: 
```java
    System.out.println(new GsonBuilder().setPrettyPrinting().create().toJson(this.reference.getJson()));
```
Logging: 
```
    {
      "class": "com.simiacryptus.mindseye.layers.aparapi.ConvolutionLayer",
      "id": "47458076-d406-4dcc-89dd-939cf3221d03",
      "isFrozen": false,
      "name": "ConvolutionLayer/47458076-d406-4dcc-89dd-939cf3221d03",
      "filter": [
        [
          [
            -0.984
          ]
        ]
      ],
      "skip": [
        [
          0.0
        ]
      ],
      "simple": true
    }
    
```

Code from [EquivalencyTester.java:64](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/EquivalencyTester.java#L64) executed in 0.01 seconds: 
```java
    return test(subject, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ 0.728 ] ]
    ]
    Error: [
    	[ [ 0.0 ] ]
    ]
    Accuracy:
    absoluteTol: 0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (1#)
    relativeTol: 0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (1#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (1#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (1#)}
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
      "class": "com.simiacryptus.mindseye.layers.cudnn.SimpleConvolutionLayer",
      "id": "4317b409-edc3-4672-bf69-3d9905917f16",
      "isFrozen": false,
      "name": "SimpleConvolutionLayer/4317b409-edc3-4672-bf69-3d9905917f16",
      "filter": [
        [
          [
            -0.984
          ]
        ]
      ],
      "strideX": 1,
      "strideY": 1,
      "simple": false
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
    	[ [ -1.86 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 1.83024 ] ]
    ]
    --------------------
    Derivative: 
    [
    	[ [ -0.984 ] ]
    ]
```



[GPU Log](etc/cuda.log)

### Input Learning
In this test, we use a network to learn this target input, given it's pre-evaluated output:

Code from [LearningTester.java:127](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L127) executed in 0.00 seconds: 
```java
    return Arrays.stream(input_target).map(x -> x.prettyPrint()).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [
    	[ [ 0.248 ], [ 0.28 ], [ -0.068 ], [ 1.392 ], [ -0.032 ], [ -0.516 ], [ -1.032 ], [ 1.96 ], ... ],
    	[ [ 1.804 ], [ -0.724 ], [ -0.084 ], [ 1.864 ], [ -1.752 ], [ -0.676 ], [ 1.388 ], [ -1.008 ], ... ],
    	[ [ 1.88 ], [ -1.872 ], [ -0.236 ], [ -1.732 ], [ -0.684 ], [ -1.412 ], [ 0.136 ], [ -0.032 ], ... ],
    	[ [ 1.184 ], [ -1.692 ], [ -1.088 ], [ -1.932 ], [ -1.212 ], [ -0.256 ], [ -1.824 ], [ 0.384 ], ... ],
    	[ [ -1.044 ], [ -0.616 ], [ 1.496 ], [ -0.068 ], [ 1.696 ], [ -0.508 ], [ -0.456 ], [ -0.168 ], ... ],
    	[ [ -1.924 ], [ 1.3 ], [ -0.624 ], [ -0.856 ], [ 0.472 ], [ -0.504 ], [ 0.652 ], [ 0.072 ], ... ],
    	[ [ -0.704 ], [ -0.176 ], [ -0.792 ], [ -0.452 ], [ -1.44 ], [ -0.4 ], [ 1.7 ], [ -1.94 ], ... ],
    	[ [ -1.468 ], [ 1.376 ], [ 0.34 ], [ -0.16 ], [ -1.072 ], [ 0.084 ], [ 0.496 ], [ 0.544 ], ... ],
    	...
    ]
```



First, we use a conjugate gradient descent method, which converges the fastest for purely linear functions.

Code from [LearningTester.java:225](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L225) executed in 0.22 seconds: 
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
    F(0.0) = LineSearchPoint{point=PointSample{avg=2.577156277176111}, derivative=-9.981388113253747E-4}
    New Minimum: 2.577156277176111 > 2.5771562771760173
    F(1.0E-10) = LineSearchPoint{point=PointSample{avg=2.5771562771760173}, derivative=-9.981388113253554E-4}, delta = -9.370282327836321E-14
    New Minimum: 2.5771562771760173 > 2.5771562771753915
    F(7.000000000000001E-10) = LineSearchPoint{point=PointSample{avg=2.5771562771753915}, derivative=-9.981388113252394E-4}, delta = -7.194245199571014E-13
    New Minimum: 2.5771562771753915 > 2.5771562771712313
    F(4.900000000000001E-9) = LineSearchPoint{point=PointSample{avg=2.5771562771712313}, derivative=-9.981388113244275E-4}, delta = -4.879652237832488E-12
    New Minimum: 2.5771562771712313 > 2.5771562771418894
    F(3.430000000000001E-8) = LineSearchPoint{point=PointSample{avg=2.5771562771418894}, derivative=-9.981388113187448E-4}, delta = -3.4221514511045825E-11
    New Minimum: 2.5771562771418894 > 2.5771562769364724
    F(2.40100000000
```
...[skipping 7729 bytes](etc/144.txt)...
```
    : 0.0004; Line Search: 0.0046
    Zero gradient: 4.016464265943389E-145
    F(0.0) = LineSearchPoint{point=PointSample{avg=4.165216946654646E-286}, derivative=-1.6131985199600165E-289}
    New Minimum: 4.165216946654646E-286 > 8.048453082395E-312
    F(5163.923590456135) = LineSearchPoint{point=PointSample{avg=8.048453082395E-312}, derivative=2.2424622046213372E-302}, delta = -4.165216946654646E-286
    8.048453082395E-312 <= 4.165216946654646E-286
    Converged to right
    Iteration 12 complete. Error: 8.048453082395E-312 Total: 239595646020433.3800; Orientation: 0.0003; Line Search: 0.0064
    Zero gradient: 5.583175796754097E-158
    F(0.0) = LineSearchPoint{point=PointSample{avg=8.048453082395E-312}, derivative=-3.1171852E-315}
    New Minimum: 8.048453082395E-312 > 0.0
    F(5163.923590456135) = LineSearchPoint{point=PointSample{avg=0.0}, derivative=0.0}, delta = -8.048453082395E-312
    0.0 <= 8.048453082395E-312
    Converged to right
    Iteration 13 complete. Error: 0.0 Total: 239595656205560.3400; Orientation: 0.0003; Line Search: 0.0067
    
```

Returns: 

```
    0.0
```



Training Converged

Next, we run the same optimization using L-BFGS, which is nearly ideal for purely second-order or quadratic functions.

Code from [LearningTester.java:249](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L249) executed in 0.12 seconds: 
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
    th(0)=2.577156277176111;dx=-9.981388113253747E-4
    New Minimum: 2.577156277176111 > 2.5750063008836563
    WOLFE (weak): th(2.154434690031884)=2.5750063008836563; dx=-9.977223789667392E-4 delta=0.002149976292454614
    New Minimum: 2.5750063008836563 > 2.5728572217675794
    WOLFE (weak): th(4.308869380063768)=2.5728572217675794; dx=-9.973059466081037E-4 delta=0.0042990554085315935
    New Minimum: 2.5728572217675794 > 2.5642698770663492
    WOLFE (weak): th(12.926608140191302)=2.5642698770663492; dx=-9.95640217173562E-4 delta=0.01288640010976172
    New Minimum: 2.5642698770663492 > 2.5258044668220427
    WOLFE (weak): th(51.70643256076521)=2.5258044668220427; dx=-9.88144434718124E-4 delta=0.05135181035406822
    New Minimum: 2.5258044668220427 > 2.3255649610060996
    WOLFE (weak): th(258.53216280382605)=2.3255649610060996; dx=-9.48166928289121E-4 delta=0.2515913161700114
    New Minimum: 2.3255649610060996 > 1.26139846516715
    END: th(1551.1929768229563)=1.261
```
...[skipping 2527 bytes](etc/145.txt)...
```
    E-7; dx=-1.6231220642850755E-9 delta=3.205706475242274E-5
    Iteration 6 complete. Error: 5.388145143337783E-7 Total: 239595749900185.2500; Orientation: 0.0005; Line Search: 0.0112
    LBFGS Accumulation History: 1 points
    th(0)=5.388145143337783E-7;dx=-2.0868415455630567E-10
    New Minimum: 5.388145143337783E-7 > 4.148336984389806E-7
    WOLF (strong): th(9694.956105143481)=4.148336984389806E-7; dx=1.831078003056623E-10 delta=1.2398081589479769E-7
    New Minimum: 4.148336984389806E-7 > 2.0233807944869963E-9
    END: th(4847.478052571741)=2.0233807944869963E-9; dx=-1.2788177125321394E-11 delta=5.367911335392913E-7
    Iteration 7 complete. Error: 2.0233807944869963E-9 Total: 239595763326657.2500; Orientation: 0.0005; Line Search: 0.0101
    LBFGS Accumulation History: 1 points
    th(0)=2.0233807944869963E-9;dx=-7.836602378187217E-13
    MAX ALPHA: th(0)=2.0233807944869963E-9;th'(0)=-7.836602378187217E-13;
    Iteration 8 failed, aborting. Error: 2.0233807944869963E-9 Total: 239595775004787.2500; Orientation: 0.0006; Line Search: 0.0075
    
```

Returns: 

```
    2.0233807944869963E-9
```



Training Converged

Code from [LearningTester.java:95](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L95) executed in 0.00 seconds: 
```java
    return TestUtil.compare(runs);
```

Returns: 

![Result](etc/test.86.png)



Code from [LearningTester.java:98](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L98) executed in 0.00 seconds: 
```java
    return TestUtil.compareTime(runs);
```

Returns: 

![Result](etc/test.87.png)



### Model Learning
In this test, attempt to train a network to emulate a randomized network given an example input/output. The target state is:

Code from [LearningTester.java:176](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L176) executed in 0.00 seconds: 
```java
    return network_target.state().stream().map(Arrays::toString).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [-0.984]
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

Returns: 

```
    0.0
```



This training run resulted in the following configuration:

Code from [LearningTester.java:189](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L189) executed in 0.00 seconds: 
```java
    return network_gd.state().stream().map(Arrays::toString).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [-0.984]
    [-0.7872, 0.901344, 1.5744, -1.1650559999999999, 1.8577919999999999, 1.9325759999999998, 0.448704, 1.609824, -1.263456, 0.724224, 1.420896, -0.188928, 0.495936, -0.153504, -1.365792, 0.08265600000000001, -0.72816, -0.6769919999999999, 1.676736, -0.10627199999999999, 1.598016, -1.936512, -0.645504, -0.7832640000000001, 1.251648, 1.235904, 0.547104, -0.657312, -1.117824, 1.1650559999999999, 1.1217599999999999, -0.16924799999999998, -0.08659199999999999, -0.82656, 1.302816, 1.5744, 1.271328, 0.110208, -0.181056, 0.251904, 1.0469760000000001, -0.031488, -1.519296, 0.21648, 0.035424, -0.35424, 1.6413119999999999, -1.63344, -0.72816, -0.617952, -0.11414400000000001, -0.34636799999999995, 0.621888, -1.708224, -0.47231999999999996, -0.594336, -1.06272, 1.3815359999999999, -1.444512, -1.747584, -1.389408, -1.7987520000000001, -1.168992, -0.397536, -0.066912, 1.3224960000000001, -0.6769919999999999, 1.763328, 1.511424, 0.5628479999999999, -1.06272, -1.0824, -0.287328, -0.21648, -0.74784, 0.586464, 0.07872, 1.3
```
...[skipping 125060 bytes](etc/146.txt)...
```
    2, -0.9525119999999999, -0.940704, 0.283392, 1.8459839999999998, 0.220416, 1.806624, 0.059039999999999995, 0.586464, 1.3854719999999998, -0.47231999999999996, 1.3776, -1.8696, -1.015488, -0.72816, -0.6612480000000001, -0.267648, 0.9800639999999999, 0.47231999999999996, 0.11414400000000001, 1.25952, -0.637632, -1.542912, -0.873792, 0.810816, -1.25952, 0.247968, -0.960384, -0.940704, 1.157184, -1.235904, -0.743904, -0.51168, 0.16924799999999998, -0.507744, 1.8892799999999998, -0.488064, -0.279456, 0.8068799999999999, 0.0, 1.1375039999999998, -0.299136, 0.976128, -0.940704, -0.889536, 0.33456, 1.527168, -1.7712, -0.680928, 1.8499199999999998, 0.8147519999999999, 0.566784, -0.6927359999999999, -0.279456, 0.13776000000000002, 1.605888, 0.22828800000000002, -0.015744, -1.039104, 1.901088, -0.110208, -1.464192, -0.318816, 0.799008, -0.495936, -0.917088, 1.476, 0.673056, -1.302816, -1.015488, -0.755712, -0.82656, 0.23222399999999999, -0.8147519999999999, -0.578592, -0.881664, 0.802944, -1.664928, -1.1375039999999998]
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

Returns: 

```
    0.0
```



This training run resulted in the following configuration:

Code from [LearningTester.java:203](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L203) executed in 0.00 seconds: 
```java
    return network_lbfgs.state().stream().map(Arrays::toString).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [-0.984]
    [-0.7872, 0.901344, 1.5744, -1.1650559999999999, 1.8577919999999999, 1.9325759999999998, 0.448704, 1.609824, -1.263456, 0.724224, 1.420896, -0.188928, 0.495936, -0.153504, -1.365792, 0.08265600000000001, -0.72816, -0.6769919999999999, 1.676736, -0.10627199999999999, 1.598016, -1.936512, -0.645504, -0.7832640000000001, 1.251648, 1.235904, 0.547104, -0.657312, -1.117824, 1.1650559999999999, 1.1217599999999999, -0.16924799999999998, -0.08659199999999999, -0.82656, 1.302816, 1.5744, 1.271328, 0.110208, -0.181056, 0.251904, 1.0469760000000001, -0.031488, -1.519296, 0.21648, 0.035424, -0.35424, 1.6413119999999999, -1.63344, -0.72816, -0.617952, -0.11414400000000001, -0.34636799999999995, 0.621888, -1.708224, -0.47231999999999996, -0.594336, -1.06272, 1.3815359999999999, -1.444512, -1.747584, -1.389408, -1.7987520000000001, -1.168992, -0.397536, -0.066912, 1.3224960000000001, -0.6769919999999999, 1.763328, 1.511424, 0.5628479999999999, -1.06272, -1.0824, -0.287328, -0.21648, -0.74784, 0.586464, 0.07872, 1.3
```
...[skipping 125060 bytes](etc/147.txt)...
```
    2, -0.9525119999999999, -0.940704, 0.283392, 1.8459839999999998, 0.220416, 1.806624, 0.059039999999999995, 0.586464, 1.3854719999999998, -0.47231999999999996, 1.3776, -1.8696, -1.015488, -0.72816, -0.6612480000000001, -0.267648, 0.9800639999999999, 0.47231999999999996, 0.11414400000000001, 1.25952, -0.637632, -1.542912, -0.873792, 0.810816, -1.25952, 0.247968, -0.960384, -0.940704, 1.157184, -1.235904, -0.743904, -0.51168, 0.16924799999999998, -0.507744, 1.8892799999999998, -0.488064, -0.279456, 0.8068799999999999, 0.0, 1.1375039999999998, -0.299136, 0.976128, -0.940704, -0.889536, 0.33456, 1.527168, -1.7712, -0.680928, 1.8499199999999998, 0.8147519999999999, 0.566784, -0.6927359999999999, -0.279456, 0.13776000000000002, 1.605888, 0.22828800000000002, -0.015744, -1.039104, 1.901088, -0.110208, -1.464192, -0.318816, 0.799008, -0.495936, -0.917088, 1.476, 0.673056, -1.302816, -1.015488, -0.755712, -0.82656, 0.23222399999999999, -0.8147519999999999, -0.578592, -0.881664, 0.802944, -1.664928, -1.1375039999999998]
```



Code from [LearningTester.java:95](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L95) executed in 0.00 seconds: 
```java
    return TestUtil.compare(runs);
```

Code from [LearningTester.java:98](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L98) executed in 0.00 seconds: 
```java
    return TestUtil.compareTime(runs);
```

### Performance
Now we execute larger-scale runs to benchmark performance:

Code from [PerformanceTester.java:66](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/PerformanceTester.java#L66) executed in 0.37 seconds: 
```java
    test(component, inputPrototype);
```
Logging: 
```
    100 batches
    Input Dimensions:
    	[100, 100, 1]
    Performance:
    	Evaluation performance: 0.007610s +- 0.000296s [0.007113s - 0.008017s]
    	Learning performance: 0.057109s +- 0.015353s [0.036570s - 0.073752s]
    
```

