# ImgReshapeLayer
## ImgReshapeLayerTest
### Batch Execution
Code from [BatchingTester.java:66](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/BatchingTester.java#L66) executed in 0.00 seconds: 
```java
    return test(reference, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (240#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (240#)}
```



Code from [SingleDerivativeTester.java:77](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/SingleDerivativeTester.java#L77) executed in 0.00 seconds: 
```java
    return test(component, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ -1.68, 1.568, -1.508 ], [ -0.908, 1.732, -1.44 ] ],
    	[ [ -1.888, 0.028, 1.108 ], [ 1.824, -1.964, -1.14 ] ]
    ]
    Inputs Statistics: {meanExponent=0.027735321698515128, negative=7, min=-1.14, max=-1.14, mean=-0.35566666666666663, count=12.0, positive=5, stdDev=1.4494784885912892, zeros=0}
    Output: [
    	[ [ -1.68, 1.568, -1.508, -0.908, 1.732, -1.44, -1.888, 0.028, ... ] ]
    ]
    Outputs Statistics: {meanExponent=0.027735321698515124, negative=7, min=-1.14, max=-1.14, mean=-0.35566666666666663, count=12.0, positive=5, stdDev=1.4494784885912892, zeros=0}
    Feedback for input 0
    Inputs Values: [
    	[ [ -1.68, 1.568, -1.508 ], [ -0.908, 1.732, -1.44 ] ],
    	[ [ -1.888, 0.028, 1.108 ], [ 1.824, -1.964, -1.14 ] ]
    ]
    Value Statistics: {meanExponent=0.027735321698515128, negative=7, min=-1.14, max=-1.14, mean=-0.35566666666666663, count=12.0, positive=5, stdDev=1.4494784885912892, zeros=0}
    Implemented Feedback: [ [ 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, ... ], [ 0.0
```
...[skipping 1096 bytes](etc/260.txt)...
```
    08333333333332488, count=144.0, positive=12, stdDev=0.27638539919625527, zeros=132}
    Feedback Error: [ [ -1.1013412404281553E-13, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.1013412404281553E-13, 0.0, ... ], [ 0.0, 0.0, 0.0, -1.1013412404281553E-13, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, -1.1013412404281553E-13, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -5.995204332975845E-15, ... ], [ 0.0, 0.0, 0.0, 0.0, -1.1013412404281553E-13, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], ... ]
    Error Statistics: {meanExponent=-13.063421257397758, negative=12, min=-1.1013412404281553E-13, max=-1.1013412404281553E-13, mean=-8.454656727805184E-15, count=144.0, positive=0, stdDev=2.924601406899229E-14, zeros=132}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 8.4547e-15 +- 2.9246e-14 [0.0000e+00 - 1.1013e-13] (144#)
    relativeTol: 5.0728e-14 +- 1.4391e-14 [2.9976e-15 - 5.5067e-14] (12#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=8.4547e-15 +- 2.9246e-14 [0.0000e+00 - 1.1013e-13] (144#), relativeTol=5.0728e-14 +- 1.4391e-14 [2.9976e-15 - 5.5067e-14] (12#)}
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
      "class": "com.simiacryptus.mindseye.layers.java.ImgReshapeLayer",
      "id": "a2191006-91a3-4977-bd4f-acb2474d1b66",
      "isFrozen": false,
      "name": "ImgReshapeLayer/a2191006-91a3-4977-bd4f-acb2474d1b66",
      "kernelSizeX": 2,
      "kernelSizeY": 2,
      "expand": false
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
    	[ [ -1.012, -0.936, 0.248 ], [ -0.672, 0.356, -1.416 ] ],
    	[ [ 0.656, -0.252, -0.512 ], [ -1.812, 1.812, 0.7 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ -1.012, -0.936, 0.248, -0.672, 0.356, -1.416, 0.656, -0.252, ... ] ]
    ]
    --------------------
    Derivative: 
    [
    	[ [ 1.0, 1.0, 1.0 ], [ 1.0, 1.0, 1.0 ] ],
    	[ [ 1.0, 1.0, 1.0 ], [ 1.0, 1.0, 1.0 ] ]
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
    	[ [ 0.36, 1.204, 0.576 ], [ -1.256, 0.172, 1.732 ] ],
    	[ [ -1.344, 1.984, 0.6 ], [ -0.084, 1.22, -1.216 ] ]
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
    F(0.0) = LineSearchPoint{point=PointSample{avg=2.360186666666667}, derivative=-0.7867288888888888}
    New Minimum: 2.360186666666667 > 2.360186666587994
    F(1.0E-10) = LineSearchPoint{point=PointSample{avg=2.360186666587994}, derivative=-0.7867288888757766}, delta = -7.867306806019769E-11
    New Minimum: 2.360186666587994 > 2.3601866661159563
    F(7.000000000000001E-10) = LineSearchPoint{point=PointSample{avg=2.3601866661159563}, derivative=-0.7867288887971038}, delta = -5.507105882429641E-10
    New Minimum: 2.3601866661159563 > 2.360186662811695
    F(4.900000000000001E-9) = LineSearchPoint{point=PointSample{avg=2.360186662811695}, derivative=-0.7867288882463935}, delta = -3.8549718972547E-9
    New Minimum: 2.360186662811695 > 2.360186639681866
    F(3.430000000000001E-8) = LineSearchPoint{point=PointSample{avg=2.360186639681866}, derivative=-0.7867288843914221}, delta = -2.698480106033685E-8
    New Minimum: 2.360186639681866 > 2.3601864777730643
    F(2.4010000000000004E-7) = LineSearchPo
```
...[skipping 1546 bytes](etc/261.txt)...
```
    87967158687 > 1.3968546910365254
    F(1.3841287201) = LineSearchPoint{point=PointSample{avg=1.3968546910365254}, derivative=-0.60523988054831}, delta = -0.9633319756301415
    Loops = 12
    New Minimum: 1.3968546910365254 > 6.477544379622403E-32
    F(6.000000000000001) = LineSearchPoint{point=PointSample{avg=6.477544379622403E-32}, derivative=8.35350307445045E-17}, delta = -2.360186666666667
    Right bracket at 6.000000000000001
    Converged to right
    Iteration 1 complete. Error: 6.477544379622403E-32 Total: 239662326159766.7000; Orientation: 0.0000; Line Search: 0.0020
    Zero gradient: 1.4694153462769246E-16
    F(0.0) = LineSearchPoint{point=PointSample{avg=6.477544379622403E-32}, derivative=-2.1591814598741342E-32}
    New Minimum: 6.477544379622403E-32 > 0.0
    F(6.000000000000001) = LineSearchPoint{point=PointSample{avg=0.0}, derivative=0.0}, delta = -6.477544379622403E-32
    0.0 <= 6.477544379622403E-32
    Converged to right
    Iteration 2 complete. Error: 0.0 Total: 239662326587233.7000; Orientation: 0.0000; Line Search: 0.0003
    
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
    th(0)=2.360186666666667;dx=-0.7867288888888888
    New Minimum: 2.360186666666667 > 0.9695366589398096
    END: th(2.154434690031884)=0.9695366589398096; dx=-0.5042362205768118 delta=1.3906500077268573
    Iteration 1 complete. Error: 0.9695366589398096 Total: 239662331497696.6600; Orientation: 0.0001; Line Search: 0.0003
    LBFGS Accumulation History: 1 points
    th(0)=0.9695366589398096;dx=-0.32317888631326985
    New Minimum: 0.9695366589398096 > 0.04969631876803823
    END: th(4.641588833612779)=0.04969631876803823; dx=-0.07316830131808866 delta=0.9198403401717713
    Iteration 2 complete. Error: 0.04969631876803823 Total: 239662331836535.6600; Orientation: 0.0000; Line Search: 0.0002
    LBFGS Accumulation History: 1 points
    th(0)=0.04969631876803823;dx=-0.016565439589346077
    New Minimum: 0.04969631876803823 > 0.022087252785794792
    WOLF (strong): th(10.000000000000002)=0.022087252785794792; dx=0.011043626392897389 delta=0.02760906598224344
    New Minimu
```
...[skipping 9319 bytes](etc/262.txt)...
```
    al: 239662341340280.6600; Orientation: 0.0000; Line Search: 0.0003
    LBFGS Accumulation History: 1 points
    th(0)=4.4456240906290565E-30;dx=-1.481874696876352E-30
    New Minimum: 4.4456240906290565E-30 > 3.564254350412645E-30
    WOLF (strong): th(11.30280671296297)=3.564254350412645E-30; dx=1.326580545693928E-30 delta=8.813697402164118E-31
    New Minimum: 3.564254350412645E-30 > 1.855312513093037E-32
    END: th(5.651403356481485)=1.855312513093037E-32; dx=-9.263723032502603E-32 delta=4.4270709654981264E-30
    Iteration 21 complete. Error: 1.855312513093037E-32 Total: 239662341868914.6600; Orientation: 0.0000; Line Search: 0.0004
    LBFGS Accumulation History: 1 points
    th(0)=1.855312513093037E-32;dx=-6.184375043643457E-33
    Armijo: th(12.175579438566336)=1.855312513093037E-32; dx=6.184375043643457E-33 delta=0.0
    New Minimum: 1.855312513093037E-32 > 0.0
    END: th(6.087789719283168)=0.0; dx=0.0 delta=1.855312513093037E-32
    Iteration 22 complete. Error: 0.0 Total: 239662342256770.6600; Orientation: 0.0000; Line Search: 0.0003
    
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

![Result](etc/test.168.png)



Code from [LearningTester.java:98](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L98) executed in 0.00 seconds: 
```java
    return TestUtil.compareTime(runs);
```

Returns: 

![Result](etc/test.169.png)



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
    	[2, 2, 3]
    Performance:
    	Evaluation performance: 0.000392s +- 0.000076s [0.000341s - 0.000542s]
    	Learning performance: 0.000061s +- 0.000011s [0.000047s - 0.000075s]
    
```

