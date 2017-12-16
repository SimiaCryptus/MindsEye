# MonitoringSynapse
## MonitoringSynapseTest
### Batch Execution
Code from [BatchingTester.java:66](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/BatchingTester.java#L66) executed in 0.00 seconds: 
```java
    return test(reference, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (60#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (60#)}
```



Code from [SingleDerivativeTester.java:77](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/SingleDerivativeTester.java#L77) executed in 0.00 seconds: 
```java
    return test(component, inputPrototype);
```
Logging: 
```
    Inputs: [ -1.744, 0.264, -0.936 ]
    Inputs Statistics: {meanExponent=-0.1218579145985051, negative=2, min=-0.936, max=-0.936, mean=-0.8053333333333333, count=3.0, positive=1, stdDev=0.8249530626378421, zeros=0}
    Output: [ -1.744, 0.264, -0.936 ]
    Outputs Statistics: {meanExponent=-0.1218579145985051, negative=2, min=-0.936, max=-0.936, mean=-0.8053333333333333, count=3.0, positive=1, stdDev=0.8249530626378421, zeros=0}
    Feedback for input 0
    Inputs Values: [ -1.744, 0.264, -0.936 ]
    Value Statistics: {meanExponent=-0.1218579145985051, negative=2, min=-0.936, max=-0.936, mean=-0.8053333333333333, count=3.0, positive=1, stdDev=0.8249530626378421, zeros=0}
    Implemented Feedback: [ [ 1.0, 0.0, 0.0 ], [ 0.0, 1.0, 0.0 ], [ 0.0, 0.0, 1.0 ] ]
    Implemented Statistics: {meanExponent=0.0, negative=0, min=1.0, max=1.0, mean=0.3333333333333333, count=9.0, positive=3, stdDev=0.4714045207910317, zeros=6}
    Measured Feedback: [ [ 0.9999999999998899, 0.0, 0.0 ], [ 0.0, 0.9999999999998899, 0.0 ], [ 0.0, 0.0, 0.9999999999998899 ] ]
    Measured Statistics: {meanExponent=-4.7830642341045674E-14, negative=0, min=0.9999999999998899, max=0.9999999999998899, mean=0.3333333333332966, count=9.0, positive=3, stdDev=0.4714045207909798, zeros=6}
    Feedback Error: [ [ -1.1013412404281553E-13, 0.0, 0.0 ], [ 0.0, -1.1013412404281553E-13, 0.0 ], [ 0.0, 0.0, -1.1013412404281553E-13 ] ]
    Error Statistics: {meanExponent=-12.958078098036827, negative=3, min=-1.1013412404281553E-13, max=-1.1013412404281553E-13, mean=-3.671137468093851E-14, count=9.0, positive=0, stdDev=5.1917723967143496E-14, zeros=6}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 3.6711e-14 +- 5.1918e-14 [0.0000e+00 - 1.1013e-13] (9#)
    relativeTol: 5.5067e-14 +- 0.0000e+00 [5.5067e-14 - 5.5067e-14] (3#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=3.6711e-14 +- 5.1918e-14 [0.0000e+00 - 1.1013e-13] (9#), relativeTol=5.5067e-14 +- 0.0000e+00 [5.5067e-14 - 5.5067e-14] (3#)}
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
      "class": "com.simiacryptus.mindseye.layers.java.MonitoringSynapse",
      "id": "4f3c4e56-cf80-4d39-9b82-829b33e9508b",
      "isFrozen": false,
      "name": "MonitoringSynapse/4f3c4e56-cf80-4d39-9b82-829b33e9508b",
      "totalBatches": 0,
      "totalItems": 0
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
    [[ -1.672, 0.152, 0.708 ]]
    --------------------
    Output: 
    [ -1.672, 0.152, 0.708 ]
    --------------------
    Derivative: 
    [ 1.0, 1.0, 1.0 ]
```



### Input Learning
In this test, we use a network to learn this target input, given it's pre-evaluated output:

Code from [LearningTester.java:127](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L127) executed in 0.00 seconds: 
```java
    return Arrays.stream(input_target).map(x -> x.prettyPrint()).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [ 1.572, -0.764, 0.84 ]
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
    F(0.0) = LineSearchPoint{point=PointSample{avg=0.35721600000000014}, derivative=-0.4762880000000001}
    New Minimum: 0.35721600000000014 > 0.35721599995237124
    F(1.0E-10) = LineSearchPoint{point=PointSample{avg=0.35721599995237124}, derivative=-0.47628799996824756}, delta = -4.76289008233266E-11
    New Minimum: 0.35721599995237124 > 0.35721599966659845
    F(7.000000000000001E-10) = LineSearchPoint{point=PointSample{avg=0.35721599966659845}, derivative=-0.4762879997777323}, delta = -3.334016951406227E-10
    New Minimum: 0.35721599966659845 > 0.357215997666189
    F(4.900000000000001E-9) = LineSearchPoint{point=PointSample{avg=0.357215997666189}, derivative=-0.47628799844412595}, delta = -2.3338111443393927E-9
    New Minimum: 0.357215997666189 > 0.3572159836633219
    F(3.430000000000001E-8) = LineSearchPoint{point=PointSample{avg=0.3572159836633219}, derivative=-0.47628798910888115}, delta = -1.6336678232420354E-8
    New Minimum: 0.3572159836633219 > 0.3572158856432604
    F(2.4010000000000
```
...[skipping 1053 bytes](etc/291.txt)...
```
    t{point=PointSample{avg=0.35529659143577086}, derivative=-0.47500667074861236}, delta = -0.0019194085642292857
    New Minimum: 0.35529659143577086 > 0.34388872319019675
    F(0.028247524900000005) = LineSearchPoint{point=PointSample{avg=0.34388872319019675}, derivative=-0.46731869524028596}, delta = -0.01332727680980339
    New Minimum: 0.34388872319019675 > 0.26924563618162667
    F(0.19773267430000002) = LineSearchPoint{point=PointSample{avg=0.26924563618162667}, derivative=-0.41350286668200115}, delta = -0.08797036381837348
    New Minimum: 0.26924563618162667 > 0.0021315719336352508
    F(1.3841287201) = LineSearchPoint{point=PointSample{avg=0.0021315719336352508}, derivative=-0.03679206677400743}, delta = -0.3550844280663649
    Loops = 12
    New Minimum: 0.0021315719336352508 > 0.0
    F(1.5) = LineSearchPoint{point=PointSample{avg=0.0}, derivative=0.0}, delta = -0.35721600000000014
    Right bracket at 1.5
    Converged to right
    Iteration 1 complete. Error: 0.0 Total: 239673951547396.0600; Orientation: 0.0000; Line Search: 0.0013
    
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
    th(0)=0.35721600000000014;dx=-0.4762880000000001
    New Minimum: 0.35721600000000014 > 0.0679956311486819
    WOLF (strong): th(2.154434690031884)=0.0679956311486819; dx=0.20779959309727059 delta=0.28922036885131824
    New Minimum: 0.0679956311486819 > 0.028378060375694025
    END: th(1.077217345015942)=0.028378060375694025; dx=-0.13424420345136473 delta=0.32883793962430613
    Iteration 1 complete. Error: 0.028378060375694025 Total: 239673956570141.0300; Orientation: 0.0002; Line Search: 0.0003
    LBFGS Accumulation History: 1 points
    th(0)=0.028378060375694025;dx=-0.0378374138342587
    New Minimum: 0.028378060375694025 > 0.008497065724102972
    WOLF (strong): th(2.3207944168063896)=0.008497065724102972; dx=0.020704492014368255 delta=0.01988099465159105
    New Minimum: 0.008497065724102972 > 0.001454597019561149
    END: th(1.1603972084031948)=0.001454597019561149; dx=-0.00856646090994521 delta=0.026923463356132875
    Iteration 2 complete. Error: 0.0014545
```
...[skipping 9132 bytes](etc/292.txt)...
```
    : 239673962250900.0300; Orientation: 0.0000; Line Search: 0.0002
    LBFGS Accumulation History: 1 points
    th(0)=8.414516322357459E-30;dx=-1.1219355096476612E-29
    New Minimum: 8.414516322357459E-30 > 4.733165431326071E-30
    WOLF (strong): th(2.623149071368624)=4.733165431326071E-30; dx=8.414516322357459E-30 delta=3.681350891031388E-30
    New Minimum: 4.733165431326071E-30 > 1.314768175368353E-31
    END: th(1.311574535684312)=1.314768175368353E-31; dx=-1.4024193870595765E-30 delta=8.283039504820624E-30
    Iteration 20 complete. Error: 1.314768175368353E-31 Total: 239673962614247.0300; Orientation: 0.0000; Line Search: 0.0003
    LBFGS Accumulation History: 1 points
    th(0)=1.314768175368353E-31;dx=-1.7530242338244706E-31
    Armijo: th(2.8257016782407427)=1.314768175368353E-31; dx=1.7530242338244706E-31 delta=0.0
    New Minimum: 1.314768175368353E-31 > 0.0
    END: th(1.4128508391203713)=0.0; dx=0.0 delta=1.314768175368353E-31
    Iteration 21 complete. Error: 0.0 Total: 239673962955366.0300; Orientation: 0.0000; Line Search: 0.0002
    
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

![Result](etc/test.196.png)



Code from [LearningTester.java:98](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L98) executed in 0.00 seconds: 
```java
    return TestUtil.compareTime(runs);
```

Returns: 

![Result](etc/test.197.png)



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
    	[3]
    Performance:
    	Evaluation performance: 0.000261s +- 0.000166s [0.000166s - 0.000592s]
    	Learning performance: 0.000178s +- 0.000018s [0.000159s - 0.000211s]
    
```

