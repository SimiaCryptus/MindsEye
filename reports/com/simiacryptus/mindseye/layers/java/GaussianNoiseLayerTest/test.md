# GaussianNoiseLayer
## GaussianNoiseLayerTest
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
    	[ [ 1.16 ], [ -1.216 ], [ 1.332 ] ],
    	[ [ -1.768 ], [ -1.152 ], [ -0.668 ] ]
    ]
    Inputs Statistics: {meanExponent=0.06793449853961835, negative=4, min=-0.668, max=-0.668, mean=-0.3853333333333333, count=6.0, positive=2, stdDev=1.1976881434200177, zeros=0}
    Output: [
    	[ [ 0.8032092603326616 ], [ -0.5191241728545672 ], [ 2.7439503568683588 ] ],
    	[ [ -2.1970630325148934 ], [ -0.9397456414481602 ], [ 0.8993384936766019 ] ]
    ]
    Outputs Statistics: {meanExponent=0.0545420380674064, negative=3, min=0.8993384936766019, max=0.8993384936766019, mean=0.13176087734333355, count=6.0, positive=3, stdDev=1.5736767232909907, zeros=0}
    Feedback for input 0
    Inputs Values: [
    	[ [ 1.16 ], [ -1.216 ], [ 1.332 ] ],
    	[ [ -1.768 ], [ -1.152 ], [ -0.668 ] ]
    ]
    Value Statistics: {meanExponent=0.06793449853961835, negative=4, min=-0.668, max=-0.668, mean=-0.3853333333333333, count=6.0, positive=2, stdDev=1.1976881434200177, zeros=0}
    Implemented Feedback: [ [ 1.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 1.0, 0.0, 0.0, 0.0, 0.0 ], [
```
...[skipping 540 bytes](etc/234.txt)...
```
    0000000000021103, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.9999999999998899 ] ]
    Measured Statistics: {meanExponent=2.7361184650972856E-13, negative=0, min=0.9999999999998899, max=0.9999999999998899, mean=0.16666666666677166, count=36.0, positive=6, stdDev=0.37267799625019976, zeros=30}
    Feedback Error: [ [ -1.1013412404281553E-13, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 2.1103119252074976E-12, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, -1.1013412404281553E-13, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, -1.1013412404281553E-13, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 2.1103119252074976E-12, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, -1.1013412404281553E-13 ] ]
    Error Statistics: {meanExponent=-12.530603180987852, negative=4, min=-1.1013412404281553E-13, max=-1.1013412404281553E-13, mean=1.0500242650677037E-13, count=36.0, positive=2, stdDev=4.875799820387737E-13, zeros=30}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 1.2948e-13 +- 4.8166e-13 [0.0000e+00 - 2.1103e-12] (36#)
    relativeTol: 3.8843e-13 +- 4.7145e-13 [5.5067e-14 - 1.0552e-12] (6#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=1.2948e-13 +- 4.8166e-13 [0.0000e+00 - 2.1103e-12] (36#), relativeTol=3.8843e-13 +- 4.7145e-13 [5.5067e-14 - 1.0552e-12] (6#)}
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
      "class": "com.simiacryptus.mindseye.layers.java.GaussianNoiseLayer",
      "id": "7e48a729-e341-4d10-84f0-c456b067db85",
      "isFrozen": false,
      "name": "GaussianNoiseLayer/7e48a729-e341-4d10-84f0-c456b067db85",
      "value": 1.0
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
    	[ [ -1.0 ], [ -1.576 ], [ 0.46 ] ],
    	[ [ -0.032 ], [ -0.812 ], [ 1.64 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ -1.2069500965251498 ], [ -0.9809058452263564 ], [ -0.009139376602160132 ] ],
    	[ [ 0.3256117094355263 ], [ 0.4220415449662329 ], [ 3.024465971659243 ] ]
    ]
    --------------------
    Derivative: 
    [
    	[ [ 1.0 ], [ 1.0 ], [ 1.0 ] ],
    	[ [ 1.0 ], [ 1.0 ], [ 1.0 ] ]
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
    	[ [ -1.68 ], [ 0.568 ], [ 0.236 ], [ -1.992 ], [ 0.444 ], [ 1.796 ], [ 1.456 ], [ 1.136 ], ... ],
    	[ [ -1.676 ], [ -1.876 ], [ 0.5 ], [ -0.912 ], [ -1.56 ], [ -0.116 ], [ -0.944 ], [ -1.864 ], ... ],
    	[ [ 1.012 ], [ -0.792 ], [ -1.032 ], [ 0.58 ], [ 0.588 ], [ 0.12 ], [ 1.628 ], [ 0.92 ], ... ],
    	[ [ -1.296 ], [ -1.192 ], [ -1.64 ], [ -1.532 ], [ -1.208 ], [ -0.592 ], [ -0.556 ], [ 0.396 ], ... ],
    	[ [ 0.708 ], [ -1.408 ], [ 0.26 ], [ 1.072 ], [ -1.88 ], [ 0.016 ], [ 1.216 ], [ 1.672 ], ... ],
    	[ [ 0.328 ], [ 0.292 ], [ 0.136 ], [ -0.144 ], [ 0.748 ], [ -1.356 ], [ -0.936 ], [ -0.3 ], ... ],
    	[ [ -0.26 ], [ 0.752 ], [ -0.392 ], [ -1.22 ], [ 1.788 ], [ -1.484 ], [ 0.916 ], [ 0.116 ], ... ],
    	[ [ -0.696 ], [ -1.848 ], [ 0.104 ], [ 1.26 ], [ -1.732 ], [ 1.216 ], [ 0.916 ], [ -0.78 ], ... ],
    	...
    ]
```



First, we use a conjugate gradient descent method, which converges the fastest for purely linear functions.

Code from [LearningTester.java:225](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L225) executed in 0.11 seconds: 
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
    F(0.0) = LineSearchPoint{point=PointSample{avg=2.6778474719999967}, derivative=-0.0010711389888000001}
    New Minimum: 2.6778474719999967 > 2.6778474719998853
    F(1.0E-10) = LineSearchPoint{point=PointSample{avg=2.6778474719998853}, derivative=-0.0010711389887999787}, delta = -1.1146639167236572E-13
    New Minimum: 2.6778474719998853 > 2.677847471999244
    F(7.000000000000001E-10) = LineSearchPoint{point=PointSample{avg=2.677847471999244}, derivative=-0.00107113898879985}, delta = -7.527312106958561E-13
    New Minimum: 2.677847471999244 > 2.6778474719947427
    F(4.900000000000001E-9) = LineSearchPoint{point=PointSample{avg=2.6778474719947427}, derivative=-0.0010711389887989504}, delta = -5.254019441736091E-12
    New Minimum: 2.6778474719947427 > 2.6778474719632674
    F(3.430000000000001E-8) = LineSearchPoint{point=PointSample{avg=2.6778474719632674}, derivative=-0.001071138988792652}, delta = -3.672928627906913E-11
    New Minimum: 2.6778474719632674 > 2.6778474717428074
    F(2.4010000000
```
...[skipping 2430 bytes](etc/235.txt)...
```
    intSample{avg=4.498683460596925E-33}, derivative=6.329279872736993E-35}, delta = -4.501151565542294E-26
    4.498683460596925E-33 <= 4.50115201541064E-26
    Converged to right
    Iteration 2 complete. Error: 4.498683460596925E-33 Total: 239655793700449.2200; Orientation: 0.0004; Line Search: 0.0035
    Zero gradient: 1.3414445140365552E-18
    F(0.0) = LineSearchPoint{point=PointSample{avg=4.498683460596925E-33}, derivative=-1.79947338423877E-36}
    New Minimum: 4.498683460596925E-33 > 4.4789619379663996E-33
    F(4999.9999999993515) = LineSearchPoint{point=PointSample{avg=4.4789619379663996E-33}, derivative=1.79158477518656E-36}, delta = -1.9721522630525176E-35
    4.4789619379663996E-33 <= 4.498683460596925E-33
    New Minimum: 4.4789619379663996E-33 > 0.0
    F(2505.4918416115306) = LineSearchPoint{point=PointSample{avg=0.0}, derivative=0.0}, delta = -4.498683460596925E-33
    Right bracket at 2505.4918416115306
    Converged to right
    Iteration 3 complete. Error: 0.0 Total: 239655800625994.2200; Orientation: 0.0003; Line Search: 0.0056
    
```

Returns: 

```
    0.0
```



Training Converged

Next, we run the same optimization using L-BFGS, which is nearly ideal for purely second-order or quadratic functions.

Code from [LearningTester.java:249](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L249) executed in 0.06 seconds: 
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
    th(0)=2.6778474719999967;dx=-0.0010711389888000001
    New Minimum: 2.6778474719999967 > 2.675540270183372
    WOLFE (weak): th(2.154434690031884)=2.675540270183372; dx=-0.0010706774490009368 delta=0.0023072018166248043
    New Minimum: 2.675540270183372 > 2.6732340627240765
    WOLFE (weak): th(4.308869380063768)=2.6732340627240765; dx=-0.0010702159092018735 delta=0.004613409275920244
    New Minimum: 2.6732340627240765 > 2.6640191764604637
    WOLFE (weak): th(12.926608140191302)=2.6640191764604637; dx=-0.0010683697500056203 delta=0.01382829553953302
    New Minimum: 2.6640191764604637 > 2.622749071030354
    WOLFE (weak): th(51.70643256076521)=2.622749071030354; dx=-0.0010600620336224814 delta=0.05509840096964291
    New Minimum: 2.622749071030354 > 2.408082965510337
    WOLFE (weak): th(258.53216280382605)=2.408082965510337; dx=-0.0010157542129124067 delta=0.2697645064896599
    New Minimum: 2.408082965510337 > 1.2740416215114427
    END: th(1551.1929768229563)=1
```
...[skipping 2542 bytes](etc/236.txt)...
```
    5.988018663168456E-10 delta=1.4820346191342109E-5
    Iteration 6 complete. Error: 1.4970046657920986E-7 Total: 239655853270052.1600; Orientation: 0.0005; Line Search: 0.0051
    LBFGS Accumulation History: 1 points
    th(0)=1.4970046657920986E-7;dx=-5.988018663168411E-11
    New Minimum: 1.4970046657920986E-7 > 1.3199157700639085E-7
    WOLF (strong): th(9694.956105143481)=1.3199157700639085E-7; dx=5.622696956071165E-11 delta=1.7708889572819012E-8
    New Minimum: 1.3199157700639085E-7 > 1.392989455107056E-10
    END: th(4847.478052571741)=1.392989455107056E-10; dx=-1.8266085354862288E-12 delta=1.4956116763369915E-7
    Iteration 7 complete. Error: 1.392989455107056E-10 Total: 239655859983288.1600; Orientation: 0.0005; Line Search: 0.0052
    LBFGS Accumulation History: 1 points
    th(0)=1.392989455107056E-10;dx=-5.5719578204282355E-14
    MAX ALPHA: th(0)=1.392989455107056E-10;th'(0)=-5.5719578204282355E-14;
    Iteration 8 failed, aborting. Error: 1.392989455107056E-10 Total: 239655864560611.1200; Orientation: 0.0005; Line Search: 0.0028
    
```

Returns: 

```
    1.392989455107056E-10
```



Training Converged

Code from [LearningTester.java:95](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L95) executed in 0.00 seconds: 
```java
    return TestUtil.compare(runs);
```

Returns: 

![Result](etc/test.146.png)



Code from [LearningTester.java:98](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L98) executed in 0.00 seconds: 
```java
    return TestUtil.compareTime(runs);
```

Returns: 

![Result](etc/test.147.png)



### Performance
Now we execute larger-scale runs to benchmark performance:

Code from [PerformanceTester.java:66](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/PerformanceTester.java#L66) executed in 0.70 seconds: 
```java
    test(component, inputPrototype);
```
Logging: 
```
    100 batches
    Input Dimensions:
    	[100, 100, 1]
    Performance:
    	Evaluation performance: 0.060352s +- 0.001401s [0.058641s - 0.062648s]
    	Learning performance: 0.012402s +- 0.002049s [0.010733s - 0.016367s]
    
```

### Function Plots
Code from [ActivationLayerTestBase.java:110](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/java/ActivationLayerTestBase.java#L110) executed in 0.00 seconds: 
```java
    return plot("Value Plot", plotData, x -> new double[]{x[0], x[1]});
```

Returns: 

![Result](etc/test.148.png)



Code from [ActivationLayerTestBase.java:114](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/java/ActivationLayerTestBase.java#L114) executed in 0.00 seconds: 
```java
    return plot("Derivative Plot", plotData, x -> new double[]{x[0], x[2]});
```

Returns: 

![Result](etc/test.149.png)



