# AbsActivationLayer
## AbsActivationLayerTest
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
    	[ [ -0.44 ], [ 0.552 ], [ -1.476 ] ],
    	[ [ -0.976 ], [ 1.592 ], [ 1.34 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.021170701477406878, negative=3, min=1.34, max=1.34, mean=0.09866666666666672, count=6.0, positive=3, stdDev=1.1475415267237852, zeros=0}
    Output: [
    	[ [ 0.44 ], [ 0.552 ], [ 1.476 ] ],
    	[ [ 0.976 ], [ 1.592 ], [ 1.34 ] ]
    ]
    Outputs Statistics: {meanExponent=-0.021170701477406878, negative=0, min=1.34, max=1.34, mean=1.0626666666666666, count=6.0, positive=6, stdDev=0.4442141625637597, zeros=0}
    Feedback for input 0
    Inputs Values: [
    	[ [ -0.44 ], [ 0.552 ], [ -1.476 ] ],
    	[ [ -0.976 ], [ 1.592 ], [ 1.34 ] ]
    ]
    Value Statistics: {meanExponent=-0.021170701477406878, negative=3, min=1.34, max=1.34, mean=0.09866666666666672, count=6.0, positive=3, stdDev=1.1475415267237852, zeros=0}
    Implemented Feedback: [ [ -1.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, -1.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 1.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 1.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, -1.0, 0.0 ], [ 0.0, 0.0, 0
```
...[skipping 376 bytes](etc/183.txt)...
```
    0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, -0.9999999999998899, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.9999999999998899 ] ]
    Measured Statistics: {meanExponent=-4.7830642341045674E-14, negative=3, min=0.9999999999998899, max=0.9999999999998899, mean=0.0, count=36.0, positive=3, stdDev=0.40824829046381805, zeros=30}
    Feedback Error: [ [ 1.1013412404281553E-13, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 1.1013412404281553E-13, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, -1.1013412404281553E-13, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, -1.1013412404281553E-13, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 1.1013412404281553E-13, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, -1.1013412404281553E-13 ] ]
    Error Statistics: {meanExponent=-12.958078098036824, negative=3, min=-1.1013412404281553E-13, max=-1.1013412404281553E-13, mean=0.0, count=36.0, positive=3, stdDev=4.496206786221447E-14, zeros=30}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 1.8356e-14 +- 4.1045e-14 [0.0000e+00 - 1.1013e-13] (36#)
    relativeTol: 5.5067e-14 +- 0.0000e+00 [5.5067e-14 - 5.5067e-14] (6#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=1.8356e-14 +- 4.1045e-14 [0.0000e+00 - 1.1013e-13] (36#), relativeTol=5.5067e-14 +- 0.0000e+00 [5.5067e-14 - 5.5067e-14] (6#)}
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
      "class": "com.simiacryptus.mindseye.layers.java.AbsActivationLayer",
      "id": "3102189a-04e1-4860-a360-efb2cd4e2e43",
      "isFrozen": true,
      "name": "AbsActivationLayer/3102189a-04e1-4860-a360-efb2cd4e2e43"
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
    	[ [ 0.252 ], [ -0.188 ], [ -0.884 ] ],
    	[ [ -1.788 ], [ 0.328 ], [ 1.808 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 0.252 ], [ 0.188 ], [ 0.884 ] ],
    	[ [ 1.788 ], [ 0.328 ], [ 1.808 ] ]
    ]
    --------------------
    Derivative: 
    [
    	[ [ 1.0 ], [ -1.0 ], [ -1.0 ] ],
    	[ [ -1.0 ], [ 1.0 ], [ 1.0 ] ]
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
    	[ [ 0.512 ], [ -0.244 ], [ 1.976 ], [ 1.608 ], [ -0.66 ], [ -0.872 ], [ 0.06 ], [ 0.808 ], ... ],
    	[ [ -1.312 ], [ 1.296 ], [ -1.232 ], [ -0.94 ], [ 1.692 ], [ 0.368 ], [ 0.004 ], [ 1.056 ], ... ],
    	[ [ 1.688 ], [ 0.58 ], [ 0.732 ], [ -1.752 ], [ 0.62 ], [ 0.68 ], [ -1.128 ], [ -0.052 ], ... ],
    	[ [ -0.692 ], [ -1.74 ], [ 0.844 ], [ -1.916 ], [ -1.152 ], [ -0.672 ], [ 1.288 ], [ -0.444 ], ... ],
    	[ [ -0.304 ], [ -1.996 ], [ -1.628 ], [ -0.468 ], [ -1.216 ], [ 0.26 ], [ -0.796 ], [ 0.512 ], ... ],
    	[ [ -0.34 ], [ -0.628 ], [ -1.992 ], [ 1.272 ], [ -1.22 ], [ 1.384 ], [ 0.572 ], [ 0.568 ], ... ],
    	[ [ 1.868 ], [ -0.916 ], [ 1.776 ], [ 1.468 ], [ -0.08 ], [ -0.224 ], [ -0.332 ], [ 1.248 ], ... ],
    	[ [ 0.188 ], [ 0.376 ], [ 1.14 ], [ 0.98 ], [ -0.344 ], [ 1.984 ], [ 0.084 ], [ -1.072 ], ... ],
    	...
    ]
```



First, we use a conjugate gradient descent method, which converges the fastest for purely linear functions.

Code from [LearningTester.java:225](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L225) executed in 0.08 seconds: 
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
    F(0.0) = LineSearchPoint{point=PointSample{avg=0.6444937215999977}, derivative=-2.5779748864000003E-4}
    New Minimum: 0.6444937215999977 > 0.6444937215999716
    F(1.0E-10) = LineSearchPoint{point=PointSample{avg=0.6444937215999716}, derivative=-2.577974886399949E-4}, delta = -2.609024107869118E-14
    New Minimum: 0.6444937215999716 > 0.6444937215998212
    F(7.000000000000001E-10) = LineSearchPoint{point=PointSample{avg=0.6444937215998212}, derivative=-2.577974886399639E-4}, delta = -1.765254609153999E-13
    New Minimum: 0.6444937215998212 > 0.6444937215987404
    F(4.900000000000001E-9) = LineSearchPoint{point=PointSample{avg=0.6444937215987404}, derivative=-2.5779748863974736E-4}, delta = -1.2573275753879898E-12
    New Minimum: 0.6444937215987404 > 0.6444937215911526
    F(3.430000000000001E-8) = LineSearchPoint{point=PointSample{avg=0.6444937215911526}, derivative=-2.5779748863823153E-4}, delta = -8.845146837188622E-12
    New Minimum: 0.6444937215911526 > 0.6444937215380999
    F(2.401000
```
...[skipping 6204 bytes](etc/184.txt)...
```
    332.464428904737) = LineSearchPoint{point=PointSample{avg=2.4612754898199816E-44}, derivative=1.4806248583815985E-46}, delta = -5.5422416573893464E-42
    2.4612754898199816E-44 <= 5.566854412287546E-42
    New Minimum: 2.4612754898199816E-44 > 2.1733629789184108E-74
    F(5000.0) = LineSearchPoint{point=PointSample{avg=2.1733629789184108E-74}, derivative=7.316281428386506E-62}, delta = -5.566854412287546E-42
    Right bracket at 5000.0
    Converged to right
    Iteration 4 complete. Error: 2.1733629789184108E-74 Total: 239633489749075.5000; Orientation: 0.0003; Line Search: 0.0041
    Zero gradient: 2.9484660275596946E-39
    F(0.0) = LineSearchPoint{point=PointSample{avg=2.1733629789184108E-74}, derivative=-8.693451915673645E-78}
    New Minimum: 2.1733629789184108E-74 > 0.0
    F(5000.0) = LineSearchPoint{point=PointSample{avg=0.0}, derivative=0.0}, delta = -2.1733629789184108E-74
    0.0 <= 2.1733629789184108E-74
    Converged to right
    Iteration 5 complete. Error: 0.0 Total: 239633493373430.5000; Orientation: 0.0003; Line Search: 0.0022
    
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
    th(0)=0.6444937215999977;dx=-2.5779748864000003E-4
    New Minimum: 0.6444937215999977 > 0.6439384334064687
    WOLFE (weak): th(2.154434690031884)=0.6439384334064687; dx=-2.576864070694942E-4 delta=5.552881935290133E-4
    New Minimum: 0.6439384334064687 > 0.6433833845309116
    WOLFE (weak): th(4.308869380063768)=0.6433833845309116; dx=-2.575753254989884E-4 delta=0.0011103370690861425
    New Minimum: 0.6433833845309116 > 0.6411655822086273
    WOLFE (weak): th(12.926608140191302)=0.6411655822086273; dx=-2.571309992169651E-4 delta=0.003328139391370444
    New Minimum: 0.6411655822086273 > 0.6312328567201083
    WOLFE (weak): th(51.70643256076521)=0.6312328567201083; dx=-2.5513153094786024E-4 delta=0.013260864879889422
    New Minimum: 0.6312328567201083 > 0.579567868816738
    WOLFE (weak): th(258.53216280382605)=0.579567868816738; dx=-2.444677001793012E-4 delta=0.06492585278325969
    New Minimum: 0.579567868816738 > 0.30663129050734983
    END: th(1551.1929768229
```
...[skipping 2564 bytes](etc/185.txt)...
```
    =-1.4411726110573716E-10 delta=3.5669022123670567E-6
    Iteration 6 complete. Error: 3.6029315276433255E-8 Total: 239633544673815.4700; Orientation: 0.0009; Line Search: 0.0048
    LBFGS Accumulation History: 1 points
    th(0)=3.6029315276433255E-8;dx=-1.4411726110573344E-11
    New Minimum: 3.6029315276433255E-8 > 3.176720988561907E-8
    WOLF (strong): th(9694.956105143481)=3.176720988561907E-8; dx=1.3532484297698427E-11 delta=4.262105390814185E-9
    New Minimum: 3.176720988561907E-8 > 3.352591839001095E-11
    END: th(4847.478052571741)=3.352591839001095E-11; dx=-4.396209064374743E-13 delta=3.5995789358043245E-8
    Iteration 7 complete. Error: 3.352591839001095E-11 Total: 239633550580277.4700; Orientation: 0.0007; Line Search: 0.0042
    LBFGS Accumulation History: 1 points
    th(0)=3.352591839001095E-11;dx=-1.3410367356004227E-14
    MAX ALPHA: th(0)=3.352591839001095E-11;th'(0)=-1.3410367356004227E-14;
    Iteration 8 failed, aborting. Error: 3.352591839001095E-11 Total: 239633554428340.4400; Orientation: 0.0007; Line Search: 0.0023
    
```

Returns: 

```
    3.352591839001095E-11
```



Training Converged

Code from [LearningTester.java:95](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L95) executed in 0.00 seconds: 
```java
    return TestUtil.compare(runs);
```

Returns: 

![Result](etc/test.108.png)



Code from [LearningTester.java:98](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L98) executed in 0.01 seconds: 
```java
    return TestUtil.compareTime(runs);
```

Returns: 

![Result](etc/test.109.png)



### Performance
Now we execute larger-scale runs to benchmark performance:

Code from [PerformanceTester.java:66](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/PerformanceTester.java#L66) executed in 0.24 seconds: 
```java
    test(component, inputPrototype);
```
Logging: 
```
    100 batches
    Input Dimensions:
    	[100, 100, 1]
    Performance:
    	Evaluation performance: 0.010878s +- 0.009250s [0.005443s - 0.029353s]
    	Learning performance: 0.030741s +- 0.026378s [0.013675s - 0.082599s]
    
```

### Function Plots
Code from [ActivationLayerTestBase.java:110](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/java/ActivationLayerTestBase.java#L110) executed in 0.00 seconds: 
```java
    return plot("Value Plot", plotData, x -> new double[]{x[0], x[1]});
```

Returns: 

![Result](etc/test.110.png)



Code from [ActivationLayerTestBase.java:114](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/java/ActivationLayerTestBase.java#L114) executed in 0.00 seconds: 
```java
    return plot("Derivative Plot", plotData, x -> new double[]{x[0], x[2]});
```

Returns: 

![Result](etc/test.111.png)



