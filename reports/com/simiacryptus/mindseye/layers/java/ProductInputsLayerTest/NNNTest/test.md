# ProductInputsLayer
## NNNTest
### Batch Execution
Code from [BatchingTester.java:66](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/BatchingTester.java#L66) executed in 0.00 seconds: 
```java
    return test(reference, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (120#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (120#)}
```



Code from [SingleDerivativeTester.java:77](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/SingleDerivativeTester.java#L77) executed in 0.00 seconds: 
```java
    return test(component, inputPrototype);
```
Logging: 
```
    Inputs: [ -1.132, -1.032, 0.46 ],
    [ -0.872, -1.108, 1.348 ],
    [ -0.716, 1.364, -1.268 ]
    Inputs Statistics: {meanExponent=-0.08990534805832691, negative=2, min=0.46, max=0.46, mean=-0.568, count=3.0, positive=1, stdDev=0.7280512802451943, zeros=0},
    {meanExponent=0.038248712508093086, negative=2, min=1.348, max=1.348, mean=-0.21066666666666664, count=3.0, positive=1, stdDev=1.1063469417662597, zeros=0},
    {meanExponent=0.030948882058009858, negative=2, min=-1.268, max=-1.268, mean=-0.20666666666666664, count=3.0, positive=1, stdDev=1.1332611741734069, zeros=0}
    Output: [ -0.706766464, 1.5596739840000002, -0.7862614400000001 ]
    Outputs Statistics: {meanExponent=-0.02070775349222397, negative=2, min=-0.7862614400000001, max=-0.7862614400000001, mean=0.022215360000000055, count=3.0, positive=1, stdDev=1.0876317171378342, zeros=0}
    Feedback for input 0
    Inputs Values: [ -1.132, -1.032, 0.46 ]
    Value Statistics: {meanExponent=-0.08990534805832691, negative=2, min=0.46, max=0.46, mean=-0.568, count=3.0, positive=1, std
```
...[skipping 2661 bytes](etc/314.txt)...
```
    .05165663555023386, negative=0, min=0.6200800000000001, max=0.6200800000000001, mean=0.3056266666666667, count=9.0, positive=3, stdDev=0.45039442007990177, zeros=6}
    Measured Feedback: [ [ 0.9871040000009046, 0.0, 0.0 ], [ 0.0, 1.1434560000012084, 0.0 ], [ 0.0, 0.0, 0.6200799999989126 ] ]
    Measured Statistics: {meanExponent=-0.051656635550202064, negative=0, min=0.6200799999989126, max=0.6200799999989126, mean=0.3056266666667806, count=9.0, positive=3, stdDev=0.45039442008021935, zeros=6}
    Feedback Error: [ [ 9.047207427670401E-13, 0.0, 0.0 ], [ 0.0, 1.2083667400020204E-12, 0.0 ], [ 0.0, 0.0, -1.0874634526203408E-12 ] ]
    Error Statistics: {meanExponent=-11.97495733977683, negative=1, min=-1.0874634526203408E-12, max=-1.0874634526203408E-12, mean=1.1395822557207995E-13, count=9.0, positive=2, stdDev=6.095871341363553E-13, zeros=6}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 3.0100e-13 +- 5.2044e-13 [0.0000e+00 - 1.9595e-12] (27#)
    relativeTol: 4.8419e-13 +- 2.5716e-13 [4.1138e-14 - 8.7687e-13] (9#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=3.0100e-13 +- 5.2044e-13 [0.0000e+00 - 1.9595e-12] (27#), relativeTol=4.8419e-13 +- 2.5716e-13 [4.1138e-14 - 8.7687e-13] (9#)}
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
      "class": "com.simiacryptus.mindseye.layers.java.ProductInputsLayer",
      "id": "d1687387-9ad7-43b4-8c7a-c879e292e9f1",
      "isFrozen": false,
      "name": "ProductInputsLayer/d1687387-9ad7-43b4-8c7a-c879e292e9f1"
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
    [[ -1.24, -1.444, -0.416 ],
    [ 1.88, 0.012, 1.528 ],
    [ -0.684, -1.1, -0.868 ]]
    --------------------
    Output: 
    [ 1.5945407999999999, 0.019060800000000003, 0.5517424639999999 ]
    --------------------
    Derivative: 
    [ -1.28592, -0.013200000000000002, -1.326304 ],
    [ 0.84816, 1.5884, 0.36108799999999996 ],
    [ -2.3312, -0.017328, -0.635648 ]
```



### Input Learning
In this test, we use a network to learn this target input, given it's pre-evaluated output:

Code from [LearningTester.java:127](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L127) executed in 0.00 seconds: 
```java
    return Arrays.stream(input_target).map(x -> x.prettyPrint()).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [ -1.432, -1.0, -1.856 ]
    [ 0.568, 1.496, -0.116 ]
    [ -0.324, -1.268, 0.904 ]
```



First, we use a conjugate gradient descent method, which converges the fastest for purely linear functions.

Code from [LearningTester.java:225](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L225) executed in 0.09 seconds: 
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
    F(0.0) = LineSearchPoint{point=PointSample{avg=20.678290029217365}, derivative=-65916.4427144095}
    New Minimum: 20.678290029217365 > 20.678288824848497
    F(1.0E-10) = LineSearchPoint{point=PointSample{avg=20.678288824848497}, derivative=-65916.43497021617}, delta = -1.2043688677465525E-6
    New Minimum: 20.678288824848497 > 20.678281598636925
    F(7.000000000000001E-10) = LineSearchPoint{point=PointSample{avg=20.678281598636925}, derivative=-65916.38850507335}, delta = -8.430580439977575E-6
    New Minimum: 20.678281598636925 > 20.678231015234573
    F(4.900000000000001E-9) = LineSearchPoint{point=PointSample{avg=20.678231015234573}, derivative=-65916.06324990839}, delta = -5.90139827920666E-5
    New Minimum: 20.678231015234573 > 20.677876935272753
    F(3.430000000000001E-8) = LineSearchPoint{point=PointSample{avg=20.677876935272753}, derivative=-65913.78650464973}, delta = -4.1309394461208626E-4
    New Minimum: 20.677876935272753 > 20.67539856440844
    F(2.4010000000000004E-7) = LineSea
```
...[skipping 116002 bytes](etc/315.txt)...
```
     0.0
    Left bracket at 2.0937281861520276E30
    F(2.3743829836020546E30) = LineSearchPoint{point=PointSample{avg=2.28543686733952E-32}, derivative=-9.644000422344302E-95}, delta = 0.0
    Left bracket at 2.3743829836020546E30
    F(2.6506405425043616E30) = LineSearchPoint{point=PointSample{avg=2.28543686733952E-32}, derivative=-9.644000422344302E-95}, delta = 0.0
    Left bracket at 2.6506405425043616E30
    F(2.922569757839621E30) = LineSearchPoint{point=PointSample{avg=2.28543686733952E-32}, derivative=-9.644000422344302E-95}, delta = 0.0
    Left bracket at 2.922569757839621E30
    F(3.190238445156881E30) = LineSearchPoint{point=PointSample{avg=2.28543686733952E-32}, derivative=-9.644000422344302E-95}, delta = 0.0
    Left bracket at 3.190238445156881E30
    F(3.4537133574858535E30) = LineSearchPoint{point=PointSample{avg=2.28543686733952E-32}, derivative=-9.644000422344302E-95}, delta = 0.0
    Loops = 12
    Iteration 54 failed, aborting. Error: 2.28543686733952E-32 Total: 239706258095768.7500; Orientation: 0.0000; Line Search: 0.0045
    
```

Returns: 

```
    2.28543686733952E-32
```



Training Converged

Next, we run the same optimization using L-BFGS, which is nearly ideal for purely second-order or quadratic functions.

Code from [LearningTester.java:249](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L249) executed in 0.01 seconds: 
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
    th(0)=20.678290029217365;dx=-65916.4427144095
    Armijo: th(2.154434690031884)=8.805430128344409E15; dx=9.700035668353932E26 delta=-8.805430128344388E15
    Armijo: th(1.077217345015942)=1.348010598009383E14; dx=1.8373318418816432E24 delta=-1.3480105980091762E14
    Armijo: th(0.3590724483386473)=1.7027502345634152E11; dx=8.248493062729486E19 delta=-1.7027502343566324E11
    Armijo: th(0.08976811208466183)=2.82632822970269E7; dx=1.7639361910551216E14 delta=-2.826326161873687E7
    Armijo: th(0.017953622416932366)=130.51529600067897; dx=1713116.779208492 delta=-109.83700597146161
    New Minimum: 20.678290029217365 > 5.639178189886898
    END: th(0.002992270402822061)=5.639178189886898; dx=-1668.4304390221155 delta=15.039111839330467
    Iteration 1 complete. Error: 5.639178189886898 Total: 239706266084852.7500; Orientation: 0.0001; Line Search: 0.0014
    LBFGS Accumulation History: 1 points
    th(0)=5.639178189886898;dx=-629.2135296751978
    New Minimum: 5.639
```
...[skipping 6620 bytes](etc/316.txt)...
```
     1 points
    th(0)=7.773783605623918E-5;dx=-1.878436931655977E-12
    New Minimum: 7.773783605623918E-5 > 3.565428414327586E-5
    END: th(3223.3255788977654)=3.565428414327586E-5; dx=-5.817905139819706E-13 delta=4.208355191296332E-5
    Iteration 20 complete. Error: 3.565428414327586E-5 Total: 239706274665551.7200; Orientation: 0.0000; Line Search: 0.0002
    LBFGS Accumulation History: 1 points
    th(0)=3.565428414327586E-5;dx=-1.8019683803988835E-13
    New Minimum: 3.565428414327586E-5 > 1.6415932481117212E-5
    END: th(6944.44444444445)=1.6415932481117212E-5; dx=-5.61928779588966E-14 delta=1.9238351662158646E-5
    Iteration 21 complete. Error: 1.6415932481117212E-5 Total: 239706275089599.7200; Orientation: 0.0000; Line Search: 0.0003
    LBFGS Accumulation History: 1 points
    th(0)=1.6415932481117212E-5;dx=-1.7523424033351896E-14
    MAX ALPHA: th(0)=1.6415932481117212E-5;th'(0)=-1.7523424033351896E-14;
    Iteration 22 failed, aborting. Error: 1.6415932481117212E-5 Total: 239706275466341.7200; Orientation: 0.0000; Line Search: 0.0002
    
```

Returns: 

```
    1.6415932481117212E-5
```



This training run resulted in the following regressed input:

Code from [LearningTester.java:154](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L154) executed in 0.00 seconds: 
```java
    return Arrays.stream(input_lbgfs).map(x -> x.prettyPrint()).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [ 0.6375077235159358, 1.2373973409020225, 0.5745837964554902 ]
    [ 1.496, 0.568, -0.116 ]
    [ 0.904, -1.268, -0.324 ]
```



Code from [LearningTester.java:95](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L95) executed in 0.00 seconds: 
```java
    return TestUtil.compare(runs);
```

Returns: 

![Result](etc/test.223.png)



Code from [LearningTester.java:98](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L98) executed in 0.00 seconds: 
```java
    return TestUtil.compareTime(runs);
```

Returns: 

![Result](etc/test.224.png)



### Performance
Now we execute larger-scale runs to benchmark performance:

Code from [PerformanceTester.java:66](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/PerformanceTester.java#L66) executed in 0.01 seconds: 
```java
    test(component, inputPrototype);
```
Logging: 
```
    100 batches
    Input Dimensions:
    	[3]
    	[3]
    	[3]
    Performance:
    	Evaluation performance: 0.000296s +- 0.000078s [0.000195s - 0.000436s]
    	Learning performance: 0.000133s +- 0.000049s [0.000085s - 0.000217s]
    
```

