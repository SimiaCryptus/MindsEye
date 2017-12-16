# ProductInputsLayer
## N1Test
### Batch Execution
Code from [BatchingTester.java:66](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/BatchingTester.java#L66) executed in 0.00 seconds: 
```java
    return test(reference, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (70#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (70#)}
```



Code from [SingleDerivativeTester.java:77](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/SingleDerivativeTester.java#L77) executed in 0.00 seconds: 
```java
    return test(component, inputPrototype);
```
Logging: 
```
    Inputs: [ 0.568, -0.472, -1.58 ],
    [ -1.992 ]
    Inputs Statistics: {meanExponent=-0.12435085956682361, negative=2, min=-1.58, max=-1.58, mean=-0.49466666666666664, count=3.0, positive=1, stdDev=0.8770637883808048, zeros=0},
    {meanExponent=0.29928933408767994, negative=1, min=-1.992, max=-1.992, mean=-1.992, count=1.0, positive=0, stdDev=0.0, zeros=0}
    Output: [ -1.1314559999999998, 0.940224, 3.14736 ]
    Outputs Statistics: {meanExponent=0.1749384745208563, negative=1, min=3.14736, max=3.14736, mean=0.985376, count=3.0, positive=2, stdDev=1.7471110664545628, zeros=0}
    Feedback for input 0
    Inputs Values: [ 0.568, -0.472, -1.58 ]
    Value Statistics: {meanExponent=-0.12435085956682361, negative=2, min=-1.58, max=-1.58, mean=-0.49466666666666664, count=3.0, positive=1, stdDev=0.8770637883808048, zeros=0}
    Implemented Feedback: [ [ -1.992, 0.0, 0.0 ], [ 0.0, -1.992, 0.0 ], [ 0.0, 0.0, -1.992 ] ]
    Implemented Statistics: {meanExponent=0.29928933408767994, negative=3, min=-1.992, max=-1.992, mean=-0.664, count=9.0, posit
```
...[skipping 910 bytes](etc/312.txt)...
```
    =0}
    Implemented Feedback: [ [ 0.568, -0.472, -1.58 ] ]
    Implemented Statistics: {meanExponent=-0.12435085956682361, negative=2, min=-1.58, max=-1.58, mean=-0.49466666666666664, count=3.0, positive=1, stdDev=0.8770637883808048, zeros=0}
    Measured Feedback: [ [ 0.5679999999985696, -0.47200000000025, -1.5799999999988046 ] ]
    Measured Statistics: {meanExponent=-0.12435085956722099, negative=2, min=-1.5799999999988046, max=-1.5799999999988046, mean=-0.4946666666668283, count=3.0, positive=1, stdDev=0.8770637883797318, zeros=0}
    Feedback Error: [ [ -1.4303003226245892E-12, -2.5002222514558525E-13, 1.1954881529163686E-12 ] ]
    Error Statistics: {meanExponent=-12.123016290240562, negative=2, min=1.1954881529163686E-12, max=1.1954881529163686E-12, mean=-1.6161146495126863E-13, count=3.0, positive=1, stdDev=1.0737950227920688E-12, zeros=0}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 4.6221e-13 +- 6.1324e-13 [0.0000e+00 - 1.7701e-12] (12#)
    relativeTol: 4.2877e-13 +- 3.9124e-13 [1.1303e-13 - 1.2591e-12] (6#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=4.6221e-13 +- 6.1324e-13 [0.0000e+00 - 1.7701e-12] (12#), relativeTol=4.2877e-13 +- 3.9124e-13 [1.1303e-13 - 1.2591e-12] (6#)}
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
      "id": "772ac481-4da7-46fd-bf37-e0ab61827e5f",
      "isFrozen": false,
      "name": "ProductInputsLayer/772ac481-4da7-46fd-bf37-e0ab61827e5f"
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
    [[ 1.424, 1.592, 1.5 ],
    [ -0.996 ]]
    --------------------
    Output: 
    [ -1.418304, -1.5856320000000002, -1.494 ]
    --------------------
    Derivative: 
    [ -0.996, -0.996, -0.996 ],
    [ 4.516 ]
```



### Input Learning
In this test, we use a network to learn this target input, given it's pre-evaluated output:

Code from [LearningTester.java:127](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L127) executed in 0.00 seconds: 
```java
    return Arrays.stream(input_target).map(x -> x.prettyPrint()).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [ -0.844, 0.176, 0.556 ]
    [ 0.276 ]
```



First, we use a conjugate gradient descent method, which converges the fastest for purely linear functions.

Code from [LearningTester.java:225](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L225) executed in 0.00 seconds: 
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
    F(0.0) = LineSearchPoint{point=PointSample{avg=0.3060334336}, derivative=-0.6313297705191696}
    F(1.0E-10) = LineSearchPoint{point=PointSample{avg=0.30603343368424235}, derivative=-0.6313297706981116}, delta = 8.42423353297761E-11
    F(7.692307692307693E-12) = LineSearchPoint{point=PointSample{avg=0.3060334336064802}, derivative=-0.6313297705329343}, delta = 6.480205261283345E-12
    F(5.91715976331361E-13) = LineSearchPoint{point=PointSample{avg=0.30603343360049845}, derivative=-0.6313297705202282}, delta = 4.98434626905464E-13
    0.0 ~= 5.91715976331361E-13
    Converged to right
    Iteration 1 failed, aborting. Error: 0.3060334336 Total: 239706134983679.8800; Orientation: 0.0001; Line Search: 0.0005
    
```

Returns: 

```
    0.3060334336
```



This training run resulted in the following regressed input:

Code from [LearningTester.java:144](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L144) executed in 0.00 seconds: 
```java
    return Arrays.stream(input_gd).map(x -> x.prettyPrint()).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [ -0.8440000000004699, 0.17599999999999982, 0.5559999999999873 ]
    [ 0.276 ]
```



Next, we run the same optimization using L-BFGS, which is nearly ideal for purely second-order or quadratic functions.

Code from [LearningTester.java:249](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L249) executed in 0.00 seconds: 
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
    th(0)=0.3060334336;dx=-0.6313297705191696
    Armijo: th(2.154434690031884)=15.245463139818861; dx=-32.282749246440616 delta=-14.939429706218862
    Armijo: th(1.077217345015942)=3.253883141060332; dx=-6.879999022383068 delta=-2.947849707460332
    Armijo: th(0.3590724483386473)=0.7653798028151167; dx=-1.6060263269461574 delta=-0.45934636921511673
    Armijo: th(0.08976811208466183)=0.39010679295545736; dx=-0.8098574671385456 delta=-0.08407335935545734
    Armijo: th(0.017953622416932366)=0.3214826353724868; dx=-0.6641438037466347 delta=-0.015449201772486765
    Armijo: th(0.002992270402822061)=0.3085631360053257; dx=-0.6367031371374302 delta=-0.0025297024053256845
    Armijo: th(4.2746720040315154E-4)=0.3063937244764929; dx=-0.632095075208721 delta=-3.602908764928636E-4
    Armijo: th(5.343340005039394E-5)=0.30607845002846334; dx=-0.6314253914000715 delta=-4.5016428463329206E-5
    Armijo: th(5.9370444500437714E-6)=0.30603843514423706; dx=-0.631340394466145
```
...[skipping 46 bytes](etc/313.txt)...
```
    044450043771E-7)=0.3060339337512609; dx=-0.6313308329071696 delta=-5.00151260907078E-7
    Armijo: th(5.397313136403428E-8)=0.3060334790682674; dx=-0.6313298670998352 delta=-4.5468267373394156E-8
    Armijo: th(4.4977609470028565E-9)=0.30603343738902206; dx=-0.6313297785675578 delta=-3.789022040567858E-9
    Armijo: th(3.4598161130791205E-10)=0.3060334338914632; dx=-0.6313297711382762 delta=-2.914631869188611E-10
    Armijo: th(2.4712972236279432E-11)=0.30603343362081875; dx=-0.6313297705633912 delta=-2.0818735624317242E-11
    Armijo: th(1.6475314824186289E-12)=0.30603343360138796; dx=-0.6313297705221177 delta=-1.3879453142351394E-12
    Armijo: th(1.029707176511643E-13)=0.3060334336000868; dx=-0.6313297705193539 delta=-8.676392937445598E-14
    Armijo: th(6.057101038303783E-15)=0.30603343360000507; dx=-0.6313297705191803 delta=-5.051514762044462E-15
    MIN ALPHA: th(0)=0.3060334336;th'(0)=-0.6313297705191696;
    Iteration 1 failed, aborting. Error: 0.3060334336 Total: 239706139066279.8800; Orientation: 0.0001; Line Search: 0.0025
    
```

Returns: 

```
    0.3060334336
```



This training run resulted in the following regressed input:

Code from [LearningTester.java:154](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L154) executed in 0.00 seconds: 
```java
    return Arrays.stream(input_lbgfs).map(x -> x.prettyPrint()).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [ -0.844, 0.176, 0.556 ]
    [ 0.276 ]
```



Code from [LearningTester.java:95](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L95) executed in 0.00 seconds: 
```java
    return TestUtil.compare(runs);
```

Code from [LearningTester.java:98](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L98) executed in 0.00 seconds: 
```java
    return TestUtil.compareTime(runs);
```

### Performance
Now we execute larger-scale runs to benchmark performance:

Code from [PerformanceTester.java:66](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/PerformanceTester.java#L66) executed in 0.00 seconds: 
```java
    test(component, inputPrototype);
```
Logging: 
```
    100 batches
    Input Dimensions:
    	[3]
    	[1]
    Performance:
    	Evaluation performance: 0.000147s +- 0.000017s [0.000130s - 0.000172s]
    	Learning performance: 0.000062s +- 0.000006s [0.000055s - 0.000072s]
    
```

