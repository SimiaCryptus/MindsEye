# NthPowerActivationLayer
## InvSqrtPowerTest
### Batch Execution
Code from [BatchingTester.java:66](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/BatchingTester.java#L66) executed in 0.00 seconds: 
```java
    return test(reference, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (120#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (70#)}
```



Code from [SingleDerivativeTester.java:77](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/SingleDerivativeTester.java#L77) executed in 0.00 seconds: 
```java
    return test(component, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ -0.436 ], [ -1.328 ], [ 0.312 ] ],
    	[ [ 1.02 ], [ 1.04 ], [ 0.984 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.12075537203148884, negative=2, min=0.984, max=0.984, mean=0.26533333333333337, count=6.0, positive=4, stdDev=0.8868360740419969, zeros=0}
    Output: [
    	[ [ 0.0 ], [ 0.0 ], [ 1.7902871850985822 ] ],
    	[ [ 0.9901475429766743 ], [ 0.9805806756909201 ], [ 1.0080972981818899 ] ]
    ]
    Outputs Statistics: {meanExponent=0.06090209956118973, negative=0, min=1.0080972981818899, max=1.0080972981818899, mean=0.7948521169913444, count=6.0, positive=4, stdDev=0.6288322058675254, zeros=2}
    Feedback for input 0
    Inputs Values: [
    	[ [ -0.436 ], [ -1.328 ], [ 0.312 ] ],
    	[ [ 1.02 ], [ 1.04 ], [ 0.984 ] ]
    ]
    Value Statistics: {meanExponent=-0.12075537203148884, negative=2, min=0.984, max=0.984, mean=0.26533333333333337, count=6.0, positive=4, stdDev=0.8868360740419969, zeros=0}
    Implemented Feedback: [ [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, -0.4853664426356247, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0,
```
...[skipping 525 bytes](etc/300.txt)...
```
    -0.4713990223093045, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, -2.868360484800103, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, -0.5122055223605315 ] ]
    Measured Statistics: {meanExponent=-0.1183738800971975, negative=4, min=-0.5122055223605315, max=-0.5122055223605315, mean=-0.12048043850866454, count=36.0, positive=0, stdDev=0.48376561276538455, zeros=32}
    Feedback Error: [ [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 3.568579364066071E-5, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 3.399484979166312E-5, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 6.894913194197549E-4, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 3.903972376206877E-5 ] ]
    Error Statistics: {meanExponent=-4.1215139937006695, negative=0, min=3.903972376206877E-5, max=3.903972376206877E-5, mean=2.217254685039299E-5, count=36.0, positive=4, stdDev=1.1324176683940321E-4, zeros=32}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 2.2173e-05 +- 1.1324e-04 [0.0000e+00 - 6.8949e-04] (36#)
    relativeTol: 5.7775e-05 +- 3.6034e-05 [3.6056e-05 - 1.2017e-04] (4#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=2.2173e-05 +- 1.1324e-04 [0.0000e+00 - 6.8949e-04] (36#), relativeTol=5.7775e-05 +- 3.6034e-05 [3.6056e-05 - 1.2017e-04] (4#)}
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
      "class": "com.simiacryptus.mindseye.layers.java.NthPowerActivationLayer",
      "id": "0f9a7df7-8ba9-46cd-88cd-ee2eefecbfc3",
      "isFrozen": false,
      "name": "NthPowerActivationLayer/0f9a7df7-8ba9-46cd-88cd-ee2eefecbfc3",
      "power": -0.5
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
    	[ [ -0.768 ], [ 1.156 ], [ -1.408 ] ],
    	[ [ -1.008 ], [ -1.888 ], [ -1.548 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 0.0 ], [ 0.9300816647554058 ], [ 0.0 ] ],
    	[ [ 0.0 ], [ 0.0 ], [ 0.0 ] ]
    ]
    --------------------
    Derivative: 
    [
    	[ [ 0.0 ], [ -0.40228445707413746 ], [ 0.0 ] ],
    	[ [ 0.0 ], [ 0.0 ], [ 0.0 ] ]
    ]
```



### Input Learning
In this test, we use a network to learn this target input, given it's pre-evaluated output:

Code from [LearningTester.java:127](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L127) executed in 0.00 seconds: 
```java
    return Arrays.stream(input_target).map(x -> x.prettyPrint()).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [
    	[ [ 0.332 ], [ 1.756 ], [ -1.36 ], [ 0.592 ], [ -1.768 ], [ -0.756 ], [ 0.64 ], [ -0.76 ], ... ],
    	[ [ 0.512 ], [ -1.808 ], [ -0.892 ], [ 0.728 ], [ 1.408 ], [ 1.512 ], [ 1.556 ], [ -1.868 ], ... ],
    	[ [ -0.404 ], [ -0.292 ], [ -1.152 ], [ -0.224 ], [ 1.004 ], [ -1.244 ], [ 1.652 ], [ 1.18 ], ... ],
    	[ [ 0.732 ], [ -0.392 ], [ 1.416 ], [ 0.968 ], [ -0.884 ], [ -1.376 ], [ 1.084 ], [ -1.94 ], ... ],
    	[ [ 1.28 ], [ -1.86 ], [ 0.656 ], [ -1.4 ], [ 1.152 ], [ -0.452 ], [ -0.248 ], [ 1.9 ], ... ],
    	[ [ 0.404 ], [ -0.56 ], [ -1.06 ], [ -0.548 ], [ 0.3 ], [ 0.856 ], [ 1.856 ], [ 0.44 ], ... ],
    	[ [ -0.98 ], [ 1.412 ], [ -0.576 ], [ 1.556 ], [ 1.48 ], [ -1.644 ], [ 1.08 ], [ -1.636 ], ... ],
    	[ [ -1.724 ], [ -1.42 ], [ -1.028 ], [ -1.42 ], [ -1.36 ], [ -0.548 ], [ -1.692 ], [ 0.5 ], ... ],
    	...
    ]
```



First, we use a conjugate gradient descent method, which converges the fastest for purely linear functions.

Code from [LearningTester.java:225](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L225) executed in 6.19 seconds: 
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
    F(0.0) = LineSearchPoint{point=PointSample{avg=0.6933974985881487}, derivative=-6.690032321176461E-4}
    New Minimum: 0.6933974985881487 > 0.6933974985880801
    F(1.0E-10) = LineSearchPoint{point=PointSample{avg=0.6933974985880801}, derivative=-6.690032321168415E-4}, delta = -6.861178292183467E-14
    New Minimum: 0.6933974985880801 > 0.6933974985876804
    F(7.000000000000001E-10) = LineSearchPoint{point=PointSample{avg=0.6933974985876804}, derivative=-6.690032321120139E-4}, delta = -4.68292071786891E-13
    New Minimum: 0.6933974985876804 > 0.6933974985848703
    F(4.900000000000001E-9) = LineSearchPoint{point=PointSample{avg=0.6933974985848703}, derivative=-6.690032320782205E-4}, delta = -3.2783775694156247E-12
    New Minimum: 0.6933974985848703 > 0.6933974985652019
    F(3.430000000000001E-8) = LineSearchPoint{point=PointSample{avg=0.6933974985652019}, derivative=-6.690032318416669E-4}, delta = -2.294675560676751E-11
    New Minimum: 0.6933974985652019 > 0.6933974984275207
    F(2.4010000000
```
...[skipping 430815 bytes](etc/301.txt)...
```
    e{avg=0.4280874960629643}, derivative=-1.2525372059162857E-9}, delta = -1.5570956202193287E-7
    F(742.399257537106) = LineSearchPoint{point=PointSample{avg=0.4280876779121543}, derivative=2.064473942135178E-9}, delta = 2.6139627962162848E-8
    F(57.107635195162) = LineSearchPoint{point=PointSample{avg=0.42808756230774014}, derivative=-1.4530243798535695E-9}, delta = -8.946478619042963E-8
    New Minimum: 0.4280874960629643 > 0.42808732028131397
    F(399.75344636613397) = LineSearchPoint{point=PointSample{avg=0.42808732028131397}, derivative=1.0043312453448733E-10}, delta = -3.3149121236863976E-7
    0.42808732028131397 <= 0.42808765177252633
    New Minimum: 0.42808732028131397 > 0.42808731930233196
    F(377.1875097860353) = LineSearchPoint{point=PointSample{avg=0.42808731930233196}, derivative=-1.3374345816014728E-11}, delta = -3.3247019437521175E-7
    Left bracket at 377.1875097860353
    Converged to left
    Iteration 250 complete. Error: 0.42808731930233196 Total: 239692332185319.7000; Orientation: 0.0003; Line Search: 0.0124
    
```

Returns: 

```
    0.42808731930233196
```



This training run resulted in the following regressed input:

Code from [LearningTester.java:144](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L144) executed in 0.01 seconds: 
```java
    return Arrays.stream(input_gd).map(x -> x.prettyPrint()).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [
    	[ [ -1.672 ], [ -1.044 ], [ 25.610814423018894 ], [ -1.564 ], [ 25.89287226032564 ], [ 26.396741910184364 ], [ -0.2 ], [ 25.78119484209683 ], ... ],
    	[ [ 0.512 ], [ -0.924 ], [ -1.856 ], [ 0.728 ], [ -1.52 ], [ -0.652 ], [ 1.57172325065697 ], [ -0.268 ], ... ],
    	[ [ 25.642435248966244 ], [ 26.35401137140119 ], [ 26.03730734977391 ], [ 26.437991099263137 ], [ -1.588 ], [ -1.972 ], [ -1.476 ], [ -0.252 ], ... ],
    	[ [ 0.732 ], [ -0.732 ], [ -1.344 ], [ 0.9680000000155264 ], [ 26.41693134053351 ], [ -1.588 ], [ 1.0840000377470722 ], [ 22.097793302796706 ], ... ],
    	[ [ 1.2799637233147008 ], [ -0.244 ], [ 0.656 ], [ 26.33559258838332 ], [ -0.688 ], [ 26.051457135825114 ], [ -1.624 ], [ -1.724 ], ... ],
    	[ [ -204.9062326750834 ], [ -1.852 ], [ 26.078913569181033 ], [ -0.436 ], [ 12128.594164213375 ], [ -1.16 ], [ -1.504 ], [ 0.44 ], ... ],
    	[ [ 25.63721134377659 ], [ -1.632 ], [ 26.0118641144232 ], [ 1.569256832491544 ], [ -1.072 ], [ 26.29016239474529 ], [ 1.0800000007114323 ], [ 26.390095577262244 ], ... ],
    	[ [ 26.43825179883715 ], [ 26.366141224476657 ], [ 26.114847717197485 ], [ -0.612 ], [ 26.414814187251185 ], [ -0.816 ], [ -1.668 ], [ 0.4999999999999999 ], ... ],
    	...
    ]
```



Next, we run the same optimization using L-BFGS, which is nearly ideal for purely second-order or quadratic functions.

Code from [LearningTester.java:249](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L249) executed in 2.11 seconds: 
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
    th(0)=0.6933974985881487;dx=-6.690032321176461E-4
    New Minimum: 0.6933974985881487 > 0.6919745279173611
    WOLFE (weak): th(2.154434690031884)=0.6919745279173611; dx=-6.521113865401052E-4 delta=0.00142297067078756
    New Minimum: 0.6919745279173611 > 0.6905870322407682
    WOLFE (weak): th(4.308869380063768)=0.6905870322407682; dx=-6.360602174434209E-4 delta=0.002810466347380469
    New Minimum: 0.6905870322407682 > 0.6853587857292347
    END: th(12.926608140191302)=0.6853587857292347; dx=-5.791019182255756E-4 delta=0.008038712858913999
    Iteration 1 complete. Error: 0.6853587857292347 Total: 239692355397094.6600; Orientation: 0.0005; Line Search: 0.0081
    LBFGS Accumulation History: 1 points
    th(0)=0.6853587857292347;dx=-5.050393679205765E-4
    New Minimum: 0.6853587857292347 > 0.672695601273769
    END: th(27.849533001676672)=0.672695601273769; dx=-4.106500133349569E-4 delta=0.01266318445546566
    Iteration 2 complete. Error: 0.672695601273769 Total: 
```
...[skipping 119933 bytes](etc/302.txt)...
```
    ion 248 complete. Error: 0.391150845929988 Total: 239694436886377.5600; Orientation: 0.0005; Line Search: 0.0088
    LBFGS Accumulation History: 1 points
    th(0)=0.39115915348492475;dx=-2.1892839138297878E-7
    New Minimum: 0.39115915348492475 > 0.39113056225491444
    END: th(139.95319722741473)=0.39113056225491444; dx=-1.8913224205398885E-7 delta=2.859123001031083E-5
    Iteration 249 complete. Error: 0.39113056225491444 Total: 239694442110601.5600; Orientation: 0.0006; Line Search: 0.0034
    LBFGS Accumulation History: 1 points
    th(0)=0.39113056225491444;dx=-1.8130608775902574E-7
    New Minimum: 0.39113056225491444 > 0.39107823758651733
    WOLFE (weak): th(301.5200230876164)=0.39107823758651733; dx=-1.6386027326880302E-7 delta=5.232466839710792E-5
    New Minimum: 0.39107823758651733 > 0.39103343869111845
    END: th(603.0400461752328)=0.39103343869111845; dx=-1.2920966889412307E-7 delta=9.712356379598663E-5
    Iteration 250 complete. Error: 0.39103343869111845 Total: 239694449754576.5600; Orientation: 0.0008; Line Search: 0.0054
    
```

Returns: 

```
    0.39103343869111845
```



This training run resulted in the following regressed input:

Code from [LearningTester.java:154](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L154) executed in 0.00 seconds: 
```java
    return Arrays.stream(input_lbgfs).map(x -> x.prettyPrint()).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [
    	[ [ -1.672 ], [ -1.044 ], [ 3.638197945851071 ], [ -1.564 ], [ 3.57537152262455 ], [ 3.4407082641872577 ], [ -0.2 ], [ 3.6002899228755094 ], ... ],
    	[ [ 0.5119999999999999 ], [ -0.924 ], [ -1.856 ], [ 0.7280000763606789 ], [ -1.52 ], [ -0.652 ], [ 1.525931976058526 ], [ -0.268 ], ... ],
    	[ [ 3.631168410111063 ], [ 3.470049877525578 ], [ 3.5429774391335354 ], [ 3.441866617510607 ], [ -1.588 ], [ -1.972 ], [ -1.476 ], [ -0.252 ], ... ],
    	[ [ 0.7320000008442512 ], [ -0.732 ], [ -1.344 ], [ 0.9741936888507035 ], [ 3.4410185805351206 ], [ -1.588 ], [ 1.1110502342324853 ], [ 3.439838580318489 ], ... ],
    	[ [ 1.2957296081953855 ], [ -0.244 ], [ 0.6560000000738866 ], [ 3.440336075812895 ], [ -0.688 ], [ 3.539788993714771 ], [ -1.624 ], [ -1.724 ], ... ],
    	[ [ 0.40399999999999997 ], [ -1.852 ], [ 3.5335921327859015 ], [ -0.436 ], [ 0.3 ], [ -1.16 ], [ -1.504 ], [ 0.44000000000000006 ], ... ],
    	[ [ 3.632329841659347 ], [ -1.632 ], [ 3.548702783533778 ], [ 1.4999365604593855 ], [ -1.072 ], [ 3.4851940597381814 ], [ 1.0808543810299336 ], [ 3.44064100093271 ], ... ],
    	[ [ 3.448101471401368 ], [ 3.467109422316887 ], [ 3.5254593611838456 ], [ -0.612 ], [ 3.4548508776216273 ], [ -0.816 ], [ -1.668 ], [ 0.5 ], ... ],
    	...
    ]
```



Code from [LearningTester.java:95](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L95) executed in 0.00 seconds: 
```java
    return TestUtil.compare(runs);
```

Returns: 

![Result](etc/test.205.png)



Code from [LearningTester.java:98](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L98) executed in 0.00 seconds: 
```java
    return TestUtil.compareTime(runs);
```

Returns: 

![Result](etc/test.206.png)



### Performance
Now we execute larger-scale runs to benchmark performance:

Code from [PerformanceTester.java:66](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/PerformanceTester.java#L66) executed in 0.19 seconds: 
```java
    test(component, inputPrototype);
```
Logging: 
```
    100 batches
    Input Dimensions:
    	[100, 100, 1]
    Performance:
    	Evaluation performance: 0.016198s +- 0.012303s [0.009246s - 0.040786s]
    	Learning performance: 0.011364s +- 0.000239s [0.011098s - 0.011774s]
    
```

### Function Plots
Code from [ActivationLayerTestBase.java:110](../../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/java/ActivationLayerTestBase.java#L110) executed in 0.00 seconds: 
```java
    return plot("Value Plot", plotData, x -> new double[]{x[0], x[1]});
```

Returns: 

![Result](etc/test.207.png)



Code from [ActivationLayerTestBase.java:114](../../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/java/ActivationLayerTestBase.java#L114) executed in 0.00 seconds: 
```java
    return plot("Derivative Plot", plotData, x -> new double[]{x[0], x[2]});
```

Returns: 

![Result](etc/test.208.png)



