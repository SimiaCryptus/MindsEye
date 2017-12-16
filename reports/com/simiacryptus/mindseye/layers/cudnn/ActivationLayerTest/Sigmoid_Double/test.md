# ActivationLayer
## Sigmoid_Double
### Batch Execution
Code from [BatchingTester.java:66](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/BatchingTester.java#L66) executed in 0.01 seconds: 
```java
    return test(reference, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (20#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (20#)}
```



Code from [SingleDerivativeTester.java:77](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/SingleDerivativeTester.java#L77) executed in 0.01 seconds: 
```java
    return test(component, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ -1.22 ] ]
    ]
    Inputs Statistics: {meanExponent=0.08635983067474821, negative=1, min=-1.22, max=-1.22, mean=-1.22, count=1.0, positive=0, stdDev=0.0, zeros=0}
    Output: [
    	[ [ 0.22793645057321626 ] ]
    ]
    Outputs Statistics: {meanExponent=-0.6421862188433686, negative=0, min=0.22793645057321626, max=0.22793645057321626, mean=0.22793645057321626, count=1.0, positive=1, stdDev=0.0, zeros=0}
    Feedback for input 0
    Inputs Values: [
    	[ [ -1.22 ] ]
    ]
    Value Statistics: {meanExponent=0.08635983067474821, negative=1, min=-1.22, max=-1.22, mean=-1.22, count=1.0, positive=0, stdDev=0.0, zeros=0}
    Implemented Feedback: [ [ 0.17598142507330003 ] ]
    Implemented Statistics: {meanExponent=-0.7545331697647699, negative=0, min=0.17598142507330003, max=0.17598142507330003, mean=0.17598142507330003, count=1.0, positive=1, stdDev=0.0, zeros=0}
    Measured Feedback: [ [ 0.17598621287001626 ] ]
    Measured Statistics: {meanExponent=-0.7545213543961377, negative=0, min=0.17598621287001626, max=0.17598621287001626, mean=0.17598621287001626, count=1.0, positive=1, stdDev=0.0, zeros=0}
    Feedback Error: [ [ 4.787796716226866E-6 ] ]
    Error Statistics: {meanExponent=-5.3198642974667845, negative=0, min=4.787796716226866E-6, max=4.787796716226866E-6, mean=4.787796716226866E-6, count=1.0, positive=1, stdDev=0.0, zeros=0}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 4.7878e-06 +- 0.0000e+00 [4.7878e-06 - 4.7878e-06] (1#)
    relativeTol: 1.3603e-05 +- 0.0000e+00 [1.3603e-05 - 1.3603e-05] (1#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=4.7878e-06 +- 0.0000e+00 [4.7878e-06 - 4.7878e-06] (1#), relativeTol=1.3603e-05 +- 0.0000e+00 [1.3603e-05 - 1.3603e-05] (1#)}
```



### Reference Implementation
Code from [EquivalencyTester.java:61](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/EquivalencyTester.java#L61) executed in 0.00 seconds: 
```java
    System.out.println(new GsonBuilder().setPrettyPrinting().create().toJson(this.reference.getJson()));
```
Logging: 
```
    {
      "class": "com.simiacryptus.mindseye.layers.java.SigmoidActivationLayer",
      "id": "601297a1-3057-4a3c-b437-e1072564e6ab",
      "isFrozen": true,
      "name": "SigmoidActivationLayer/601297a1-3057-4a3c-b437-e1072564e6ab",
      "balanced": false
    }
    
```

Code from [EquivalencyTester.java:64](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/EquivalencyTester.java#L64) executed in 0.00 seconds: 
```java
    return test(subject, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ -0.94 ] ]
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
      "class": "com.simiacryptus.mindseye.layers.cudnn.ActivationLayer",
      "id": "84837399-5778-4c83-b4ed-229f3d90c380",
      "isFrozen": false,
      "name": "ActivationLayer/84837399-5778-4c83-b4ed-229f3d90c380",
      "mode": 0
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
    	[ [ -0.096 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 0.4760184150288955 ] ]
    ]
    --------------------
    Derivative: 
    [
    	[ [ 0.2494248835822737 ] ]
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
    	[ [ 1.788 ], [ 0.348 ], [ -1.028 ], [ -1.312 ], [ 1.952 ], [ -0.048 ], [ -1.964 ], [ 0.604 ], ... ],
    	[ [ -0.452 ], [ 0.768 ], [ -0.188 ], [ -1.676 ], [ 0.524 ], [ -1.264 ], [ -0.652 ], [ -1.588 ], ... ],
    	[ [ -0.332 ], [ -0.676 ], [ 0.056 ], [ -0.42 ], [ 0.06 ], [ -0.996 ], [ 1.404 ], [ 0.096 ], ... ],
    	[ [ 1.48 ], [ 1.924 ], [ 1.948 ], [ -0.508 ], [ -1.916 ], [ -0.228 ], [ -0.12 ], [ 1.008 ], ... ],
    	[ [ -1.192 ], [ -1.056 ], [ -1.912 ], [ -1.064 ], [ 1.26 ], [ -1.292 ], [ -1.872 ], [ 1.18 ], ... ],
    	[ [ -1.14 ], [ 0.88 ], [ -1.336 ], [ 0.776 ], [ 0.38 ], [ -1.112 ], [ -0.416 ], [ -0.68 ], ... ],
    	[ [ -0.9 ], [ -1.972 ], [ 1.952 ], [ 0.184 ], [ -0.164 ], [ 0.876 ], [ -1.38 ], [ 0.656 ], ... ],
    	[ [ -0.024 ], [ 0.74 ], [ -0.004 ], [ 0.108 ], [ 1.732 ], [ 1.4 ], [ -1.652 ], [ 0.164 ], ... ],
    	...
    ]
```



First, we use a conjugate gradient descent method, which converges the fastest for purely linear functions.

Code from [LearningTester.java:225](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L225) executed in 3.66 seconds: 
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
    F(0.0) = LineSearchPoint{point=PointSample{avg=0.11897268403125082}, derivative=-1.512199590312867E-6}
    New Minimum: 0.11897268403125082 > 0.11897268403125072
    F(1.0E-10) = LineSearchPoint{point=PointSample{avg=0.11897268403125072}, derivative=-1.5121995903128668E-6}, delta = -9.71445146547012E-17
    New Minimum: 0.11897268403125072 > 0.11897268403124983
    F(7.000000000000001E-10) = LineSearchPoint{point=PointSample{avg=0.11897268403124983}, derivative=-1.5121995903128657E-6}, delta = -9.853229343548264E-16
    New Minimum: 0.11897268403124983 > 0.11897268403124299
    F(4.900000000000001E-9) = LineSearchPoint{point=PointSample{avg=0.11897268403124299}, derivative=-1.5121995903128579E-6}, delta = -7.827072323607354E-15
    New Minimum: 0.11897268403124299 > 0.11897268403119908
    F(3.430000000000001E-8) = LineSearchPoint{point=PointSample{avg=0.11897268403119908}, derivative=-1.5121995903128034E-6}, delta = -5.1736392947532295E-14
    New Minimum: 0.11897268403119908 > 0.11897268403088
```
...[skipping 320255 bytes](etc/25.txt)...
```
    rivative=-1.842545014981016E-12}, delta = -6.9759836214666054E-6
    Left bracket at 156716.8320655504
    New Minimum: 0.0027513587770539617 > 0.002751355704432646
    F(159979.341754203) = LineSearchPoint{point=PointSample{avg=0.002751355704432646}, derivative=-4.077842461731897E-14}, delta = -6.979056242782406E-6
    Left bracket at 159979.341754203
    Converged to left
    Iteration 249 complete. Error: 0.002751355704432646 Total: 239440976256681.0300; Orientation: 0.0004; Line Search: 0.0167
    Low gradient: 9.176939438651089E-6
    F(0.0) = LineSearchPoint{point=PointSample{avg=0.002751355704432646}, derivative=-8.421621746066976E-11}
    New Minimum: 0.002751355704432646 > 0.002744618104565846
    F(159979.341754203) = LineSearchPoint{point=PointSample{avg=0.002744618104565846}, derivative=5.461216921165401E-13}, delta = -6.737599866800045E-6
    0.002744618104565846 <= 0.002751355704432646
    Converged to right
    Iteration 250 complete. Error: 0.002744618104565846 Total: 239440980871906.0300; Orientation: 0.0003; Line Search: 0.0033
    
```

Returns: 

```
    0.002744618104565846
```



This training run resulted in the following regressed input:

Code from [LearningTester.java:144](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L144) executed in 0.00 seconds: 
```java
    return Arrays.stream(input_gd).map(x -> x.prettyPrint()).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [
    	[ [ 1.788000000000001 ], [ 0.3479999999999984 ], [ -1.028 ], [ -1.312 ], [ 1.9520000000000008 ], [ -0.02571730534071623 ], [ -9.061350263524021 ], [ 0.6039999999999999 ], ... ],
    	[ [ -0.45199999999999957 ], [ 0.768 ], [ -0.18798905299276006 ], [ -8.409339926797387 ], [ 0.5239999999999997 ], [ -1.264000000046103 ], [ -0.6519999999999998 ], [ -7.778767284482382 ], ... ],
    	[ [ -0.33200000000004165 ], [ -0.6760000000000002 ], [ 0.07247759254300384 ], [ -0.4200000000000002 ], [ 0.050451435747487006 ], [ -0.996 ], [ 1.4039999999999997 ], [ 0.102386497411034 ], ... ],
    	[ [ 1.48 ], [ 9.080353346982541 ], [ 7.645967742804252 ], [ -0.5080000000000003 ], [ -1.9160000000000001 ], [ -0.22799991493826835 ], [ -0.12215410429895074 ], [ 1.008 ], ... ],
    	[ [ -1.192 ], [ -1.056 ], [ -1.9120000000000004 ], [ -1.0639999999999998 ], [ 1.26 ], [ -1.2920000000000003 ], [ -8.636133341862669 ], [ 1.18 ], ... ],
    	[ [ -1.1399999999999997 ], [ 0.8800000000000002 ], [ -1.336 ], [ 0.7760000000000005 ], [ 0.3799999999999999 ], [ -1.112 ], [ -0.4160000000000002 ], [ -0.6799999999999999 ], ... ],
    	[ [ -0.9 ], [ -8.623345154939251 ], [ 1.9520000000000002 ], [ 0.184017828064402 ], [ -0.16402942034722115 ], [ 0.8760000000000001 ], [ -1.38 ], [ 0.6559999999999999 ], ... ],
    	[ [ -0.05889951087639189 ], [ 0.74 ], [ -0.022252365168256384 ], [ 0.10443403093927252 ], [ 1.7320000000000007 ], [ 1.3999999999999995 ], [ -5.824077848850571 ], [ 0.1640975565830361 ], ... ],
    	...
    ]
```



Next, we run the same optimization using L-BFGS, which is nearly ideal for purely second-order or quadratic functions.

Code from [LearningTester.java:249](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L249) executed in 4.34 seconds: 
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
    th(0)=0.11897268403125082;dx=-1.512199590312867E-6
    New Minimum: 0.11897268403125082 > 0.11896942610029054
    WOLFE (weak): th(2.154434690031884)=0.11896942610029054; dx=-1.5121956022619115E-6 delta=3.257930960273714E-6
    New Minimum: 0.11896942610029054 > 0.11896616817792383
    WOLFE (weak): th(4.308869380063768)=0.11896616817792383; dx=-1.5121916128767443E-6 delta=6.515853326991006E-6
    New Minimum: 0.11896616817792383 > 0.11895313657445042
    WOLFE (weak): th(12.926608140191302)=0.11895313657445042; dx=-1.5121756419935814E-6 delta=1.9547456800397822E-5
    New Minimum: 0.11895313657445042 > 0.11889449606581952
    WOLFE (weak): th(51.70643256076521)=0.11889449606581952; dx=-1.5121035088094377E-6 delta=7.81879654312978E-5
    New Minimum: 0.11889449606581952 > 0.11858179448077935
    WOLFE (weak): th(258.53216280382605)=0.11858179448077935; dx=-1.5117114926767684E-6 delta=3.9088955047146545E-4
    New Minimum: 0.11858179448077935 > 0.11662937657089424
```
...[skipping 323973 bytes](etc/26.txt)...
```
    -4 > 2.539311217538591E-4
    WOLFE (weak): th(4.308869380063768)=2.539311217538591E-4; dx=-2.5259068741075716E-9 delta=1.0883943430382587E-8
    New Minimum: 2.539311217538591E-4 > 2.539093547108543E-4
    WOLFE (weak): th(12.926608140191302)=2.539093547108543E-4; dx=-2.5257763132910024E-9 delta=3.2650986435183924E-8
    New Minimum: 2.539093547108543E-4 > 2.5381141694073254E-4
    WOLFE (weak): th(51.70643256076521)=2.5381141694073254E-4; dx=-2.525188803849481E-9 delta=1.305887565569491E-7
    New Minimum: 2.5381141694073254E-4 > 2.5328946692530183E-4
    WOLFE (weak): th(258.53216280382605)=2.5328946692530183E-4; dx=-2.5220558134878625E-9 delta=6.525387719876559E-7
    New Minimum: 2.5328946692530183E-4 > 2.5004195321171036E-4
    WOLFE (weak): th(1551.1929768229563)=2.5004195321171036E-4; dx=-2.502489626937494E-9 delta=3.900052485579129E-6
    MAX ALPHA: th(0)=2.539420056972895E-4;th'(0)=-2.5259721549471654E-9;
    Iteration 250 complete. Error: 2.5004195321171036E-4 Total: 239445330109783.7000; Orientation: 0.0005; Line Search: 0.0150
    
```

Returns: 

```
    2.5004195321171036E-4
```



This training run resulted in the following regressed input:

Code from [LearningTester.java:154](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L154) executed in 0.00 seconds: 
```java
    return Arrays.stream(input_lbgfs).map(x -> x.prettyPrint()).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [
    	[ [ 1.7429127222255087 ], [ 0.33785764467173174 ], [ -0.9731049266997784 ], [ -1.1822154862236491 ], [ 1.9039496034623458 ], [ -0.03657255163224757 ], [ -1.656555080591914 ], [ 0.5811995398748493 ], ... ],
    	[ [ -0.44466161780609953 ], [ 0.7156358419359938 ], [ -0.19342309721607104 ], [ -1.4809666390915397 ], [ 0.5288035415283522 ], [ -1.1844453212497372 ], [ -0.6349927703334737 ], [ -1.4134254219113396 ], ... ],
    	[ [ -0.3562167735471529 ], [ -0.6712540716966978 ], [ 0.09026007767500199 ], [ -0.41061144855855514 ], [ 0.06777767211953968 ], [ -0.9267370769658154 ], [ 1.2395110365933317 ], [ 0.08380156590765747 ], ... ],
    	[ [ 1.4999600073410817 ], [ 1.6240102116946635 ], [ 1.6125135035115337 ], [ -0.5212378914784848 ], [ -1.8489264792149511 ], [ -0.233553688873855 ], [ -0.11053356416415128 ], [ 0.9648361783309852 ], ... ],
    	[ [ -1.2845160785539187 ], [ -1.1054832441943403 ], [ -1.7625918387453312 ], [ -0.9960973553407936 ], [ 1.364518331058305 ], [ -1.1544987673343878 ], [ -1.6105017892193147 ], [ 1.1471213327707324 ], ... ],
    	[ [ -1.210096624471513 ], [ 0.8722273756038437 ], [ -1.2088589233249118 ], [ 0.7718642946685851 ], [ 0.3813571184221135 ], [ -1.133279048858351 ], [ -0.4182031846540118 ], [ -0.6371196151982998 ], ... ],
    	[ [ -0.9012076868193623 ], [ -1.669229395688809 ], [ 1.8168259239428413 ], [ 0.22486373316950853 ], [ -0.1512116332954336 ], [ 0.8481372275941373 ], [ -1.3838408022467208 ], [ 0.6306416691478661 ], ... ],
    	[ [ -0.03369734448695939 ], [ 0.7464482167097805 ], [ -0.00411974751121621 ], [ 0.13237530111752863 ], [ 1.461003780102434 ], [ 1.4741776293341209 ], [ -1.4405572964710855 ], [ 0.166457175132229 ], ... ],
    	...
    ]
```



Code from [LearningTester.java:95](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L95) executed in 0.00 seconds: 
```java
    return TestUtil.compare(runs);
```

Returns: 

![Result](etc/test.21.png)



Code from [LearningTester.java:98](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L98) executed in 0.00 seconds: 
```java
    return TestUtil.compareTime(runs);
```

Returns: 

![Result](etc/test.22.png)



### Performance
Now we execute larger-scale runs to benchmark performance:

Code from [PerformanceTester.java:66](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/PerformanceTester.java#L66) executed in 0.20 seconds: 
```java
    test(component, inputPrototype);
```
Logging: 
```
    100 batches
    Input Dimensions:
    	[100, 100, 1]
    Performance:
    	Evaluation performance: 0.006072s +- 0.000600s [0.005442s - 0.007213s]
    	Learning performance: 0.026983s +- 0.002062s [0.023788s - 0.029933s]
    
```

### Function Plots
Code from [ActivationLayerTest.java:90](../../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/cudnn/ActivationLayerTest.java#L90) executed in 0.00 seconds: 
```java
    return ActivationLayerTestBase.plot("Value Plot", plotData, x -> new double[]{x[0], x[1]});
```

Returns: 

![Result](etc/test.23.png)



Code from [ActivationLayerTest.java:94](../../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/cudnn/ActivationLayerTest.java#L94) executed in 0.00 seconds: 
```java
    return ActivationLayerTestBase.plot("Derivative Plot", plotData, x -> new double[]{x[0], x[2]});
```

Returns: 

![Result](etc/test.24.png)



