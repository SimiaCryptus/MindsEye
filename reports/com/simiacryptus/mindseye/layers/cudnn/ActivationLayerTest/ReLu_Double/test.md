# ActivationLayer
## ReLu_Double
### Batch Execution
Code from [BatchingTester.java:66](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/BatchingTester.java#L66) executed in 0.02 seconds: 
```java
    return test(reference, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (20#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (10#)}
```



Code from [SingleDerivativeTester.java:77](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/SingleDerivativeTester.java#L77) executed in 0.01 seconds: 
```java
    return test(component, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ 1.008 ] ]
    ]
    Inputs Statistics: {meanExponent=0.003460532109506489, negative=0, min=1.008, max=1.008, mean=1.008, count=1.0, positive=1, stdDev=0.0, zeros=0}
    Output: [
    	[ [ 1.008 ] ]
    ]
    Outputs Statistics: {meanExponent=0.003460532109506489, negative=0, min=1.008, max=1.008, mean=1.008, count=1.0, positive=1, stdDev=0.0, zeros=0}
    Feedback for input 0
    Inputs Values: [
    	[ [ 1.008 ] ]
    ]
    Value Statistics: {meanExponent=0.003460532109506489, negative=0, min=1.008, max=1.008, mean=1.008, count=1.0, positive=1, stdDev=0.0, zeros=0}
    Implemented Feedback: [ [ 1.0 ] ]
    Implemented Statistics: {meanExponent=0.0, negative=0, min=1.0, max=1.0, mean=1.0, count=1.0, positive=1, stdDev=0.0, zeros=0}
    Measured Feedback: [ [ 0.9999999999998899 ] ]
    Measured Statistics: {meanExponent=-4.7830642341045674E-14, negative=0, min=0.9999999999998899, max=0.9999999999998899, mean=0.9999999999998899, count=1.0, positive=1, stdDev=0.0, zeros=0}
    Feedback Error: [ [ -1.1013412404281553E-13 ] ]
    Error Statistics: {meanExponent=-12.958078098036825, negative=1, min=-1.1013412404281553E-13, max=-1.1013412404281553E-13, mean=-1.1013412404281553E-13, count=1.0, positive=0, stdDev=0.0, zeros=0}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 1.1013e-13 +- 0.0000e+00 [1.1013e-13 - 1.1013e-13] (1#)
    relativeTol: 5.5067e-14 +- 0.0000e+00 [5.5067e-14 - 5.5067e-14] (1#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=1.1013e-13 +- 0.0000e+00 [1.1013e-13 - 1.1013e-13] (1#), relativeTol=5.5067e-14 +- 0.0000e+00 [5.5067e-14 - 5.5067e-14] (1#)}
```



### Reference Implementation
Code from [EquivalencyTester.java:61](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/EquivalencyTester.java#L61) executed in 0.00 seconds: 
```java
    System.out.println(new GsonBuilder().setPrettyPrinting().create().toJson(this.reference.getJson()));
```
Logging: 
```
    {
      "class": "com.simiacryptus.mindseye.layers.java.ReLuActivationLayer",
      "id": "057389e6-b61e-4310-b59e-78f57108d839",
      "isFrozen": true,
      "name": "ReLuActivationLayer/057389e6-b61e-4310-b59e-78f57108d839",
      "weights": [
        1.0
      ]
    }
    
```

Code from [EquivalencyTester.java:64](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/EquivalencyTester.java#L64) executed in 0.01 seconds: 
```java
    return test(subject, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ 1.664 ] ]
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
      "id": "ffbf9129-925c-4161-9612-5253ec29c240",
      "isFrozen": false,
      "name": "ActivationLayer/ffbf9129-925c-4161-9612-5253ec29c240",
      "mode": 1
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
    	[ [ -0.724 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 0.0 ] ]
    ]
    --------------------
    Derivative: 
    [
    	[ [ 0.0 ] ]
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
    	[ [ -1.976 ], [ -1.512 ], [ 0.036 ], [ 1.888 ], [ -1.376 ], [ 0.232 ], [ -0.84 ], [ -0.688 ], ... ],
    	[ [ 0.496 ], [ 0.384 ], [ 0.936 ], [ 0.852 ], [ -1.468 ], [ 0.284 ], [ -0.024 ], [ 0.628 ], ... ],
    	[ [ 0.56 ], [ 0.232 ], [ 1.092 ], [ 0.02 ], [ 0.936 ], [ 0.028 ], [ 1.552 ], [ 1.276 ], ... ],
    	[ [ 0.776 ], [ 0.484 ], [ -1.172 ], [ 0.072 ], [ -0.264 ], [ -1.744 ], [ 1.912 ], [ 0.924 ], ... ],
    	[ [ -0.124 ], [ -1.028 ], [ -1.272 ], [ 1.344 ], [ -0.56 ], [ 1.104 ], [ 1.136 ], [ 0.924 ], ... ],
    	[ [ -0.64 ], [ -1.052 ], [ 1.592 ], [ 0.056 ], [ 1.7 ], [ -0.048 ], [ -0.652 ], [ 1.416 ], ... ],
    	[ [ -1.52 ], [ 1.304 ], [ -1.308 ], [ -0.512 ], [ -0.52 ], [ 1.296 ], [ -0.092 ], [ -0.3 ], ... ],
    	[ [ -0.536 ], [ -1.128 ], [ -0.952 ], [ 1.856 ], [ 0.304 ], [ 1.088 ], [ -1.308 ], [ -1.292 ], ... ],
    	...
    ]
```



First, we use a conjugate gradient descent method, which converges the fastest for purely linear functions.

Code from [LearningTester.java:225](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L225) executed in 0.10 seconds: 
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
    F(0.0) = LineSearchPoint{point=PointSample{avg=0.8517080575999928}, derivative=-2.0676370624000002E-4}
    New Minimum: 0.8517080575999928 > 0.8517080575999713
    F(1.0E-10) = LineSearchPoint{point=PointSample{avg=0.8517080575999713}, derivative=-2.0676370623999587E-4}, delta = -2.1538326677728037E-14
    New Minimum: 0.8517080575999713 > 0.8517080575998532
    F(7.000000000000001E-10) = LineSearchPoint{point=PointSample{avg=0.8517080575998532}, derivative=-2.0676370623997107E-4}, delta = -1.3955503419538218E-13
    New Minimum: 0.8517080575998532 > 0.8517080575989853
    F(4.900000000000001E-9) = LineSearchPoint{point=PointSample{avg=0.8517080575989853}, derivative=-2.0676370623979738E-4}, delta = -1.0075273948473296E-12
    New Minimum: 0.8517080575989853 > 0.8517080575929018
    F(3.430000000000001E-8) = LineSearchPoint{point=PointSample{avg=0.8517080575929018}, derivative=-2.067637062385816E-4}, delta = -7.090994458280875E-12
    New Minimum: 0.8517080575929018 > 0.8517080575503538
    F(2.401
```
...[skipping 3242 bytes](etc/21.txt)...
```
    lta = 0.0
    Right bracket at 5964.222669518625
    F(5596.2521213946) = LineSearchPoint{point=PointSample{avg=0.3347987919999996}, derivative=6.573177472056334E-33}, delta = 0.0
    Right bracket at 5596.2521213946
    F(5377.638262887134) = LineSearchPoint{point=PointSample{avg=0.3347987919999996}, derivative=4.1595851257519414E-33}, delta = 0.0
    Right bracket at 5377.638262887134
    F(5242.766426025059) = LineSearchPoint{point=PointSample{avg=0.3347987919999996}, derivative=2.6765318763085096E-33}, delta = 0.0
    Right bracket at 5242.766426025059
    F(5157.3948017448765) = LineSearchPoint{point=PointSample{avg=0.3347987919999996}, derivative=1.7338165701807237E-33}, delta = 0.0
    Right bracket at 5157.3948017448765
    F(5102.67907769948) = LineSearchPoint{point=PointSample{avg=0.3347987919999996}, derivative=1.1299720573768589E-33}, delta = 0.0
    Right bracket at 5102.67907769948
    Converged to right
    Iteration 2 failed, aborting. Error: 0.3347987919999996 Total: 239433248828829.7800; Orientation: 0.0004; Line Search: 0.0375
    
```

Returns: 

```
    0.3347987919999996
```



This training run resulted in the following regressed input:

Code from [LearningTester.java:144](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L144) executed in 0.01 seconds: 
```java
    return Arrays.stream(input_gd).map(x -> x.prettyPrint()).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [
    	[ [ -1.812 ], [ -6.292616526750005E-16 ], [ 0.03599999999999986 ], [ 1.8880000000000003 ], [ -0.092 ], [ 0.23199999999999904 ], [ -0.092 ], [ -1.36 ], ... ],
    	[ [ 0.4960000000000002 ], [ 0.38399999999999956 ], [ -1.96 ], [ 0.8519999999999994 ], [ -8.754944732869534E-16 ], [ -1.296 ], [ -1.944 ], [ -1.036 ], ... ],
    	[ [ -1.264 ], [ -0.52 ], [ 1.092 ], [ -0.04 ], [ 0.9360000000000004 ], [ 0.027999999999999574 ], [ 1.552 ], [ -0.768 ], ... ],
    	[ [ -1.692 ], [ 0.4839999999999993 ], [ -1.008 ], [ 0.07199999999999976 ], [ -7.934168664163066E-16 ], [ -8.390155368999944E-16 ], [ -0.868 ], [ 0.9239999999999995 ], ... ],
    	[ [ -8.937339414804297E-16 ], [ -1.164 ], [ -1.812 ], [ 1.344 ], [ -7.022195254489122E-16 ], [ 1.1039999999999996 ], [ 1.1360000000000003 ], [ -0.496 ], ... ],
    	[ [ -6.566208549652182E-16 ], [ -3.967084332081533E-16 ], [ 1.5920000000000007 ], [ 0.055999999999998995 ], [ 1.7 ], [ -0.96 ], [ -1.1171674268505387E-15 ], [ 1.416 ], ... ],
    	[ [ -0.308 ], [ 1.3039999999999998 ], [ -5.312245111350517E-16 ], [ -2.0633398393872296E-16 ], [ -7.979767334646741E-16 ], [ -0.504 ], [ -8.709346062385859E-16 ], [ -5.791031151429326E-16 ], ... ],
    	[ [ -7.124792263077431E-17 ], [ -0.072 ], [ -3.1463082633750025E-16 ], [ 1.8560000000000012 ], [ 0.3039999999999995 ], [ -0.276 ], [ -0.828 ], [ -0.3 ], ... ],
    	...
    ]
```



Next, we run the same optimization using L-BFGS, which is nearly ideal for purely second-order or quadratic functions.

Code from [LearningTester.java:249](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L249) executed in 0.09 seconds: 
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
    th(0)=0.8517080575999928;dx=-2.0676370624000002E-4
    New Minimum: 0.8517080575999928 > 0.8512626946698488
    WOLFE (weak): th(2.154434690031884)=0.8512626946698488; dx=-2.066746144597274E-4 delta=4.4536293014396655E-4
    New Minimum: 0.8512626946698488 > 0.8508175236821142
    WOLFE (weak): th(4.308869380063768)=0.8508175236821142; dx=-2.065855226794548E-4 delta=8.905339178786331E-4
    New Minimum: 0.8508175236821142 > 0.8490387591554217
    WOLFE (weak): th(12.926608140191302)=0.8490387591554217; dx=-2.062291555583644E-4 delta=0.0026692984445710888
    New Minimum: 0.8490387591554217 > 0.8410723233848264
    WOLFE (weak): th(51.70643256076521)=0.8410723233848264; dx=-2.0462550351345754E-4 delta=0.010635734215166393
    New Minimum: 0.8410723233848264 > 0.7996349748749498
    WOLFE (weak): th(258.53216280382605)=0.7996349748749498; dx=-1.960726926072876E-4 delta=0.05207308272504296
    New Minimum: 0.7996349748749498 > 0.5807291244051709
    END: th(1551.1929768
```
...[skipping 2445 bytes](etc/22.txt)...
```
    2)=0.3347988208969258; dx=-1.1558770101516378E-10 delta=2.860795598746968E-6
    Iteration 6 complete. Error: 0.3347988208969258 Total: 239433329715385.7000; Orientation: 0.0006; Line Search: 0.0072
    LBFGS Accumulation History: 1 points
    th(0)=0.3347988208969258;dx=-1.1558770101516215E-11
    New Minimum: 0.3347988208969258 > 0.33479880068558016
    WOLF (strong): th(9694.956105143481)=0.33479880068558016; dx=3.699963102874744E-12 delta=2.0211345652665358E-8
    New Minimum: 0.33479880068558016 > 0.33479879202688767
    END: th(4847.478052571741)=0.33479879202688767; dx=-3.525932251517205E-13 delta=2.8870038137895904E-8
    Iteration 7 complete. Error: 0.33479879202688767 Total: 239433338205461.6600; Orientation: 0.0005; Line Search: 0.0067
    LBFGS Accumulation History: 1 points
    th(0)=0.33479879202688767;dx=-1.0755641070029068E-14
    MAX ALPHA: th(0)=0.33479879202688767;th'(0)=-1.0755641070029068E-14;
    Iteration 8 failed, aborting. Error: 0.33479879202688767 Total: 239433345630003.6600; Orientation: 0.0006; Line Search: 0.0043
    
```

Returns: 

```
    0.33479879202688767
```



This training run resulted in the following regressed input:

Code from [LearningTester.java:154](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L154) executed in 0.01 seconds: 
```java
    return Arrays.stream(input_lbgfs).map(x -> x.prettyPrint()).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [
    	[ [ -1.812 ], [ 7.875962408821272E-6 ], [ 0.036001759830428345 ], [ 1.8879935376718695 ], [ -0.092 ], [ 0.23201226111364012 ], [ -0.092 ], [ -1.36 ], ... ],
    	[ [ 0.4959975189275928 ], [ 0.3840057410861515 ], [ -1.96 ], [ 0.8520076740146547 ], [ 1.0991727757365998E-5 ], [ -1.296 ], [ -1.944 ], [ -1.036 ], ... ],
    	[ [ -1.264 ], [ -0.52 ], [ 1.091999971150321 ], [ -0.04 ], [ 0.9359959898945978 ], [ 0.02800533719064334 ], [ 1.5519996538038503 ], [ -0.768 ], ... ],
    	[ [ -1.692 ], [ 0.48400871260310424 ], [ -1.008 ], [ 0.07200288496791532 ], [ 9.953139307851107E-6 ], [ 1.0587832249221306E-5 ], [ -0.868 ], [ 0.92400715472043 ], ... ],
    	[ [ 1.125137486974472E-5 ], [ -1.164 ], [ -1.812 ], [ 1.3440012405362036 ], [ 8.799152141723409E-6 ], [ 1.1040063180797346 ], [ 1.1359950090055064 ], [ -0.496 ], ... ],
    	[ [ 8.279857916965964E-6 ], [ 4.990994493502134E-6 ], [ 1.5919910277497835 ], [ 0.05601257846011079 ], [ 1.6999993941567377 ], [ -0.96 ], [ 1.4020944068451077E-5 ], [ 1.4159996538038502 ], ... ],
    	[ [ -0.308 ], [ 1.3040036927589316 ], [ 6.664275884387251E-6 ], [ 2.59647112378724E-6 ], [ 1.006853802446386E-5 ], [ -0.504 ], [ 1.0934028399059621E-5 ], [ 7.241269467451073E-6 ], ... ],
    	[ [ 8.943400537489385E-7 ], [ -0.072 ], [ 3.952406043987244E-6 ], [ 1.8559866714482314 ], [ 0.30400637577909284 ], [ -0.276 ], [ -0.828 ], [ -0.3 ], ... ],
    	...
    ]
```



Code from [LearningTester.java:95](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L95) executed in 0.01 seconds: 
```java
    return TestUtil.compare(runs);
```

Returns: 

![Result](etc/test.13.png)



Code from [LearningTester.java:98](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L98) executed in 0.01 seconds: 
```java
    return TestUtil.compareTime(runs);
```

Returns: 

![Result](etc/test.14.png)



### Performance
Now we execute larger-scale runs to benchmark performance:

Code from [PerformanceTester.java:66](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/PerformanceTester.java#L66) executed in 0.25 seconds: 
```java
    test(component, inputPrototype);
```
Logging: 
```
    100 batches
    Input Dimensions:
    	[100, 100, 1]
    Performance:
    	Evaluation performance: 0.008905s +- 0.000968s [0.007900s - 0.010350s]
    	Learning performance: 0.030900s +- 0.003937s [0.024830s - 0.037271s]
    
```

### Function Plots
Code from [ActivationLayerTest.java:90](../../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/cudnn/ActivationLayerTest.java#L90) executed in 0.01 seconds: 
```java
    return ActivationLayerTestBase.plot("Value Plot", plotData, x -> new double[]{x[0], x[1]});
```

Returns: 

![Result](etc/test.15.png)



Code from [ActivationLayerTest.java:94](../../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/cudnn/ActivationLayerTest.java#L94) executed in 0.01 seconds: 
```java
    return ActivationLayerTestBase.plot("Derivative Plot", plotData, x -> new double[]{x[0], x[2]});
```

Returns: 

![Result](etc/test.16.png)



