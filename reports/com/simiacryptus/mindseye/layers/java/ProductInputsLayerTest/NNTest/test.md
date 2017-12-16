# ProductInputsLayer
## NNTest
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
      "id": "36355148-3c2f-4ab7-a55f-de8b1ca3d8f7",
      "isFrozen": false,
      "name": "ProductInputsLayer/36355148-3c2f-4ab7-a55f-de8b1ca3d8f7"
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
    [[ -0.224, -0.608, -1.116 ],
    [ -0.276, 1.708, -0.216 ]]
    --------------------
    Output: 
    [ 0.061824000000000004, -1.038464, 0.24105600000000002 ]
    --------------------
    Derivative: 
    [ -0.276, 1.708, -0.216 ],
    [ -0.224, -0.608, -1.116 ]
```



### Batch Execution
Code from [BatchingTester.java:66](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/BatchingTester.java#L66) executed in 0.00 seconds: 
```java
    return test(reference, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (90#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (90#)}
```



Code from [SingleDerivativeTester.java:77](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/SingleDerivativeTester.java#L77) executed in 0.00 seconds: 
```java
    return test(component, inputPrototype);
```
Logging: 
```
    Inputs: [ -1.756, -1.32, 0.52 ],
    [ 1.732, -0.084, -0.144 ]
    Inputs Statistics: {meanExponent=0.027033928803577596, negative=2, min=0.52, max=0.52, mean=-0.852, count=3.0, positive=1, stdDev=0.9863440914136744, zeros=0},
    {meanExponent=-0.5596034447205137, negative=2, min=-0.144, max=-0.144, mean=0.5013333333333333, count=3.0, positive=1, stdDev=0.8705574204050083, zeros=0}
    Output: [ -3.041392, 0.11088, -0.07488 ]
    Outputs Statistics: {meanExponent=-0.532569515916936, negative=2, min=-0.07488, max=-0.07488, mean=-1.0017973333333334, count=3.0, positive=1, stdDev=1.4442037016112221, zeros=0}
    Feedback for input 0
    Inputs Values: [ -1.756, -1.32, 0.52 ]
    Value Statistics: {meanExponent=0.027033928803577596, negative=2, min=0.52, max=0.52, mean=-0.852, count=3.0, positive=1, stdDev=0.9863440914136744, zeros=0}
    Implemented Feedback: [ [ 1.732, 0.0, 0.0 ], [ 0.0, -0.084, 0.0 ], [ 0.0, 0.0, -0.144 ] ]
    Implemented Statistics: {meanExponent=-0.5596034447205137, negative=2, min=-0.144, max=-0.144, mean=0.167111111111
```
...[skipping 1087 bytes](etc/376.txt)...
```
    tatistics: {meanExponent=0.027033928803577596, negative=2, min=0.52, max=0.52, mean=-0.28400000000000003, count=9.0, positive=1, stdDev=0.6968526067652726, zeros=6}
    Measured Feedback: [ [ -1.7559999999994247, 0.0, 0.0 ], [ 0.0, -1.3199999999999323, 0.0 ], [ 0.0, 0.0, 0.5199999999999649 ] ]
    Measured Statistics: {meanExponent=0.02703392880351298, negative=2, min=0.5199999999999649, max=0.5199999999999649, mean=-0.2839999999999325, count=9.0, positive=1, stdDev=0.6968526067651218, zeros=6}
    Feedback Error: [ [ 5.753175713607561E-13, 0.0, 0.0 ], [ 0.0, 6.772360450213455E-14, 0.0 ], [ 0.0, 0.0, -3.5083047578154947E-14 ] ]
    Error Statistics: {meanExponent=-12.954751661431345, negative=1, min=-3.5083047578154947E-14, max=-3.5083047578154947E-14, mean=6.755090314274841E-14, count=9.0, positive=2, stdDev=1.8127311932610984E-13, zeros=6}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 1.1381e-13 +- 3.0843e-13 [0.0000e+00 - 1.2654e-12] (18#)
    relativeTol: 1.9299e-13 +- 1.7610e-13 [2.5653e-14 - 4.9126e-13] (6#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=1.1381e-13 +- 3.0843e-13 [0.0000e+00 - 1.2654e-12] (18#), relativeTol=1.9299e-13 +- 1.7610e-13 [2.5653e-14 - 4.9126e-13] (6#)}
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
    	[3]
    Performance:
    	Evaluation performance: 0.000138s +- 0.000013s [0.000117s - 0.000156s]
    	Learning performance: 0.000050s +- 0.000011s [0.000039s - 0.000070s]
    
```

### Input Learning
In this test, we use a network to learn this target input, given it's pre-evaluated output:

Code from [LearningTester.java:127](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L127) executed in 0.00 seconds: 
```java
    return Arrays.stream(input_target).map(x -> x.prettyPrint()).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [ -1.924, 1.104, -1.364 ]
    [ 0.56, 1.352, -1.088 ]
```



First, we use a conjugate gradient descent method, which converges the fastest for purely linear functions.

Code from [LearningTester.java:300](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L300) executed in 0.00 seconds: 
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
    F(0.0) = LineSearchPoint{point=PointSample{avg=7.682195560618666}, derivative=-412.2312947854109}
    F(1.0E-10) = LineSearchPoint{point=PointSample{avg=7.682195585521243}, derivative=-412.23129613310647}, delta = 2.490257688947395E-8
    F(7.692307692307693E-12) = LineSearchPoint{point=PointSample{avg=7.68219556253425}, derivative=-412.2312948890798}, delta = 1.9155841357587633E-9
    F(5.91715976331361E-13) = LineSearchPoint{point=PointSample{avg=7.6821955607660195}, derivative=-412.23129479338553}, delta = 1.4735324072034928E-10
    0.0 ~= 5.91715976331361E-13
    Converged to right
    Iteration 1 failed, aborting. Error: 7.682195560618666 Total: 249840369942217.6200; Orientation: 0.0000; Line Search: 0.0005
    
```

Returns: 

```
    7.682195560618666
```



This training run resulted in the following regressed input:

Code from [LearningTester.java:144](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L144) executed in 0.00 seconds: 
```java
    return Arrays.stream(input_gd).map(x -> x.prettyPrint()).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [ -1.9240000000120137, -1.3640000000000714, 1.103999999999963 ]
    [ 1.352, -1.088, 0.56 ]
```



Next, we run the same optimization using L-BFGS, which is nearly ideal for purely second-order or quadratic functions.

Code from [LearningTester.java:324](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L324) executed in 0.00 seconds: 
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
    th(0)=7.682195560618666;dx=-412.2312947854109
    Armijo: th(2.154434690031884)=1451049.3621929383; dx=-7.856182362768319E7 delta=-1451041.6799973778
    Armijo: th(1.077217345015942)=107264.81643595536; dx=-5807455.163348858 delta=-107257.13424039475
    Armijo: th(0.3590724483386473)=2464.2980890628; dx=-133414.52046903805 delta=-2456.6158935021813
    Armijo: th(0.08976811208466183)=76.22453819423144; dx=-4122.683249708489 delta=-68.54234263361278
    Armijo: th(0.017953622416932366)=13.362401725554932; dx=-719.6634280305481 delta=-5.680206164936266
    Armijo: th(0.002992270402822061)=8.457904370125943; dx=-454.21233347685154 delta=-0.775708809507277
    Armijo: th(4.2746720040315154E-4)=7.789259345337471; dx=-418.0254631572205 delta=-0.10706378471880473
    Armijo: th(5.343340005039394E-5)=7.695511414843728; dx=-412.951931970737 delta=-0.013315854225061763
    Armijo: th(5.9370444500437714E-6)=7.683674155528933; dx=-412.311314450719 delta=-0.00147859491
```
...[skipping 23 bytes](etc/377.txt)...
```
    37044450043771E-7)=7.682343409488252; dx=-412.23929617688964 delta=-1.4784886958540966E-4
    Armijo: th(5.397313136403428E-8)=7.682209001327461; dx=-412.23202217935585 delta=-1.3440708794831835E-5
    Armijo: th(4.4977609470028565E-9)=7.682196680676987; dx=-412.23135540153265 delta=-1.1200583207582326E-6
    Armijo: th(3.4598161130791205E-10)=7.682195646776996; dx=-412.23129944818925 delta=-8.615832935987555E-8
    Armijo: th(2.4712972236279432E-11)=7.682195566772834; dx=-412.23129511846656 delta=-6.154167841998515E-9
    Armijo: th(1.6475314824186289E-12)=7.682195561028943; dx=-412.2312948076146 delta=-4.1027714559049855E-10
    Armijo: th(1.029707176511643E-13)=7.682195560644307; dx=-412.23129478679857 delta=-2.5640822798322915E-11
    Armijo: th(6.057101038303783E-15)=7.682195560620175; dx=-412.23129478549254 delta=-1.5090151350705128E-12
    MIN ALPHA: th(0)=7.682195560618666;th'(0)=-412.2312947854109;
    Iteration 1 failed, aborting. Error: 7.682195560618666 Total: 249840373633257.6200; Orientation: 0.0001; Line Search: 0.0017
    
```

Returns: 

```
    7.682195560618666
```



This training run resulted in the following regressed input:

Code from [LearningTester.java:154](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L154) executed in 0.00 seconds: 
```java
    return Arrays.stream(input_lbgfs).map(x -> x.prettyPrint()).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [ -1.924, -1.364, 1.104 ]
    [ 1.352, -1.088, 0.56 ]
```



Code from [LearningTester.java:96](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L96) executed in 0.00 seconds: 
```java
    return TestUtil.compare(runs);
```

Code from [LearningTester.java:99](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L99) executed in 0.00 seconds: 
```java
    return TestUtil.compareTime(runs);
```

