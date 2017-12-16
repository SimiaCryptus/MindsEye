# ProductInputsLayer
## NNTest
### Batch Execution
Code from [BatchingTester.java:66](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/BatchingTester.java#L66) executed in 0.00 seconds: 
```java
    return test(reference, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (90#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (88#)}
```



Code from [SingleDerivativeTester.java:77](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/SingleDerivativeTester.java#L77) executed in 0.00 seconds: 
```java
    return test(component, inputPrototype);
```
Logging: 
```
    Inputs: [ -0.728, 1.516, -1.34 ],
    [ -0.96, 0.06, -0.308 ]
    Inputs Statistics: {meanExponent=0.05664512632462652, negative=2, min=-1.34, max=-1.34, mean=-0.18400000000000002, count=3.0, positive=1, stdDev=1.2277719657982096, zeros=0},
    {meanExponent=-0.5836756000254479, negative=2, min=-0.308, max=-0.308, mean=-0.4026666666666667, count=3.0, positive=1, stdDev=0.4217592783672801, zeros=0}
    Output: [ 0.69888, 0.09096, 0.41272000000000003 ]
    Outputs Statistics: {meanExponent=-0.5270304737008215, negative=0, min=0.41272000000000003, max=0.41272000000000003, mean=0.40085333333333334, count=3.0, positive=3, stdDev=0.24832410935889582, zeros=0}
    Feedback for input 0
    Inputs Values: [ -0.728, 1.516, -1.34 ]
    Value Statistics: {meanExponent=0.05664512632462652, negative=2, min=-1.34, max=-1.34, mean=-0.18400000000000002, count=3.0, positive=1, stdDev=1.2277719657982096, zeros=0}
    Implemented Feedback: [ [ -0.96, 0.0, 0.0 ], [ 0.0, 0.06, 0.0 ], [ 0.0, 0.0, -0.308 ] ]
    Implemented Statistics: {meanExponent=-0.58367560002
```
...[skipping 1144 bytes](etc/317.txt)...
```
    ed Statistics: {meanExponent=0.05664512632462652, negative=2, min=-1.34, max=-1.34, mean=-0.06133333333333334, count=9.0, positive=1, stdDev=0.7141415981597923, zeros=6}
    Measured Feedback: [ [ -0.7279999999998399, 0.0, 0.0 ], [ 0.0, 1.5160000000000173, 0.0 ], [ 0.0, 0.0, -1.34000000000023 ] ]
    Measured Statistics: {meanExponent=0.056645126324621174, negative=2, min=-1.34000000000023, max=-1.34000000000023, mean=-0.06133333333333916, count=9.0, positive=1, stdDev=0.7141415981598256, zeros=6}
    Feedback Error: [ [ 1.6009416015094757E-13, 0.0, 0.0 ], [ 0.0, 1.7319479184152442E-14, 0.0 ], [ 0.0, 0.0, -2.298161660974074E-13 ] ]
    Error Statistics: {meanExponent=-13.06523636879274, negative=1, min=-2.298161660974074E-13, max=-2.298161660974074E-13, mean=-5.82250297358971E-15, count=9.0, positive=2, stdDev=9.335741403960585E-14, zeros=6}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 3.6730e-14 +- 6.8242e-14 [0.0000e+00 - 2.2982e-13] (18#)
    relativeTol: 1.6278e-13 +- 2.2329e-13 [5.7122e-15 - 6.5642e-13] (6#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=3.6730e-14 +- 6.8242e-14 [0.0000e+00 - 2.2982e-13] (18#), relativeTol=1.6278e-13 +- 2.2329e-13 [5.7122e-15 - 6.5642e-13] (6#)}
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
      "id": "0548ccf7-1e57-44bb-ac30-25fa886a7905",
      "isFrozen": false,
      "name": "ProductInputsLayer/0548ccf7-1e57-44bb-ac30-25fa886a7905"
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
    [[ -1.428, 0.248, -1.416 ],
    [ 0.588, -0.6, -0.708 ]]
    --------------------
    Output: 
    [ -0.839664, -0.1488, 1.0025279999999999 ]
    --------------------
    Derivative: 
    [ 0.588, -0.6, -0.708 ],
    [ -1.428, 0.248, -1.416 ]
```



### Input Learning
In this test, we use a network to learn this target input, given it's pre-evaluated output:

Code from [LearningTester.java:127](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L127) executed in 0.00 seconds: 
```java
    return Arrays.stream(input_target).map(x -> x.prettyPrint()).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [ 1.592, -1.988, -1.784 ]
    [ 1.224, 0.212, -0.416 ]
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
    F(0.0) = LineSearchPoint{point=PointSample{avg=7.954554136234666}, derivative=-299.08822604153954}
    F(1.0E-10) = LineSearchPoint{point=PointSample{avg=7.954554155257157}, derivative=-299.0882269284476}, delta = 1.9022491137832276E-8
    F(7.692307692307693E-12) = LineSearchPoint{point=PointSample{avg=7.954554137697936}, derivative=-299.0882261097633}, delta = 1.4632695055638578E-9
    F(5.91715976331361E-13) = LineSearchPoint{point=PointSample{avg=7.954554136347223}, derivative=-299.08822604678744}, delta = 1.1255707477175747E-10
    0.0 ~= 5.91715976331361E-13
    Converged to right
    Iteration 1 failed, aborting. Error: 7.954554136234666 Total: 239706497183803.5000; Orientation: 0.0001; Line Search: 0.0005
    
```

Returns: 

```
    7.954554136234666
```



This training run resulted in the following regressed input:

Code from [LearningTester.java:144](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L144) executed in 0.00 seconds: 
```java
    return Arrays.stream(input_gd).map(x -> x.prettyPrint()).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [ -1.784000000000801, -1.9880000000100608, 1.5919999999983105 ]
    [ -0.416, 1.224, 0.212 ]
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
    th(0)=7.954554136234666;dx=-299.08822604153954
    Armijo: th(2.154434690031884)=742205.8710672315; dx=-3.364190619968745E7 delta=-742197.9165130953
    Armijo: th(1.077217345015942)=56790.92828805538; dx=-2573929.82941255 delta=-56782.97373391915
    Armijo: th(0.3590724483386473)=1452.101195480914; dx=-65695.12986310868 delta=-1444.1466413446792
    Armijo: th(0.08976811208466183)=55.67956103388286; dx=-2471.562601120185 delta=-47.7250068976482
    Armijo: th(0.017953622416932366)=12.22560280460637; dx=-496.55532990682036 delta=-4.271048668371703
    Armijo: th(0.002992270402822061)=8.545713552410854; dx=-326.6063595867149 delta=-0.591159416176188
    Armijo: th(4.2746720040315154E-4)=8.036310984359519; dx=-302.8991772423934 delta=-0.08175684812485251
    Armijo: th(5.343340005039394E-5)=7.964725392245342; dx=-299.5624385366873 delta=-0.010171256010675478
    Armijo: th(5.9370444500437714E-6)=7.955683595149613; dx=-299.1408859635663 delta=-0.00112945891494
```
...[skipping 12 bytes](etc/318.txt)...
```
    : th(5.937044450043771E-7)=7.954667074471118; dx=-299.0934916923451 delta=-1.1293823645175394E-4
    Armijo: th(5.397313136403428E-8)=7.9545644032767795; dx=-299.0887047339325 delta=-1.026704211337659E-5
    Armijo: th(4.4977609470028565E-9)=7.954554991820971; dx=-299.0882659325483 delta=-8.555863049153345E-7
    Armijo: th(3.4598161130791205E-10)=7.954554202048992; dx=-299.0882291100785 delta=-6.581432554497724E-8
    Armijo: th(2.4712972236279432E-11)=7.954554140935691; dx=-299.088226260721 delta=-4.701024458597658E-9
    Armijo: th(1.6475314824186289E-12)=7.954554136548066; dx=-299.0882260561516 delta=-3.133999726401271E-10
    Armijo: th(1.029707176511643E-13)=7.954554136254255; dx=-299.08822604245285 delta=-1.9588775046486262E-11
    Armijo: th(6.057101038303783E-15)=7.954554136235818; dx=-299.0882260415932 delta=-1.1519674103510624E-12
    MIN ALPHA: th(0)=7.954554136234666;th'(0)=-299.08822604153954;
    Iteration 1 failed, aborting. Error: 7.954554136234666 Total: 239706503598666.5000; Orientation: 0.0001; Line Search: 0.0041
    
```

Returns: 

```
    7.954554136234666
```



This training run resulted in the following regressed input:

Code from [LearningTester.java:154](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L154) executed in 0.00 seconds: 
```java
    return Arrays.stream(input_lbgfs).map(x -> x.prettyPrint()).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [ -1.784, -1.988, 1.592 ]
    [ -0.416, 1.224, 0.212 ]
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
    	[3]
    Performance:
    	Evaluation performance: 0.000174s +- 0.000093s [0.000080s - 0.000346s]
    	Learning performance: 0.000078s +- 0.000052s [0.000041s - 0.000180s]
    
```

