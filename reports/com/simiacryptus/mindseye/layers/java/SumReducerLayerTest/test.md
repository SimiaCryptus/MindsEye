# SumReducerLayer
## SumReducerLayerTest
### Batch Execution
Code from [BatchingTester.java:66](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/BatchingTester.java#L66) executed in 0.00 seconds: 
```java
    return test(reference, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (40#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (40#)}
```



Code from [SingleDerivativeTester.java:77](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/SingleDerivativeTester.java#L77) executed in 0.00 seconds: 
```java
    return test(component, inputPrototype);
```
Logging: 
```
    Inputs: [ 1.844, -0.296, 1.672 ]
    Inputs Statistics: {meanExponent=-0.013237033040151122, negative=1, min=1.672, max=1.672, mean=1.0733333333333333, count=3.0, positive=2, stdDev=0.9708076820645558, zeros=0}
    Output: [ 3.2199999999999998 ]
    Outputs Statistics: {meanExponent=0.5078558716958309, negative=0, min=3.2199999999999998, max=3.2199999999999998, mean=3.2199999999999998, count=1.0, positive=1, stdDev=0.0, zeros=0}
    Feedback for input 0
    Inputs Values: [ 1.844, -0.296, 1.672 ]
    Value Statistics: {meanExponent=-0.013237033040151122, negative=1, min=1.672, max=1.672, mean=1.0733333333333333, count=3.0, positive=2, stdDev=0.9708076820645558, zeros=0}
    Implemented Feedback: [ [ 1.0 ], [ 1.0 ], [ 1.0 ] ]
    Implemented Statistics: {meanExponent=0.0, negative=0, min=1.0, max=1.0, mean=1.0, count=3.0, positive=3, stdDev=0.0, zeros=0}
    Measured Feedback: [ [ 1.0000000000021103 ], [ 1.0000000000021103 ], [ 1.0000000000021103 ] ]
    Measured Statistics: {meanExponent=9.16496824211277E-13, negative=0, min=1.0000000000021103, max=1.0000000000021103, mean=1.0000000000021103, count=3.0, positive=3, stdDev=0.0, zeros=0}
    Feedback Error: [ [ 2.1103119252074976E-12 ], [ 2.1103119252074976E-12 ], [ 2.1103119252074976E-12 ] ]
    Error Statistics: {meanExponent=-11.675653346889904, negative=0, min=2.1103119252074976E-12, max=2.1103119252074976E-12, mean=2.1103119252074976E-12, count=3.0, positive=3, stdDev=0.0, zeros=0}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 2.1103e-12 +- 0.0000e+00 [2.1103e-12 - 2.1103e-12] (3#)
    relativeTol: 1.0552e-12 +- 0.0000e+00 [1.0552e-12 - 1.0552e-12] (3#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=2.1103e-12 +- 0.0000e+00 [2.1103e-12 - 2.1103e-12] (3#), relativeTol=1.0552e-12 +- 0.0000e+00 [1.0552e-12 - 1.0552e-12] (3#)}
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
      "class": "com.simiacryptus.mindseye.layers.java.SumReducerLayer",
      "id": "91cb5ff5-c527-42ff-88bd-de361ffba9f0",
      "isFrozen": false,
      "name": "SumReducerLayer/91cb5ff5-c527-42ff-88bd-de361ffba9f0"
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
    [[ 0.296, 1.012, 0.812 ]]
    --------------------
    Output: 
    [ 2.12 ]
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
    [
    	[ [ -0.72 ], [ -0.98 ], [ 0.044 ], [ 0.256 ], [ -0.216 ], [ 1.032 ], [ -0.524 ], [ 1.804 ], ... ],
    	[ [ 0.484 ], [ -1.892 ], [ -1.296 ], [ 1.804 ], [ -0.888 ], [ -1.844 ], [ 0.32 ], [ 1.132 ], ... ],
    	[ [ -0.932 ], [ -0.152 ], [ 0.6 ], [ 1.484 ], [ -1.084 ], [ -1.472 ], [ 1.564 ], [ 1.144 ], ... ],
    	[ [ -0.236 ], [ -1.092 ], [ -1.084 ], [ 1.844 ], [ 1.832 ], [ -2.0 ], [ -0.22 ], [ -0.096 ], ... ],
    	[ [ -0.616 ], [ -0.572 ], [ -1.0 ], [ -0.776 ], [ -0.496 ], [ -0.384 ], [ -1.484 ], [ -1.588 ], ... ],
    	[ [ -1.52 ], [ 0.032 ], [ -0.276 ], [ -0.172 ], [ 0.744 ], [ 0.0 ], [ 1.216 ], [ 1.46 ], ... ],
    	[ [ 1.448 ], [ 0.148 ], [ 1.868 ], [ 0.844 ], [ -0.252 ], [ -0.964 ], [ 0.66 ], [ 1.372 ], ... ],
    	[ [ 0.948 ], [ 1.824 ], [ 1.452 ], [ -0.84 ], [ 0.344 ], [ 1.224 ], [ 1.508 ], [ 1.928 ], ... ],
    	...
    ]
```



First, we use a conjugate gradient descent method, which converges the fastest for purely linear functions.

Code from [LearningTester.java:225](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L225) executed in 0.95 seconds: 
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
    Zero gradient: 6.821210263296962E-11
    Constructing line search parameters: GD
    F(0.0) = LineSearchPoint{point=PointSample{avg=1.1632227364026952E-25}, derivative=-4.652890945610781E-21}
    F(1.0E-10) = LineSearchPoint{point=PointSample{avg=1.1632227364026952E-25}, derivative=-4.652890945610781E-21}, delta = 0.0
    F(7.000000000000001E-10) = LineSearchPoint{point=PointSample{avg=1.1632227364026952E-25}, derivative=-4.652890945610781E-21}, delta = 0.0
    F(4.900000000000001E-9) = LineSearchPoint{point=PointSample{avg=1.1632227364026952E-25}, derivative=-4.652890945610781E-21}, delta = 0.0
    F(3.430000000000001E-8) = LineSearchPoint{point=PointSample{avg=1.1632227364026952E-25}, derivative=-4.652890945610781E-21}, delta = 0.0
    F(2.4010000000000004E-7) = LineSearchPoint{point=PointSample{avg=1.1632227364026952E-25}, derivative=-4.652890945610781E-21}, delta = 0.0
    F(1.6807000000000003E-6) = LineSearchPoint{point=PointSample{avg=1.1632227364026952E-25}, derivative=-4.652890945610781E-21}, delta = 0.0
    F(1.176490000000000
```
...[skipping 141876 bytes](etc/351.txt)...
```
    906781E-5) = LineSearchPoint{point=PointSample{avg=8.077935669463161E-28}, derivative=-3.2311742677852644E-23}, delta = 0.0
    Left bracket at 1.2386952115906781E-5
    F(1.4451444135224577E-5) = LineSearchPoint{point=PointSample{avg=8.077935669463161E-28}, derivative=-3.2311742677852644E-23}, delta = 0.0
    Left bracket at 1.4451444135224577E-5
    F(1.5483690144883476E-5) = LineSearchPoint{point=PointSample{avg=8.077935669463161E-28}, derivative=3.2311742677852644E-23}, delta = 0.0
    Right bracket at 1.5483690144883476E-5
    F(1.4967567140054025E-5) = LineSearchPoint{point=PointSample{avg=8.077935669463161E-28}, derivative=-3.2311742677852644E-23}, delta = 0.0
    Left bracket at 1.4967567140054025E-5
    F(1.522562864246875E-5) = LineSearchPoint{point=PointSample{avg=8.077935669463161E-28}, derivative=-3.2311742677852644E-23}, delta = 0.0
    Left bracket at 1.522562864246875E-5
    Converged to left
    Iteration 48 failed, aborting. Error: 8.077935669463161E-28 Total: 239731533170849.4700; Orientation: 0.0003; Line Search: 0.0122
    
```

Returns: 

```
    8.077935669463161E-28
```



Training Converged

Next, we run the same optimization using L-BFGS, which is nearly ideal for purely second-order or quadratic functions.

Code from [LearningTester.java:249](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L249) executed in 0.05 seconds: 
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
    th(0)=1.1632227364026952E-25;dx=-4.652890945610781E-21
    Armijo: th(2.154434690031884)=2.1593271079454686E-16; dx=2.0047058187346475E-16 delta=-2.159327106782246E-16
    Armijo: th(1.077217345015942)=5.398651892670006E-17; dx=1.0023839286402945E-16 delta=-5.398651881037779E-17
    Armijo: th(0.3590724483386473)=5.9954396541568035E-18; dx=3.34042673212762E-17 delta=-5.99543953783453E-18
    Armijo: th(0.08976811208466183)=3.7660504483652487E-19; dx=8.372101774802331E-18 delta=-3.7660492851425123E-19
    Armijo: th(0.017953622416932366)=1.4998691706197952E-20; dx=1.6707755903864045E-18 delta=-1.4998575383924312E-20
    Armijo: th(0.002992270402822061)=4.352036309537295E-22; dx=2.846018295065261E-19 delta=-4.350873086800892E-22
    Armijo: th(4.2746720040315154E-4)=6.54312789226516E-24; dx=3.4896682092080855E-20 delta=-6.426805618624891E-24
    New Minimum: 1.1632227364026952E-25 > 5.169878828456423E-26
    END: th(5.343340005039394E-5)=5.169878828456423E-26;
```
...[skipping 3289 bytes](etc/352.txt)...
```
    .2924697071141057E-26; dx=-5.169878828456423E-22 delta=0.0
    Armijo: th(5.937044450043771E-7)=1.2924697071141057E-26; dx=-5.169878828456423E-22 delta=0.0
    Armijo: th(5.397313136403428E-8)=1.2924697071141057E-26; dx=-5.169878828456423E-22 delta=0.0
    Armijo: th(4.4977609470028565E-9)=1.2924697071141057E-26; dx=-5.169878828456423E-22 delta=0.0
    Armijo: th(3.4598161130791205E-10)=1.2924697071141057E-26; dx=-5.169878828456423E-22 delta=0.0
    Armijo: th(2.4712972236279432E-11)=1.2924697071141057E-26; dx=-5.169878828456423E-22 delta=0.0
    Armijo: th(1.6475314824186289E-12)=1.2924697071141057E-26; dx=-5.169878828456423E-22 delta=0.0
    Armijo: th(1.029707176511643E-13)=1.2924697071141057E-26; dx=-5.169878828456423E-22 delta=0.0
    Armijo: th(6.057101038303783E-15)=1.2924697071141057E-26; dx=-5.169878828456423E-22 delta=0.0
    MIN ALPHA: th(0)=1.2924697071141057E-26;th'(0)=-5.169878828456423E-22;
    Iteration 4 failed, aborting. Error: 1.2924697071141057E-26 Total: 239731582334181.4400; Orientation: 0.0005; Line Search: 0.0171
    
```

Returns: 

```
    1.2924697071141057E-26
```



Training Converged

Code from [LearningTester.java:95](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L95) executed in 0.00 seconds: 
```java
    return TestUtil.compare(runs);
```

Returns: 

![Result](etc/test.251.png)



Code from [LearningTester.java:98](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L98) executed in 0.00 seconds: 
```java
    return TestUtil.compareTime(runs);
```

Returns: 

![Result](etc/test.252.png)



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
    	[100, 100, 1]
    Performance:
    	Evaluation performance: 0.000226s +- 0.000046s [0.000157s - 0.000298s]
    	Learning performance: 0.000040s +- 0.000008s [0.000032s - 0.000052s]
    
```

