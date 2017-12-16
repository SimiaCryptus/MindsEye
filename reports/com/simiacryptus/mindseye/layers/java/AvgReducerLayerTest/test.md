# AvgReducerLayer
## AvgReducerLayerTest
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
    Inputs: [ -1.532, 1.3, -0.676 ]
    Inputs Statistics: {meanExponent=0.04304960451501929, negative=2, min=-0.676, max=-0.676, mean=-0.3026666666666667, count=3.0, positive=1, stdDev=1.1859143682782871, zeros=0}
    Output: [ -0.30266666666666675 ]
    Outputs Statistics: {meanExponent=-0.5190354061985772, negative=1, min=-0.30266666666666675, max=-0.30266666666666675, mean=-0.30266666666666675, count=1.0, positive=0, stdDev=0.0, zeros=0}
    Feedback for input 0
    Inputs Values: [ -1.532, 1.3, -0.676 ]
    Value Statistics: {meanExponent=0.04304960451501929, negative=2, min=-0.676, max=-0.676, mean=-0.3026666666666667, count=3.0, positive=1, stdDev=1.1859143682782871, zeros=0}
    Implemented Feedback: [ [ 0.3333333333333333 ], [ 0.3333333333333333 ], [ 0.3333333333333333 ] ]
    Implemented Statistics: {meanExponent=-0.47712125471966244, negative=0, min=0.3333333333333333, max=0.3333333333333333, mean=0.3333333333333333, count=3.0, positive=3, stdDev=0.0, zeros=0}
    Measured Feedback: [ [ 0.3333333333332966 ], [ 0.3333333333332966 ], [ 0.3333333333332966 ] ]
    Measured Statistics: {meanExponent=-0.4771212547197103, negative=0, min=0.3333333333332966, max=0.3333333333332966, mean=0.3333333333332966, count=3.0, positive=3, stdDev=0.0, zeros=0}
    Feedback Error: [ [ -3.6692870963861424E-14 ], [ -3.6692870963861424E-14 ], [ -3.6692870963861424E-14 ] ]
    Error Statistics: {meanExponent=-13.435418306369344, negative=3, min=-3.6692870963861424E-14, max=-3.6692870963861424E-14, mean=-3.6692870963861424E-14, count=3.0, positive=0, stdDev=0.0, zeros=0}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 3.6693e-14 +- 0.0000e+00 [3.6693e-14 - 3.6693e-14] (3#)
    relativeTol: 5.5039e-14 +- 0.0000e+00 [5.5039e-14 - 5.5039e-14] (3#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=3.6693e-14 +- 0.0000e+00 [3.6693e-14 - 3.6693e-14] (3#), relativeTol=5.5039e-14 +- 0.0000e+00 [5.5039e-14 - 5.5039e-14] (3#)}
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
      "class": "com.simiacryptus.mindseye.layers.java.AvgReducerLayer",
      "id": "5d3ea985-5e22-4c1a-b3b9-c25610eaff68",
      "isFrozen": false,
      "name": "AvgReducerLayer/5d3ea985-5e22-4c1a-b3b9-c25610eaff68"
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
    [[ -1.992, 1.796, -0.66 ]]
    --------------------
    Output: 
    [ -0.2853333333333333 ]
    --------------------
    Derivative: 
    [ 0.3333333333333333, 0.3333333333333333, 0.3333333333333333 ]
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
    	[ [ -1.944 ], [ 1.828 ], [ -1.224 ], [ -0.86 ], [ 0.08 ], [ -0.936 ], [ 1.096 ], [ 0.476 ], ... ],
    	[ [ 0.712 ], [ 0.088 ], [ 1.572 ], [ 0.948 ], [ -0.4 ], [ -1.716 ], [ -0.064 ], [ -0.196 ], ... ],
    	[ [ -0.692 ], [ 1.916 ], [ 0.012 ], [ 1.548 ], [ -0.5 ], [ 1.272 ], [ -1.308 ], [ -0.896 ], ... ],
    	[ [ -1.912 ], [ 0.624 ], [ -0.392 ], [ -1.94 ], [ -0.264 ], [ 0.064 ], [ -0.62 ], [ 1.5 ], ... ],
    	[ [ 1.704 ], [ 1.604 ], [ -1.952 ], [ 1.236 ], [ -0.22 ], [ -1.988 ], [ -0.852 ], [ -0.116 ], ... ],
    	[ [ -0.432 ], [ 0.772 ], [ -1.572 ], [ -0.376 ], [ 1.884 ], [ 0.328 ], [ 1.052 ], [ 1.32 ], ... ],
    	[ [ 1.624 ], [ -0.28 ], [ 1.172 ], [ -0.904 ], [ -0.968 ], [ 1.168 ], [ -1.02 ], [ -0.812 ], ... ],
    	[ [ -0.72 ], [ -1.94 ], [ 0.664 ], [ 0.944 ], [ -0.104 ], [ -1.996 ], [ 1.544 ], [ 1.188 ], ... ],
    	...
    ]
```



First, we use a conjugate gradient descent method, which converges the fastest for purely linear functions.

Code from [LearningTester.java:225](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L225) executed in 0.03 seconds: 
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
    Zero gradient: 2.0816681711721684E-19
    Constructing line search parameters: GD
    F(0.0) = LineSearchPoint{point=PointSample{avg=1.0833355937178202E-34}, derivative=-4.33334237487128E-38}
    F(1.0E-10) = LineSearchPoint{point=PointSample{avg=1.0833355937178202E-34}, derivative=-4.33334237487128E-38}, delta = 0.0
    F(7.000000000000001E-10) = LineSearchPoint{point=PointSample{avg=1.0833355937178202E-34}, derivative=-4.33334237487128E-38}, delta = 0.0
    F(4.900000000000001E-9) = LineSearchPoint{point=PointSample{avg=1.0833355937178202E-34}, derivative=-4.33334237487128E-38}, delta = 0.0
    F(3.430000000000001E-8) = LineSearchPoint{point=PointSample{avg=1.0833355937178202E-34}, derivative=-4.33334237487128E-38}, delta = 0.0
    F(2.4010000000000004E-7) = LineSearchPoint{point=PointSample{avg=1.0833355937178202E-34}, derivative=-4.33334237487128E-38}, delta = 0.0
    F(1.6807000000000003E-6) = LineSearchPoint{point=PointSample{avg=1.0833355937178202E-34}, derivative=-4.33334237487128E-38}, delta = 0.0
    F(1.1764900000000001E-5) 
```
...[skipping 2078 bytes](etc/192.txt)...
```
    ght bracket at 0.01081350562578125
    F(0.005406752812890625) = LineSearchPoint{point=PointSample{avg=1.0833355937178202E-34}, derivative=-4.33334237487128E-38}, delta = 0.0
    Right bracket at 0.005406752812890625
    F(0.0027033764064453127) = LineSearchPoint{point=PointSample{avg=1.0833355937178202E-34}, derivative=-4.33334237487128E-38}, delta = 0.0
    Right bracket at 0.0027033764064453127
    F(0.0013516882032226563) = LineSearchPoint{point=PointSample{avg=1.0833355937178202E-34}, derivative=-4.33334237487128E-38}, delta = 0.0
    Right bracket at 0.0013516882032226563
    F(6.758441016113282E-4) = LineSearchPoint{point=PointSample{avg=1.0833355937178202E-34}, derivative=-4.33334237487128E-38}, delta = 0.0
    Right bracket at 6.758441016113282E-4
    F(3.379220508056641E-4) = LineSearchPoint{point=PointSample{avg=1.0833355937178202E-34}, derivative=-4.33334237487128E-38}, delta = 0.0
    Loops = 12
    Iteration 1 failed, aborting. Error: 1.0833355937178202E-34 Total: 239634609689924.4000; Orientation: 0.0004; Line Search: 0.0262
    
```

Returns: 

```
    1.0833355937178202E-34
```



This training run resulted in the following regressed input:

Code from [LearningTester.java:144](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L144) executed in 0.00 seconds: 
```java
    return Arrays.stream(input_gd).map(x -> x.prettyPrint()).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [
    	[ [ -1.584 ], [ 0.572 ], [ 0.408 ], [ 0.332 ], [ -0.388 ], [ -1.376 ], [ -1.164 ], [ 0.48 ], ... ],
    	[ [ 1.328 ], [ 0.792 ], [ 1.052 ], [ 0.776 ], [ 0.032 ], [ 0.448 ], [ -0.912 ], [ -0.152 ], ... ],
    	[ [ 1.42 ], [ -1.592 ], [ -0.3 ], [ 1.92 ], [ -0.736 ], [ 1.624 ], [ -0.948 ], [ 1.816 ], ... ],
    	[ [ 1.184 ], [ 0.7 ], [ 0.452 ], [ 1.78 ], [ -0.632 ], [ -0.556 ], [ -0.456 ], [ 0.208 ], ... ],
    	[ [ -0.192 ], [ 0.572 ], [ 0.204 ], [ -1.044 ], [ 0.272 ], [ 1.068 ], [ 1.868 ], [ -0.864 ], ... ],
    	[ [ -0.996 ], [ -1.852 ], [ -0.972 ], [ -0.604 ], [ 0.092 ], [ -1.44 ], [ 1.268 ], [ -1.64 ], ... ],
    	[ [ 1.192 ], [ -1.804 ], [ -0.248 ], [ -0.032 ], [ 0.316 ], [ -0.652 ], [ -0.848 ], [ 0.592 ], ... ],
    	[ [ -0.492 ], [ 0.12 ], [ -0.64 ], [ 0.504 ], [ 1.504 ], [ -1.588 ], [ -0.292 ], [ -1.292 ], ... ],
    	...
    ]
```



Next, we run the same optimization using L-BFGS, which is nearly ideal for purely second-order or quadratic functions.

Code from [LearningTester.java:249](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L249) executed in 0.02 seconds: 
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
    th(0)=1.0833355937178202E-34;dx=-4.33334237487128E-38
    Armijo: th(2.154434690031884)=1.0833355937178202E-34; dx=-4.33334237487128E-38 delta=0.0
    Armijo: th(1.077217345015942)=1.0833355937178202E-34; dx=-4.33334237487128E-38 delta=0.0
    Armijo: th(0.3590724483386473)=1.0833355937178202E-34; dx=-4.33334237487128E-38 delta=0.0
    Armijo: th(0.08976811208466183)=1.0833355937178202E-34; dx=-4.33334237487128E-38 delta=0.0
    Armijo: th(0.017953622416932366)=1.0833355937178202E-34; dx=-4.33334237487128E-38 delta=0.0
    Armijo: th(0.002992270402822061)=1.0833355937178202E-34; dx=-4.33334237487128E-38 delta=0.0
    Armijo: th(4.2746720040315154E-4)=1.0833355937178202E-34; dx=-4.33334237487128E-38 delta=0.0
    Armijo: th(5.343340005039394E-5)=1.0833355937178202E-34; dx=-4.33334237487128E-38 delta=0.0
    Armijo: th(5.9370444500437714E-6)=1.0833355937178202E-34; dx=-4.33334237487128E-38 delta=0.0
    Armijo: th(5.937044450043771E-7)=1.0833355937178202E-34; dx=
```
...[skipping 85 bytes](etc/193.txt)...
```
    78202E-34; dx=-4.33334237487128E-38 delta=0.0
    Armijo: th(3.238387881842057E-7)=1.0833355937178202E-34; dx=-4.33334237487128E-38 delta=0.0
    WOLFE (weak): th(1.8890595977412E-7)=1.0833355937178202E-34; dx=-4.33334237487128E-38 delta=0.0
    Armijo: th(2.563723739791628E-7)=1.0833355937178202E-34; dx=-4.33334237487128E-38 delta=0.0
    WOLFE (weak): th(2.2263916687664141E-7)=1.0833355937178202E-34; dx=-4.33334237487128E-38 delta=0.0
    WOLFE (weak): th(2.395057704279021E-7)=1.0833355937178202E-34; dx=-4.33334237487128E-38 delta=0.0
    Armijo: th(2.4793907220353244E-7)=1.0833355937178202E-34; dx=-4.33334237487128E-38 delta=0.0
    WOLFE (weak): th(2.4372242131571725E-7)=1.0833355937178202E-34; dx=-4.33334237487128E-38 delta=0.0
    WOLFE (weak): th(2.4583074675962484E-7)=1.0833355937178202E-34; dx=-4.33334237487128E-38 delta=0.0
    mu /= nu: th(0)=1.0833355937178202E-34;th'(0)=-4.33334237487128E-38;
    Iteration 1 failed, aborting. Error: 1.0833355937178202E-34 Total: 239634641049231.3800; Orientation: 0.0008; Line Search: 0.0207
    
```

Returns: 

```
    1.0833355937178202E-34
```



This training run resulted in the following regressed input:

Code from [LearningTester.java:154](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L154) executed in 0.01 seconds: 
```java
    return Arrays.stream(input_lbgfs).map(x -> x.prettyPrint()).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [
    	[ [ -1.584 ], [ 0.572 ], [ 0.408 ], [ 0.332 ], [ -0.388 ], [ -1.376 ], [ -1.164 ], [ 0.48 ], ... ],
    	[ [ 1.328 ], [ 0.792 ], [ 1.052 ], [ 0.776 ], [ 0.032 ], [ 0.448 ], [ -0.912 ], [ -0.152 ], ... ],
    	[ [ 1.42 ], [ -1.592 ], [ -0.3 ], [ 1.92 ], [ -0.736 ], [ 1.624 ], [ -0.948 ], [ 1.816 ], ... ],
    	[ [ 1.184 ], [ 0.7 ], [ 0.452 ], [ 1.78 ], [ -0.632 ], [ -0.556 ], [ -0.456 ], [ 0.208 ], ... ],
    	[ [ -0.192 ], [ 0.572 ], [ 0.204 ], [ -1.044 ], [ 0.272 ], [ 1.068 ], [ 1.868 ], [ -0.864 ], ... ],
    	[ [ -0.996 ], [ -1.852 ], [ -0.972 ], [ -0.604 ], [ 0.092 ], [ -1.44 ], [ 1.268 ], [ -1.64 ], ... ],
    	[ [ 1.192 ], [ -1.804 ], [ -0.248 ], [ -0.032 ], [ 0.316 ], [ -0.652 ], [ -0.848 ], [ 0.592 ], ... ],
    	[ [ -0.492 ], [ 0.12 ], [ -0.64 ], [ 0.504 ], [ 1.504 ], [ -1.588 ], [ -0.292 ], [ -1.292 ], ... ],
    	...
    ]
```



Code from [LearningTester.java:95](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L95) executed in 0.00 seconds: 
```java
    return TestUtil.compare(runs);
```

Code from [LearningTester.java:98](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L98) executed in 0.00 seconds: 
```java
    return TestUtil.compareTime(runs);
```

### Performance
Now we execute larger-scale runs to benchmark performance:

Code from [PerformanceTester.java:66](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/PerformanceTester.java#L66) executed in 0.01 seconds: 
```java
    test(component, inputPrototype);
```
Logging: 
```
    100 batches
    Input Dimensions:
    	[100, 100, 1]
    Performance:
    	Evaluation performance: 0.000489s +- 0.000036s [0.000444s - 0.000535s]
    	Learning performance: 0.000081s +- 0.000026s [0.000047s - 0.000124s]
    
```

