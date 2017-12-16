# MeanSqLossLayer
## MeanSqLossLayerTest
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
      "class": "com.simiacryptus.mindseye.layers.java.MeanSqLossLayer",
      "id": "2094a8af-03ed-4dd4-97bc-f7ba7feccf4b",
      "isFrozen": false,
      "name": "MeanSqLossLayer/2094a8af-03ed-4dd4-97bc-f7ba7feccf4b"
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
    	[ [ 0.384 ], [ 0.86 ], [ -0.944 ] ],
    	[ [ -0.956 ], [ -0.144 ], [ 0.34 ] ]
    ],
    [
    	[ [ -1.78 ], [ -0.656 ], [ -1.948 ] ],
    	[ [ 0.724 ], [ 0.048 ], [ 0.656 ] ]
    ]]
    --------------------
    Output: 
    [ 1.8247146666666667 ]
    --------------------
    Derivative: 
    [
    	[ [ 0.7213333333333334 ], [ 0.5053333333333333 ], [ 0.33466666666666667 ] ],
    	[ [ -0.5599999999999999 ], [ -0.064 ], [ -0.10533333333333333 ] ]
    ],
    [
    	[ [ -0.7213333333333334 ], [ -0.5053333333333333 ], [ -0.33466666666666667 ] ],
    	[ [ 0.5599999999999999 ], [ 0.064 ], [ 0.10533333333333333 ] ]
    ]
```



### Batch Execution
Code from [BatchingTester.java:66](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/BatchingTester.java#L66) executed in 0.00 seconds: 
```java
    return test(reference, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (130#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (128#)}
```



Code from [SingleDerivativeTester.java:77](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/SingleDerivativeTester.java#L77) executed in 0.00 seconds: 
```java
    return test(component, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ -1.596 ], [ 1.204 ], [ 1.712 ] ],
    	[ [ -1.792 ], [ 0.8 ], [ -1.128 ] ]
    ],
    [
    	[ [ 1.448 ], [ -1.152 ], [ 1.528 ] ],
    	[ [ 1.18 ], [ -0.948 ], [ -0.152 ] ]
    ]
    Inputs Statistics: {meanExponent=0.12098337104050405, negative=3, min=-1.128, max=-1.128, mean=-0.13333333333333333, count=6.0, positive=3, stdDev=1.4109546019942985, zeros=0},
    {meanExponent=-0.06052027870384058, negative=3, min=-0.152, max=-0.152, mean=0.31733333333333336, count=6.0, positive=3, stdDev=1.1156909169757645, zeros=0}
    Output: [ 4.615232 ]
    Outputs Statistics: {meanExponent=0.6641935371685036, negative=0, min=4.615232, max=4.615232, mean=4.615232, count=1.0, positive=1, stdDev=0.0, zeros=0}
    Feedback for input 0
    Inputs Values: [
    	[ [ -1.596 ], [ 1.204 ], [ 1.712 ] ],
    	[ [ -1.792 ], [ 0.8 ], [ -1.128 ] ]
    ]
    Value Statistics: {meanExponent=0.12098337104050405, negative=3, min=-1.128, max=-1.128, mean=-0.13333333333333333, count=6.0, positive=3, stdDev=1.4109546019942985, zeros=0}
    Implemented Feedback: [ [ -1.0146666666666666 ],
```
...[skipping 1693 bytes](etc/349.txt)...
```
    n=0.1502222222222222, count=6.0, positive=3, stdDev=0.7001691506387386, zeros=0}
    Measured Feedback: [ [ 1.0146833333379845 ], [ 0.9906833333417353 ], [ -0.7853166666649258 ], [ -0.5826499999983525 ], [ -0.061316666659649854 ], [ 0.3253499999988918 ] ]
    Measured Statistics: {meanExponent=-0.33955878375686493, negative=3, min=0.3253499999988918, max=0.3253499999988918, mean=0.15023888889261391, count=6.0, positive=3, stdDev=0.700169150640302, zeros=0}
    Feedback Error: [ [ 1.6666671317944193E-5 ], [ 1.666667506872166E-5 ], [ 1.666666840738351E-5 ], [ 1.6666668314124777E-5 ], [ 1.6666673683454758E-5 ], [ 1.666666555855123E-5 ] ]
    Error Statistics: {meanExponent=-4.778151153318064, negative=0, min=1.666666555855123E-5, max=1.666666555855123E-5, mean=1.6666670391696687E-5, count=6.0, positive=6, stdDev=3.302793412861763E-12, zeros=0}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 1.6667e-05 +- 3.7568e-12 [1.6667e-05 - 1.6667e-05] (12#)
    relativeTol: 3.3837e-05 +- 4.6011e-05 [8.2128e-06 - 1.3589e-04] (12#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=1.6667e-05 +- 3.7568e-12 [1.6667e-05 - 1.6667e-05] (12#), relativeTol=3.3837e-05 +- 4.6011e-05 [8.2128e-06 - 1.3589e-04] (12#)}
```



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
    	[2, 3, 1]
    	[2, 3, 1]
    Performance:
    	Evaluation performance: 0.000250s +- 0.000060s [0.000203s - 0.000366s]
    	Learning performance: 0.000038s +- 0.000003s [0.000034s - 0.000041s]
    
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
    	[ [ 1.684 ], [ 0.172 ], [ -1.904 ] ],
    	[ [ 0.628 ], [ 1.572 ], [ 0.304 ] ]
    ]
    [
    	[ [ -1.316 ], [ 1.348 ], [ 0.084 ] ],
    	[ [ 0.864 ], [ -1.252 ], [ -0.844 ] ]
    ]
```



First, we use a conjugate gradient descent method, which converges the fastest for purely linear functions.

Code from [LearningTester.java:300](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L300) executed in 0.00 seconds: 
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
    Zero gradient: 0.0
    Constructing line search parameters: GD
    F(0.0) = LineSearchPoint{point=PointSample{avg=15.581040450567109}, derivative=0.0}
    Iteration 1 failed, aborting. Error: 15.581040450567109 Total: 249808759915797.2500; Orientation: 0.0000; Line Search: 0.0001
    
```

Returns: 

```
    15.581040450567109
```



This training run resulted in the following regressed input:

Code from [LearningTester.java:144](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L144) executed in 0.00 seconds: 
```java
    return Arrays.stream(input_gd).map(x -> x.prettyPrint()).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [
    	[ [ 1.684 ], [ -1.904 ], [ 1.572 ] ],
    	[ [ 0.304 ], [ 0.172 ], [ 0.628 ] ]
    ]
    [
    	[ [ -1.252 ], [ -1.316 ], [ 0.084 ] ],
    	[ [ 0.864 ], [ -0.844 ], [ 1.348 ] ]
    ]
```



Next, we run the same optimization using L-BFGS, which is nearly ideal for purely second-order or quadratic functions.

Code from [LearningTester.java:324](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L324) executed in 0.00 seconds: 
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
    th(0)=15.581040450567109;dx=0.0 (ERROR: Starting derivative negative)
    Iteration 1 failed, aborting. Error: 15.581040450567109 Total: 249808761897822.2500; Orientation: 0.0001; Line Search: 0.0002
    
```

Returns: 

```
    15.581040450567109
```



This training run resulted in the following regressed input:

Code from [LearningTester.java:154](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L154) executed in 0.00 seconds: 
```java
    return Arrays.stream(input_lbgfs).map(x -> x.prettyPrint()).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [
    	[ [ 1.684 ], [ -1.904 ], [ 1.572 ] ],
    	[ [ 0.304 ], [ 0.172 ], [ 0.628 ] ]
    ]
    [
    	[ [ -1.252 ], [ -1.316 ], [ 0.084 ] ],
    	[ [ 0.864 ], [ -0.844 ], [ 1.348 ] ]
    ]
```



Code from [LearningTester.java:96](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L96) executed in 0.00 seconds: 
```java
    return TestUtil.compare(runs);
```

Code from [LearningTester.java:99](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L99) executed in 0.00 seconds: 
```java
    return TestUtil.compareTime(runs);
```

