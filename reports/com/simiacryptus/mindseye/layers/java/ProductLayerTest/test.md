# ProductLayer
## ProductLayerTest
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
      "class": "com.simiacryptus.mindseye.layers.java.ProductLayer",
      "id": "649e5b14-f33f-4e52-9b12-346e2d094318",
      "isFrozen": false,
      "name": "ProductLayer/649e5b14-f33f-4e52-9b12-346e2d094318"
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
    [[ 1.268, -0.396, -1.812 ]]
    --------------------
    Output: 
    [ 0.909855936 ]
    --------------------
    Derivative: 
    [ 0.717552, -2.297616, -0.502128 ]
```



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
    Inputs: [ -1.368, 1.976, 0.932 ]
    Inputs Statistics: {meanExponent=0.13376298332989603, negative=1, min=0.932, max=0.932, mean=0.5133333333333333, count=3.0, positive=2, stdDev=1.3969121502641302, zeros=0}
    Output: [ -2.519352576 ]
    Outputs Statistics: {meanExponent=0.4012889499896881, negative=1, min=-2.519352576, max=-2.519352576, mean=-2.519352576, count=1.0, positive=0, stdDev=0.0, zeros=0}
    Feedback for input 0
    Inputs Values: [ -1.368, 1.976, 0.932 ]
    Value Statistics: {meanExponent=0.13376298332989603, negative=1, min=0.932, max=0.932, mean=0.5133333333333333, count=3.0, positive=2, stdDev=1.3969121502641302, zeros=0}
    Implemented Feedback: [ [ 1.841632 ], [ -1.274976 ], [ -2.7031680000000002 ] ]
    Implemented Statistics: {meanExponent=0.26752596665979206, negative=2, min=-2.7031680000000002, max=-2.7031680000000002, mean=-0.7121706666666668, count=3.0, positive=1, stdDev=1.8976062363622463, zeros=0}
    Measured Feedback: [ [ 1.8416319999969843 ], [ -1.2749760000030363 ], [ -2.7031680000000335 ] ]
    Measured Statistics: {meanExponent=0.2675259666599015, negative=2, min=-2.7031680000000335, max=-2.7031680000000335, mean=-0.7121706666686952, count=3.0, positive=1, stdDev=1.8976062363612052, zeros=0}
    Feedback Error: [ [ -3.01558777948685E-12 ], [ -3.036237927744878E-12 ], [ -3.3306690738754696E-14 ] ]
    Error Statistics: {meanExponent=-12.171920246582152, negative=3, min=-3.3306690738754696E-14, max=-3.3306690738754696E-14, mean=-2.028377465990161E-12, count=3.0, positive=0, stdDev=1.4107532635327496E-12, zeros=0}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 2.0284e-12 +- 1.4108e-12 [3.3307e-14 - 3.0362e-12] (3#)
    relativeTol: 6.7186e-13 +- 4.9461e-13 [6.1607e-15 - 1.1907e-12] (3#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=2.0284e-12 +- 1.4108e-12 [3.3307e-14 - 3.0362e-12] (3#), relativeTol=6.7186e-13 +- 4.9461e-13 [6.1607e-15 - 1.1907e-12] (3#)}
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
    	[3]
    Performance:
    	Evaluation performance: 0.000098s +- 0.000009s [0.000089s - 0.000110s]
    	Learning performance: 0.000033s +- 0.000022s [0.000021s - 0.000078s]
    
```

### Input Learning
In this test, we use a network to learn this target input, given it's pre-evaluated output:

Code from [LearningTester.java:127](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L127) executed in 0.00 seconds: 
```java
    return Arrays.stream(input_target).map(x -> x.prettyPrint()).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [ 1.028, -0.84, -0.024 ]
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

Returns: 

```
    0.0
```



This training run resulted in the following regressed input:

Code from [LearningTester.java:144](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L144) executed in 0.00 seconds: 
```java
    return Arrays.stream(input_gd).map(x -> x.prettyPrint()).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [ 1.028, -0.84, -0.024 ]
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

Returns: 

```
    0.0
```



This training run resulted in the following regressed input:

Code from [LearningTester.java:154](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L154) executed in 0.00 seconds: 
```java
    return Arrays.stream(input_lbgfs).map(x -> x.prettyPrint()).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [ 1.028, -0.84, -0.024 ]
```



Code from [LearningTester.java:96](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L96) executed in 0.00 seconds: 
```java
    return TestUtil.compare(runs);
```

Code from [LearningTester.java:99](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L99) executed in 0.00 seconds: 
```java
    return TestUtil.compareTime(runs);
```

