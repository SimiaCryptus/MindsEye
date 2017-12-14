# ProductLayer
## ProductLayerTest
### Json Serialization
Code from [StandardLayerTests.java:68](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L68) executed in 0.00 seconds: 
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
      "id": "1549eb4f-841d-4f89-8a31-c98203c4cd3b",
      "isFrozen": false,
      "name": "ProductLayer/1549eb4f-841d-4f89-8a31-c98203c4cd3b"
    }
```



### Example Input/Output Pair
Code from [StandardLayerTests.java:152](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L152) executed in 0.00 seconds: 
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
    [[ -1.14, -0.912, 1.332 ]]
    --------------------
    Output: 
    [ 1.38485376 ]
    --------------------
    Derivative: 
    [ -1.214784, -1.5184799999999998, 1.03968 ]
```



### Batch Execution
Code from [StandardLayerTests.java:101](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L101) executed in 0.00 seconds: 
```java
    return getBatchingTester().test(layer, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (40#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (40#)}
```



### Differential Validation
Code from [StandardLayerTests.java:109](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L109) executed in 0.00 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [ 1.968, -1.152, 0.656 ]
    Inputs Statistics: {meanExponent=0.0574604708527254, negative=1, min=0.656, max=0.656, mean=0.49066666666666664, count=3.0, positive=2, stdDev=1.279088564390893, zeros=0}
    Output: [ -1.487241216 ]
    Outputs Statistics: {meanExponent=0.1723814125581762, negative=1, min=-1.487241216, max=-1.487241216, mean=-1.487241216, count=1.0, positive=0, stdDev=0.0, zeros=0}
    Feedback for input 0
    Inputs Values: [ 1.968, -1.152, 0.656 ]
    Value Statistics: {meanExponent=0.0574604708527254, negative=1, min=0.656, max=0.656, mean=0.49066666666666664, count=3.0, positive=2, stdDev=1.279088564390893, zeros=0}
    Implemented Feedback: [ [ -0.7557119999999999 ], [ 1.291008 ], [ -2.267136 ] ]
    Implemented Statistics: {meanExponent=0.1149209417054508, negative=2, min=-2.267136, max=-2.267136, mean=-0.5772799999999999, count=3.0, positive=1, stdDev=1.4580753673140494, zeros=0}
    Measured Feedback: [ [ -0.7557119999979101 ], [ 1.2910079999994828 ], [ -2.2671360000003915 ] ]
    Measured Statistics: {meanExponent=0.11492094170501747, negative=2, min=-2.2671360000003915, max=-2.2671360000003915, mean=-0.5772799999996062, count=3.0, positive=1, stdDev=1.4580753673138946, zeros=0}
    Feedback Error: [ [ 2.0898838215543947E-12 ], [ -5.171418848703979E-13 ], [ -3.9168668308775523E-13 ] ]
    Error Statistics: {meanExponent=-12.124443111915534, negative=2, min=-3.9168668308775523E-13, max=-3.9168668308775523E-13, mean=3.936850845320805E-13, count=3.0, positive=1, stdDev=1.2004866703468366E-12, zeros=0}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 9.9957e-13 +- 7.7267e-13 [3.9169e-13 - 2.0899e-12] (3#)
    relativeTol: 5.5646e-13 +- 5.8610e-13 [8.6384e-14 - 1.3827e-12] (3#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=9.9957e-13 +- 7.7267e-13 [3.9169e-13 - 2.0899e-12] (3#), relativeTol=5.5646e-13 +- 5.8610e-13 [8.6384e-14 - 1.3827e-12] (3#)}
```



### Performance
Code from [StandardLayerTests.java:119](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L119) executed in 0.00 seconds: 
```java
    getPerformanceTester().test(layer, permPrototype);
```
Logging: 
```
    100 batches
    Input Dimensions:
    	[3]
    Performance:
    	Evaluation performance: 0.000103s +- 0.000008s [0.000092s - 0.000116s]
    	Learning performance: 0.000025s +- 0.000002s [0.000023s - 0.000028s]
    
```

