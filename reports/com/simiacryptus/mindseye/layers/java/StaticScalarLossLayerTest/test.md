# StaticScalarLossLayer
## StaticScalarLossLayerTest
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
      "class": "com.simiacryptus.mindseye.layers.java.StaticScalarLossLayer",
      "id": "a4a694c0-484d-4caa-b9a1-6dd394d96472",
      "isFrozen": false,
      "name": "StaticScalarLossLayer/a4a694c0-484d-4caa-b9a1-6dd394d96472"
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
    [[ -0.764 ]]
    --------------------
    Output: 
    [ 0.764 ]
    --------------------
    Derivative: 
    [ -1.0 ]
```



### Differential Validation
Code from [StandardLayerTests.java:109](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L109) executed in 0.00 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [ -1.308 ]
    Inputs Statistics: {meanExponent=0.11660774398824848, negative=1, min=-1.308, max=-1.308, mean=-1.308, count=1.0, positive=0, stdDev=0.0, zeros=0}
    Output: [ 1.308 ]
    Outputs Statistics: {meanExponent=0.11660774398824848, negative=0, min=1.308, max=1.308, mean=1.308, count=1.0, positive=1, stdDev=0.0, zeros=0}
    Feedback for input 0
    Inputs Values: [ -1.308 ]
    Value Statistics: {meanExponent=0.11660774398824848, negative=1, min=-1.308, max=-1.308, mean=-1.308, count=1.0, positive=0, stdDev=0.0, zeros=0}
    Implemented Feedback: [ [ -1.0 ] ]
    Implemented Statistics: {meanExponent=0.0, negative=1, min=-1.0, max=-1.0, mean=-1.0, count=1.0, positive=0, stdDev=0.0, zeros=0}
    Measured Feedback: [ [ -0.9999999999998899 ] ]
    Measured Statistics: {meanExponent=-4.7830642341045674E-14, negative=1, min=-0.9999999999998899, max=-0.9999999999998899, mean=-0.9999999999998899, count=1.0, positive=0, stdDev=0.0, zeros=0}
    Feedback Error: [ [ 1.1013412404281553E-13 ] ]
    Error Statistics: {meanExponent=-12.958078098036825, negative=0, min=1.1013412404281553E-13, max=1.1013412404281553E-13, mean=1.1013412404281553E-13, count=1.0, positive=1, stdDev=0.0, zeros=0}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 1.1013e-13 +- 0.0000e+00 [1.1013e-13 - 1.1013e-13] (1#)
    relativeTol: 5.5067e-14 +- 0.0000e+00 [5.5067e-14 - 5.5067e-14] (1#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=1.1013e-13 +- 0.0000e+00 [1.1013e-13 - 1.1013e-13] (1#), relativeTol=5.5067e-14 +- 0.0000e+00 [5.5067e-14 - 5.5067e-14] (1#)}
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
    	[1]
    Performance:
    	Evaluation performance: 0.000170s +- 0.000014s [0.000152s - 0.000194s]
    	Learning performance: 0.000060s +- 0.000005s [0.000054s - 0.000070s]
    
```

