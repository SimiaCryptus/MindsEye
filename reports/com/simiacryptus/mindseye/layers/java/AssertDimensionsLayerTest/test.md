# AssertDimensionsLayer
## AssertDimensionsLayerTest
### Json Serialization
Code from [StandardLayerTests.java:69](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L69) executed in 0.00 seconds: 
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
      "class": "com.simiacryptus.mindseye.layers.java.AssertDimensionsLayer",
      "id": "4697198c-4b7c-44bb-b156-c31f91b183f3",
      "isFrozen": false,
      "name": "AssertDimensionsLayer/4697198c-4b7c-44bb-b156-c31f91b183f3",
      "dims": [
        2,
        2
      ]
    }
```



### Example Input/Output Pair
Code from [StandardLayerTests.java:153](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L153) executed in 0.00 seconds: 
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
    [[ [ 1.584, -1.084 ], [ -1.244, 0.628 ] ]]
    --------------------
    Output: 
    [ [ 1.584, -1.084 ], [ -1.244, 0.628 ] ]
    --------------------
    Derivative: 
    [ [ 1.0, 1.0 ], [ 1.0, 1.0 ] ]
```



### Batch Execution
Code from [StandardLayerTests.java:102](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L102) executed in 0.00 seconds: 
```java
    return getBatchingTester().test(layer, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (80#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (80#)}
```



### Differential Validation
Code from [StandardLayerTests.java:110](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L110) executed in 0.00 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [ [ -1.36, -0.744 ], [ -0.56, -1.672 ] ]
    Inputs Statistics: {meanExponent=-0.005865963993676435, negative=4, min=-1.672, max=-1.672, mean=-1.084, count=4.0, positive=0, stdDev=0.4505818460612898, zeros=0}
    Output: [ [ -1.36, -0.744 ], [ -0.56, -1.672 ] ]
    Outputs Statistics: {meanExponent=-0.005865963993676435, negative=4, min=-1.672, max=-1.672, mean=-1.084, count=4.0, positive=0, stdDev=0.4505818460612898, zeros=0}
    Feedback for input 0
    Inputs Values: [ [ -1.36, -0.744 ], [ -0.56, -1.672 ] ]
    Value Statistics: {meanExponent=-0.005865963993676435, negative=4, min=-1.672, max=-1.672, mean=-1.084, count=4.0, positive=0, stdDev=0.4505818460612898, zeros=0}
    Implemented Feedback: [ [ 1.0, 0.0, 0.0, 0.0 ], [ 0.0, 1.0, 0.0, 0.0 ], [ 0.0, 0.0, 1.0, 0.0 ], [ 0.0, 0.0, 0.0, 1.0 ] ]
    Implemented Statistics: {meanExponent=0.0, negative=0, min=1.0, max=1.0, mean=0.25, count=16.0, positive=4, stdDev=0.4330127018922193, zeros=12}
    Measured Feedback: [ [ 0.9999999999998899, 0.0, 0.0, 0.0 ], [ 0.0, 0.9999999999998899, 0.0, 0.0 ], [ 0.0, 0.0, 0.9999999999998899, 0.0 ], [ 0.0, 0.0, 0.0, 0.9999999999998899 ] ]
    Measured Statistics: {meanExponent=-4.7830642341045674E-14, negative=0, min=0.9999999999998899, max=0.9999999999998899, mean=0.24999999999997247, count=16.0, positive=4, stdDev=0.4330127018921716, zeros=12}
    Feedback Error: [ [ -1.1013412404281553E-13, 0.0, 0.0, 0.0 ], [ 0.0, -1.1013412404281553E-13, 0.0, 0.0 ], [ 0.0, 0.0, -1.1013412404281553E-13, 0.0 ], [ 0.0, 0.0, 0.0, -1.1013412404281553E-13 ] ]
    Error Statistics: {meanExponent=-12.958078098036825, negative=4, min=-1.1013412404281553E-13, max=-1.1013412404281553E-13, mean=-2.7533531010703882E-14, count=16.0, positive=0, stdDev=4.7689474622312385E-14, zeros=12}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 2.7534e-14 +- 4.7689e-14 [0.0000e+00 - 1.1013e-13] (16#)
    relativeTol: 5.5067e-14 +- 0.0000e+00 [5.5067e-14 - 5.5067e-14] (4#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=2.7534e-14 +- 4.7689e-14 [0.0000e+00 - 1.1013e-13] (16#), relativeTol=5.5067e-14 +- 0.0000e+00 [5.5067e-14 - 5.5067e-14] (4#)}
```



### Performance
Code from [StandardLayerTests.java:120](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L120) executed in 0.00 seconds: 
```java
    getPerformanceTester().test(layer, permPrototype);
```
Logging: 
```
    Evaluation performance: 0.000023s +- 0.000004s [0.000020s - 0.000031s]
    Learning performance: 0.000028s +- 0.000003s [0.000025s - 0.000032s]
    
```

