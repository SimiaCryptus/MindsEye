# ImgCropLayer
## ImgCropLayerTest
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
      "class": "com.simiacryptus.mindseye.layers.java.ImgCropLayer",
      "id": "9d2037e1-71b4-4648-b2f9-b63e73146621",
      "isFrozen": false,
      "name": "ImgCropLayer/9d2037e1-71b4-4648-b2f9-b63e73146621",
      "sizeX": 1,
      "sizeY": 1
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
    [[
    	[ [ 0.488 ], [ 1.412 ], [ -1.456 ] ],
    	[ [ -0.764 ], [ 1.404 ], [ -0.384 ] ],
    	[ [ 1.876 ], [ 1.428 ], [ -1.5 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 1.404 ] ]
    ]
    --------------------
    Derivative: 
    [
    	[ [ 0.0 ], [ 0.0 ], [ 0.0 ] ],
    	[ [ 0.0 ], [ 1.0 ], [ 0.0 ] ],
    	[ [ 0.0 ], [ 0.0 ], [ 0.0 ] ]
    ]
```



### Differential Validation
Code from [StandardLayerTests.java:110](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L110) executed in 0.00 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ 0.344 ], [ -0.812 ], [ -1.58 ] ],
    	[ [ -1.456 ], [ -0.532 ], [ 0.868 ] ],
    	[ [ 0.924 ], [ -1.896 ], [ -0.044 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.1822969697862191, negative=6, min=-0.044, max=-0.044, mean=-0.4648888888888888, count=9.0, positive=3, stdDev=0.9964323519932327, zeros=0}
    Output: [
    	[ [ -0.532 ] ]
    ]
    Outputs Statistics: {meanExponent=-0.2740883677049518, negative=1, min=-0.532, max=-0.532, mean=-0.532, count=1.0, positive=0, stdDev=0.0, zeros=0}
    Feedback for input 0
    Inputs Values: [
    	[ [ 0.344 ], [ -0.812 ], [ -1.58 ] ],
    	[ [ -1.456 ], [ -0.532 ], [ 0.868 ] ],
    	[ [ 0.924 ], [ -1.896 ], [ -0.044 ] ]
    ]
    Value Statistics: {meanExponent=-0.1822969697862191, negative=6, min=-0.044, max=-0.044, mean=-0.4648888888888888, count=9.0, positive=3, stdDev=0.9964323519932327, zeros=0}
    Implemented Feedback: [ [ 0.0 ], [ 0.0 ], [ 0.0 ], [ 0.0 ], [ 1.0 ], [ 0.0 ], [ 0.0 ], [ 0.0 ], [ 0.0 ] ]
    Implemented Statistics: {meanExponent=0.0, negative=0, min=0.0, max=0.0, mean=0.1111111111111111, count=9.0, positive=1, stdDev=0.31426968052735443, zeros=8}
    Measured Feedback: [ [ 0.0 ], [ 0.0 ], [ 0.0 ], [ 0.0 ], [ 0.9999999999998899 ], [ 0.0 ], [ 0.0 ], [ 0.0 ], [ 0.0 ] ]
    Measured Statistics: {meanExponent=-4.7830642341045674E-14, negative=0, min=0.0, max=0.0, mean=0.11111111111109888, count=9.0, positive=1, stdDev=0.31426968052731985, zeros=8}
    Feedback Error: [ [ 0.0 ], [ 0.0 ], [ 0.0 ], [ 0.0 ], [ -1.1013412404281553E-13 ], [ 0.0 ], [ 0.0 ], [ 0.0 ], [ 0.0 ] ]
    Error Statistics: {meanExponent=-12.958078098036825, negative=1, min=0.0, max=0.0, mean=-1.223712489364617E-14, count=9.0, positive=0, stdDev=3.461181597809566E-14, zeros=8}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 1.2237e-14 +- 3.4612e-14 [0.0000e+00 - 1.1013e-13] (9#)
    relativeTol: 5.5067e-14 +- 0.0000e+00 [5.5067e-14 - 5.5067e-14] (1#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=1.2237e-14 +- 3.4612e-14 [0.0000e+00 - 1.1013e-13] (9#), relativeTol=5.5067e-14 +- 0.0000e+00 [5.5067e-14 - 5.5067e-14] (1#)}
```



### Performance
Code from [StandardLayerTests.java:120](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L120) executed in 0.00 seconds: 
```java
    getPerformanceTester().test(layer, permPrototype);
```
Logging: 
```
    Evaluation performance: 0.000226s +- 0.000049s [0.000169s - 0.000311s]
    Learning performance: 0.000062s +- 0.000005s [0.000057s - 0.000072s]
    
```

