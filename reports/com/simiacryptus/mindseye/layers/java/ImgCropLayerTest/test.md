# ImgCropLayer
## ImgCropLayerTest
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
      "class": "com.simiacryptus.mindseye.layers.java.ImgCropLayer",
      "id": "9c30cabc-b5f7-4aab-9529-cc556c71de84",
      "isFrozen": false,
      "name": "ImgCropLayer/9c30cabc-b5f7-4aab-9529-cc556c71de84",
      "sizeX": 1,
      "sizeY": 1
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
    [[
    	[ [ -0.92 ], [ 0.848 ], [ 1.916 ] ],
    	[ [ 0.732 ], [ -1.752 ], [ 0.536 ] ],
    	[ [ -1.024 ], [ -0.244 ], [ -0.36 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ -1.752 ] ]
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
Code from [StandardLayerTests.java:109](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L109) executed in 0.00 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ 1.644 ], [ 0.688 ], [ 1.368 ] ],
    	[ [ -0.02 ], [ -1.204 ], [ 0.304 ] ],
    	[ [ 1.172 ], [ 1.196 ], [ 1.868 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.16976199130592007, negative=2, min=1.868, max=1.868, mean=0.7795555555555557, count=9.0, positive=7, stdDev=0.9076219368475584, zeros=0}
    Output: [
    	[ [ -1.204 ] ]
    ]
    Outputs Statistics: {meanExponent=0.08062648692180573, negative=1, min=-1.204, max=-1.204, mean=-1.204, count=1.0, positive=0, stdDev=0.0, zeros=0}
    Feedback for input 0
    Inputs Values: [
    	[ [ 1.644 ], [ 0.688 ], [ 1.368 ] ],
    	[ [ -0.02 ], [ -1.204 ], [ 0.304 ] ],
    	[ [ 1.172 ], [ 1.196 ], [ 1.868 ] ]
    ]
    Value Statistics: {meanExponent=-0.16976199130592007, negative=2, min=1.868, max=1.868, mean=0.7795555555555557, count=9.0, positive=7, stdDev=0.9076219368475584, zeros=0}
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
Code from [StandardLayerTests.java:119](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L119) executed in 0.00 seconds: 
```java
    getPerformanceTester().test(layer, permPrototype);
```
Logging: 
```
    100 batches
    Input Dimensions:
    	[3, 3, 1]
    Performance:
    	Evaluation performance: 0.000291s +- 0.000033s [0.000257s - 0.000344s]
    	Learning performance: 0.000106s +- 0.000033s [0.000064s - 0.000154s]
    
```

