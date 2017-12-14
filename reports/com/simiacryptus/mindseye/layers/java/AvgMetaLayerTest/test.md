# AvgMetaLayer
## AvgMetaLayerTest
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
      "class": "com.simiacryptus.mindseye.layers.java.AvgMetaLayer",
      "id": "47c5ff38-0785-4c5c-b8f3-70feb56453d5",
      "isFrozen": false,
      "name": "AvgMetaLayer/47c5ff38-0785-4c5c-b8f3-70feb56453d5",
      "minBatchCount": 0
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
    [[ 0.884, 1.948, -0.884 ]]
    --------------------
    Output: 
    [ 0.884, 1.948, -0.884 ]
    --------------------
    Derivative: 
    [ 1.0, 1.0, 1.0 ]
```



### Differential Validation
Code from [StandardLayerTests.java:109](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L109) executed in 0.01 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [ 0.516, -0.156, -0.084 ],
    [ -0.776, -1.932, 0.14 ],
    [ -1.108, -1.172, -0.692 ]
    Inputs Statistics: {meanExponent=-0.7233154713188151, negative=2, min=-0.084, max=-0.084, mean=0.09199999999999998, count=3.0, positive=1, stdDev=0.3012507261402037, zeros=0},
    {meanExponent=-0.22600104032803295, negative=2, min=0.14, max=0.14, mean=-0.856, count=3.0, positive=1, stdDev=0.8477798456360393, zeros=0},
    {meanExponent=-0.015475511156253147, negative=3, min=-0.692, max=-0.692, mean=-0.9906666666666668, count=3.0, positive=0, stdDev=0.2127993316614389, zeros=0}
    Output: [ 0.516, -0.156, -0.084 ]
    Outputs Statistics: {meanExponent=-0.7233154713188151, negative=2, min=-0.084, max=-0.084, mean=0.09199999999999998, count=3.0, positive=1, stdDev=0.3012507261402037, zeros=0}
    Feedback for input 0
    Inputs Values: [ 0.516, -0.156, -0.084 ]
    Value Statistics: {meanExponent=-0.7233154713188151, negative=2, min=-0.084, max=-0.084, mean=0.09199999999999998, count=3.0, positive=1, stdDev=0.3012507261402037, zeros=0}
    Implemen
```
...[skipping 2654 bytes](etc/95.txt)...
```
    -0.47712125471966244, negative=0, min=0.3333333333333333, max=0.3333333333333333, mean=0.1111111111111111, count=9.0, positive=3, stdDev=0.15713484026367722, zeros=6}
    Measured Feedback: [ [ 0.3333333333332966, 0.0, 0.0 ], [ 0.0, 0.3333333333332966, 0.0 ], [ 0.0, 0.0, 0.3333333333332966 ] ]
    Measured Statistics: {meanExponent=-0.4771212547197103, negative=0, min=0.3333333333332966, max=0.3333333333332966, mean=0.11111111111109888, count=9.0, positive=3, stdDev=0.15713484026365993, zeros=6}
    Feedback Error: [ [ -3.6692870963861424E-14, 0.0, 0.0 ], [ 0.0, -3.6692870963861424E-14, 0.0 ], [ 0.0, 0.0, -3.6692870963861424E-14 ] ]
    Error Statistics: {meanExponent=-13.435418306369344, negative=3, min=-3.6692870963861424E-14, max=-3.6692870963861424E-14, mean=-1.2230956987953808E-14, count=9.0, positive=0, stdDev=1.7297185253166257E-14, zeros=6}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 1.2231e-14 +- 1.7297e-14 [0.0000e+00 - 3.6693e-14] (27#)
    relativeTol: 5.5039e-14 +- NaN [5.5039e-14 - 5.5039e-14] (9#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=1.2231e-14 +- 1.7297e-14 [0.0000e+00 - 3.6693e-14] (27#), relativeTol=5.5039e-14 +- NaN [5.5039e-14 - 5.5039e-14] (9#)}
```



### Performance
Code from [StandardLayerTests.java:119](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L119) executed in 0.01 seconds: 
```java
    getPerformanceTester().test(layer, permPrototype);
```
Logging: 
```
    100 batches
    Input Dimensions:
    	[100]
    Performance:
    	Evaluation performance: 0.000446s +- 0.000154s [0.000355s - 0.000753s]
    	Learning performance: 0.000008s +- 0.000003s [0.000006s - 0.000015s]
    
```

