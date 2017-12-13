# CrossDotMetaLayer
## CrossDotMetaLayerTest
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
      "class": "com.simiacryptus.mindseye.layers.java.CrossDotMetaLayer",
      "id": "cf81eb48-0f10-4d40-857a-b4ce766ca0af",
      "isFrozen": false,
      "name": "CrossDotMetaLayer/cf81eb48-0f10-4d40-857a-b4ce766ca0af"
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
    [[ -0.52, -1.892, 0.204 ]]
    --------------------
    Output: 
    [ [ 0.0, 0.9838399999999999, -0.10608 ], [ 0.9838399999999999, 0.0, -0.385968 ], [ -0.10608, -0.385968, 0.0 ] ]
    --------------------
    Derivative: 
    [ -3.376, -0.6320000000000001, -4.824 ]
```



### Differential Validation
Code from [StandardLayerTests.java:110](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L110) executed in 0.01 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [ 0.312, -0.704, 0.26 ]
    Inputs Statistics: {meanExponent=-0.414433132956209, negative=1, min=0.26, max=0.26, mean=-0.043999999999999984, count=3.0, positive=2, stdDev=0.4671730585839327, zeros=0}
    Output: [ [ 0.0, -0.21964799999999998, 0.08112 ], [ -0.21964799999999998, 0.0, -0.18304 ], [ 0.08112, -0.18304, 0.0 ] ]
    Outputs Statistics: {meanExponent=-0.828866265912418, negative=4, min=0.0, max=0.0, mean=-0.07145955555555555, count=9.0, positive=2, stdDev=0.12050839854836407, zeros=3}
    Feedback for input 0
    Inputs Values: [ 0.312, -0.704, 0.26 ]
    Value Statistics: {meanExponent=-0.414433132956209, negative=1, min=0.26, max=0.26, mean=-0.043999999999999984, count=3.0, positive=2, stdDev=0.4671730585839327, zeros=0}
    Implemented Feedback: [ [ 0.0, -0.704, 0.26, -0.704, 0.0, 0.0, 0.26, 0.0, 0.0 ], [ 0.0, 0.312, 0.0, 0.312, 0.0, 0.26, 0.0, 0.26, 0.0 ], [ 0.0, 0.0, 0.312, 0.0, 0.0, -0.704, 0.312, -0.704, 0.0 ] ]
    Implemented Statistics: {meanExponent=-0.41443313295620904, negative=4, min=0.0, max=0.0, mean
```
...[skipping 350 bytes](etc/63.txt)...
```
    99999999512, 0.0, 0.0, -0.7039999999997049, 0.3119999999999512, -0.7039999999997049, 0.0 ] ]
    Measured Statistics: {meanExponent=-0.41443313295630574, negative=4, min=0.0, max=0.0, mean=-0.0195555555555525, count=27.0, positive=8, stdDev=0.3122151837877356, zeros=15}
    Feedback Error: [ [ 0.0, 1.7541523789077473E-14, -1.7541523789077473E-14, 1.7541523789077473E-14, 0.0, 0.0, -1.7541523789077473E-14, 0.0, 0.0 ], [ 0.0, -1.875721800104202E-13, 0.0, -1.875721800104202E-13, 0.0, -1.7541523789077473E-14, 0.0, -1.7541523789077473E-14, 0.0 ], [ 0.0, 0.0, -4.879430193227563E-14, 0.0, 0.0, 2.950972799453666E-13, -4.879430193227563E-14, 2.950972799453666E-13, 0.0 ] ]
    Error Statistics: {meanExponent=-13.30604921802589, negative=8, min=0.0, max=0.0, mean=3.0510573491550598E-15, count=27.0, positive=4, stdDev=9.639581041257319E-14, zeros=15}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 4.3266e-14 +- 8.6195e-14 [0.0000e+00 - 2.9510e-13] (27#)
    relativeTol: 1.1138e-13 +- 1.0676e-13 [1.2458e-14 - 3.0060e-13] (12#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=4.3266e-14 +- 8.6195e-14 [0.0000e+00 - 2.9510e-13] (27#), relativeTol=1.1138e-13 +- 1.0676e-13 [1.2458e-14 - 3.0060e-13] (12#)}
```



### Performance
Code from [StandardLayerTests.java:120](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L120) executed in 0.00 seconds: 
```java
    getPerformanceTester().test(layer, permPrototype);
```
Logging: 
```
    Evaluation performance: 0.000228s +- 0.000162s [0.000094s - 0.000520s]
    Learning performance: 0.000005s +- 0.000006s [0.000002s - 0.000018s]
    
```

