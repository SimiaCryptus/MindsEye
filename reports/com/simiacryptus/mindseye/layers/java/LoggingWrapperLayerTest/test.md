# LoggingWrapperLayer
## LoggingWrapperLayerTest
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
      "class": "com.simiacryptus.mindseye.layers.java.LoggingWrapperLayer",
      "id": "c5e28f31-7244-4d2d-9e88-7090d49a176b",
      "isFrozen": false,
      "name": "LoggingWrapperLayer/c5e28f31-7244-4d2d-9e88-7090d49a176b",
      "inner": {
        "class": "com.simiacryptus.mindseye.layers.java.LinearActivationLayer",
        "id": "7690ee1d-c341-4baf-868c-a5e094494f91",
        "isFrozen": false,
        "name": "LinearActivationLayer/7690ee1d-c341-4baf-868c-a5e094494f91",
        "weights": [
          1.0,
          0.0
        ]
      }
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
    [[ -0.708, -0.988, 1.052 ]]
    --------------------
    Output: 
    [ -0.708, -0.988, 1.052 ]
    --------------------
    Derivative: 
    [ 1.0, 1.0, 1.0 ]
```



### Differential Validation
Code from [StandardLayerTests.java:110](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L110) executed in 0.00 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [ -1.024, -0.98, -0.164 ]
    Inputs Statistics: {meanExponent=-0.26121003987333175, negative=3, min=-0.164, max=-0.164, mean=-0.7226666666666667, count=3.0, positive=0, stdDev=0.3954451781080263, zeros=0}
    Output: [ -1.024, -0.98, -0.164 ]
    Outputs Statistics: {meanExponent=-0.26121003987333175, negative=3, min=-0.164, max=-0.164, mean=-0.7226666666666667, count=3.0, positive=0, stdDev=0.3954451781080263, zeros=0}
    Input 0 for layer LinearActivationLayer/7690ee1d-c341-4baf-868c-a5e094494f91: 
    	[ -1.024, -0.98, -0.164 ]
    Output for layer LinearActivationLayer/7690ee1d-c341-4baf-868c-a5e094494f91: 
    	[ -1.024, -0.98, -0.164 ]
    Feedback Input for layer LinearActivationLayer/7690ee1d-c341-4baf-868c-a5e094494f91: 
    	[ 1.0, 0.0, 0.0 ]
    Feedback Output 0 for layer LinearActivationLayer/7690ee1d-c341-4baf-868c-a5e094494f91: 
    	[ 1.0, 0.0, 0.0 ]
    Input 0 for layer LinearActivationLayer/7690ee1d-c341-4baf-868c-a5e094494f91: 
    	[ -1.024, -0.98, -0.164 ]
    Output for layer LinearActivationLayer/7690ee1d-c341-4baf-868c-a
```
...[skipping 2057 bytes](etc/79.txt)...
```
    .0, max=1.0, mean=0.13866666666666666, count=6.0, positive=3, stdDev=0.9055846484760856, zeros=0}
    Measured Gradient: [ [ -1.02400000000058, -0.980000000000425, -0.16399999999999748 ], [ 0.9999999999998899, 0.9999999999998899, 0.9999999999998899 ] ]
    Measured Statistics: {meanExponent=-0.13060501993661852, negative=3, min=0.9999999999998899, max=0.9999999999998899, mean=0.13866666666644453, count=6.0, positive=3, stdDev=0.9055846484762446, zeros=0}
    Gradient Error: [ [ -5.799805080641818E-13, -4.249933738265099E-13, 2.525757381022231E-15 ], [ -1.1013412404281553E-13, -1.1013412404281553E-13, -1.1013412404281553E-13 ] ]
    Error Statistics: {meanExponent=-13.013341184386421, negative=5, min=-1.1013412404281553E-13, max=-1.1013412404281553E-13, mean=-2.22141749439686E-13, count=6.0, positive=1, stdDev=2.0708692689804732E-13, zeros=0}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 1.1122e-13 +- 1.6416e-13 [0.0000e+00 - 5.7998e-13] (15#)
    relativeTol: 9.3126e-14 +- 8.6550e-14 [7.7005e-15 - 2.8319e-13] (9#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=1.1122e-13 +- 1.6416e-13 [0.0000e+00 - 5.7998e-13] (15#), relativeTol=9.3126e-14 +- 8.6550e-14 [7.7005e-15 - 2.8319e-13] (9#)}
```



### Performance
Code from [StandardLayerTests.java:120](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L120) executed in 0.02 seconds: 
```java
    getPerformanceTester().test(layer, permPrototype);
```
Logging: 
```
    Evaluation performance: 0.001713s +- 0.000389s [0.001242s - 0.002297s]
    Learning performance: 0.000705s +- 0.000103s [0.000515s - 0.000799s]
    
```

