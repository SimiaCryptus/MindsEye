# CrossDifferenceLayer
## CrossDifferenceLayerTest
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
      "class": "com.simiacryptus.mindseye.layers.java.CrossDifferenceLayer",
      "id": "ffce18cc-cac4-4bea-bf62-027b7fbdb924",
      "isFrozen": false,
      "name": "CrossDifferenceLayer/ffce18cc-cac4-4bea-bf62-027b7fbdb924"
    }
```



### Example Input/Output Pair
Code from [StandardLayerTests.java:153](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L153) executed in 0.02 seconds: 
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
    [[ -1.94, 1.876, -1.136, 0.732 ]]
    --------------------
    Output: 
    [ -3.816, -0.804, -2.6719999999999997, 3.0119999999999996, 1.144, -1.8679999999999999 ]
    --------------------
    Derivative: 
    [ 3.0, 1.0, -1.0, -3.0 ]
```



### Batch Execution
Code from [StandardLayerTests.java:102](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L102) executed in 0.00 seconds: 
```java
    return getBatchingTester().test(layer, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (100#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (99#)}
```



### Differential Validation
Code from [StandardLayerTests.java:110](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L110) executed in 0.00 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [ 0.336, -2.0, 1.64, 0.78 ]
    Inputs Statistics: {meanExponent=-0.01642306905199911, negative=1, min=0.78, max=0.78, mean=0.189, count=4.0, positive=3, stdDev=1.3479625365713988, zeros=0}
    Output: [ 2.336, -1.3039999999999998, -0.444, -3.6399999999999997, -2.7800000000000002, 0.8599999999999999 ]
    Outputs Statistics: {meanExponent=0.1784630051269305, negative=4, min=0.8599999999999999, max=0.8599999999999999, mean=-0.8286666666666666, count=6.0, positive=2, stdDev=2.0392775409170985, zeros=0}
    Feedback for input 0
    Inputs Values: [ 0.336, -2.0, 1.64, 0.78 ]
    Value Statistics: {meanExponent=-0.01642306905199911, negative=1, min=0.78, max=0.78, mean=0.189, count=4.0, positive=3, stdDev=1.3479625365713988, zeros=0}
    Implemented Feedback: [ [ 1.0, 1.0, 1.0, 0.0, 0.0, 0.0 ], [ -1.0, 0.0, 0.0, 1.0, 1.0, 0.0 ], [ 0.0, -1.0, 0.0, -1.0, 0.0, 1.0 ], [ 0.0, 0.0, -1.0, 0.0, -1.0, -1.0 ] ]
    Implemented Statistics: {meanExponent=0.0, negative=6, min=-1.0, max=-1.0, mean=0.0, count=24.0, positive=6, stdDev=0.70710678
```
...[skipping 342 bytes](etc/62.txt)...
```
    0.9999999999998899 ] ]
    Measured Statistics: {meanExponent=-4.7830642341580963E-14, negative=6, min=-0.9999999999998899, max=-0.9999999999998899, mean=1.8503717077085943E-13, count=24.0, positive=6, stdDev=0.7071067811864696, zeros=12}
    Feedback Error: [ [ 2.1103119252074976E-12, -1.1013412404281553E-13, -1.1013412404281553E-13, 0.0, 0.0, 0.0 ], [ 2.3305801732931286E-12, 0.0, 0.0, -2.3305801732931286E-12, 2.1103119252074976E-12, 0.0 ], [ 0.0, 1.1013412404281553E-13, 0.0, -2.1103119252074976E-12, 0.0, -1.1013412404281553E-13 ], [ 0.0, 0.0, 1.1013412404281553E-13, 0.0, 2.3305801732931286E-12, 1.1013412404281553E-13 ] ]
    Error Statistics: {meanExponent=-12.306086373864746, negative=5, min=1.1013412404281553E-13, max=1.1013412404281553E-13, mean=1.8503717077085943E-13, count=24.0, positive=7, stdDev=1.0974612396254757E-12, zeros=12}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 5.8265e-13 +- 9.4825e-13 [0.0000e+00 - 2.3306e-12] (24#)
    relativeTol: 5.8265e-13 +- 5.2901e-13 [5.5067e-14 - 1.1653e-12] (12#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=5.8265e-13 +- 9.4825e-13 [0.0000e+00 - 2.3306e-12] (24#), relativeTol=5.8265e-13 +- 5.2901e-13 [5.5067e-14 - 1.1653e-12] (12#)}
```



### Performance
Code from [StandardLayerTests.java:120](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L120) executed in 0.01 seconds: 
```java
    getPerformanceTester().test(layer, permPrototype);
```
Logging: 
```
    Evaluation performance: 0.000436s +- 0.000086s [0.000320s - 0.000543s]
    Learning performance: 0.000098s +- 0.000031s [0.000046s - 0.000130s]
    
```

