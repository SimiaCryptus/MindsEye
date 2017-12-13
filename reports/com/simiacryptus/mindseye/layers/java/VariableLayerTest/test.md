# VariableLayer
## VariableLayerTest
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
      "class": "com.simiacryptus.mindseye.layers.java.VariableLayer",
      "id": "309bb69f-0eb4-4d1b-9236-0cf2f2e87232",
      "isFrozen": false,
      "name": "VariableLayer/309bb69f-0eb4-4d1b-9236-0cf2f2e87232",
      "inner": {
        "class": "com.simiacryptus.mindseye.layers.java.MonitoringSynapse",
        "id": "6d3e5f14-f438-4221-a88b-d5075213ab39",
        "isFrozen": false,
        "name": "MonitoringSynapse/6d3e5f14-f438-4221-a88b-d5075213ab39",
        "totalBatches": 0,
        "totalItems": 0
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
    [[ -0.512, 0.488, 1.016 ]]
    --------------------
    Output: 
    [ -0.512, 0.488, 1.016 ]
    --------------------
    Derivative: 
    [ 1.0, 1.0, 1.0 ]
```



### Batch Execution
Code from [StandardLayerTests.java:102](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L102) executed in 0.00 seconds: 
```java
    return getBatchingTester().test(layer, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (60#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (60#)}
```



### Differential Validation
Code from [StandardLayerTests.java:110](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L110) executed in 0.00 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [ 0.012, -1.796, -0.708 ]
    Inputs Statistics: {meanExponent=-0.6054930546437736, negative=2, min=-0.708, max=-0.708, mean=-0.8306666666666667, count=3.0, positive=1, stdDev=0.743191914798743, zeros=0}
    Output: [ 0.012, -1.796, -0.708 ]
    Outputs Statistics: {meanExponent=-0.6054930546437736, negative=2, min=-0.708, max=-0.708, mean=-0.8306666666666667, count=3.0, positive=1, stdDev=0.743191914798743, zeros=0}
    Feedback for input 0
    Inputs Values: [ 0.012, -1.796, -0.708 ]
    Value Statistics: {meanExponent=-0.6054930546437736, negative=2, min=-0.708, max=-0.708, mean=-0.8306666666666667, count=3.0, positive=1, stdDev=0.743191914798743, zeros=0}
    Implemented Feedback: [ [ 1.0, 0.0, 0.0 ], [ 0.0, 1.0, 0.0 ], [ 0.0, 0.0, 1.0 ] ]
    Implemented Statistics: {meanExponent=0.0, negative=0, min=1.0, max=1.0, mean=0.3333333333333333, count=9.0, positive=3, stdDev=0.4714045207910317, zeros=6}
    Measured Feedback: [ [ 0.999999999999994, 0.0, 0.0 ], [ 0.0, 0.9999999999998899, 0.0 ], [ 0.0, 0.0, 0.9999999999998899 ] ]
    Measured Statistics: {meanExponent=-3.275498961392841E-14, negative=0, min=0.9999999999998899, max=0.9999999999998899, mean=0.33333333333330817, count=9.0, positive=3, stdDev=0.47140452079099615, zeros=6}
    Feedback Error: [ [ -5.995204332975845E-15, 0.0, 0.0 ], [ 0.0, -1.1013412404281553E-13, 0.0 ], [ 0.0, 0.0, -1.1013412404281553E-13 ] ]
    Error Statistics: {meanExponent=-13.37945073548056, negative=3, min=-1.1013412404281553E-13, max=-1.1013412404281553E-13, mean=-2.5140383602067435E-14, count=9.0, positive=0, stdDev=4.5468723124825996E-14, zeros=6}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 2.5140e-14 +- 4.5469e-14 [0.0000e+00 - 1.1013e-13] (9#)
    relativeTol: 3.7711e-14 +- 2.4546e-14 [2.9976e-15 - 5.5067e-14] (3#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=2.5140e-14 +- 4.5469e-14 [0.0000e+00 - 1.1013e-13] (9#), relativeTol=3.7711e-14 +- 2.4546e-14 [2.9976e-15 - 5.5067e-14] (3#)}
```



### Performance
Code from [StandardLayerTests.java:120](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L120) executed in 0.00 seconds: 
```java
    getPerformanceTester().test(layer, permPrototype);
```
Logging: 
```
    Evaluation performance: 0.000153s +- 0.000019s [0.000130s - 0.000180s]
    Learning performance: 0.000167s +- 0.000023s [0.000136s - 0.000201s]
    
```

