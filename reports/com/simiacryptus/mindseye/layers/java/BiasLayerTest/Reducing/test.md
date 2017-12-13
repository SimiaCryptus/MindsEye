# BiasLayer
## Reducing
### Json Serialization
Code from [StandardLayerTests.java:69](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L69) executed in 0.00 seconds: 
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
      "class": "com.simiacryptus.mindseye.layers.java.BiasLayer",
      "id": "3da2b941-ff0b-494a-951a-d1209443240a",
      "isFrozen": false,
      "name": "BiasLayer/3da2b941-ff0b-494a-951a-d1209443240a",
      "bias": [
        0.0
      ]
    }
```



### Example Input/Output Pair
Code from [StandardLayerTests.java:153](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L153) executed in 0.00 seconds: 
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
    [[ 0.92, -0.816, 1.84 ]]
    --------------------
    Output: 
    [ 0.92, -0.816, 1.84 ]
    --------------------
    Derivative: 
    [ 1.0, 1.0, 1.0 ]
```



### Batch Execution
Code from [StandardLayerTests.java:102](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L102) executed in 0.00 seconds: 
```java
    return getBatchingTester().test(layer, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (60#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (60#)}
```



### Differential Validation
Code from [StandardLayerTests.java:110](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L110) executed in 0.00 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [ -0.092, -1.44, 1.348 ]
    Inputs Statistics: {meanExponent=-0.24938659611996464, negative=2, min=1.348, max=1.348, mean=-0.061333333333333316, count=3.0, positive=1, stdDev=1.1384027797264415, zeros=0}
    Output: [ -0.092, -1.44, 1.348 ]
    Outputs Statistics: {meanExponent=-0.24938659611996464, negative=2, min=1.348, max=1.348, mean=-0.061333333333333316, count=3.0, positive=1, stdDev=1.1384027797264415, zeros=0}
    Feedback for input 0
    Inputs Values: [ -0.092, -1.44, 1.348 ]
    Value Statistics: {meanExponent=-0.24938659611996464, negative=2, min=1.348, max=1.348, mean=-0.061333333333333316, count=3.0, positive=1, stdDev=1.1384027797264415, zeros=0}
    Implemented Feedback: [ [ 1.0, 0.0, 0.0 ], [ 0.0, 1.0, 0.0 ], [ 0.0, 0.0, 1.0 ] ]
    Implemented Statistics: {meanExponent=0.0, negative=0, min=1.0, max=1.0, mean=0.3333333333333333, count=9.0, positive=3, stdDev=0.4714045207910317, zeros=6}
    Measured Feedback: [ [ 1.0000000000000286, 0.0, 0.0 ], [ 0.0, 0.9999999999998899, 0.0 ], [ 0.0, 0.0, 0.9999999999998899 ]
```
...[skipping 550 bytes](etc/58.txt)...
```
    zeros=6}
    Learning Gradient for weight set 0
    Weights: [ 0.0 ]
    Implemented Gradient: [ [ 1.0, 1.0, 1.0 ] ]
    Implemented Statistics: {meanExponent=0.0, negative=0, min=1.0, max=1.0, mean=1.0, count=3.0, positive=3, stdDev=0.0, zeros=0}
    Measured Gradient: [ [ 1.0000000000000286, 0.9999999999998899, 0.9999999999998899 ] ]
    Measured Statistics: {meanExponent=-2.7740486787851373E-14, negative=0, min=0.9999999999998899, max=0.9999999999998899, mean=0.999999999999936, count=3.0, positive=3, stdDev=1.0536712127723509E-8, zeros=0}
    Gradient Error: [ [ 2.864375403532904E-14, -1.1013412404281553E-13, -1.1013412404281553E-13 ] ]
    Error Statistics: {meanExponent=-13.153042086767142, negative=2, min=-1.1013412404281553E-13, max=-1.1013412404281553E-13, mean=-6.387483135010068E-14, count=3.0, positive=1, stdDev=6.542051911182395E-14, zeros=0}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 4.1485e-14 +- 4.9587e-14 [0.0000e+00 - 1.1013e-13] (12#)
    relativeTol: 4.1485e-14 +- 1.9207e-14 [1.4322e-14 - 5.5067e-14] (6#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=4.1485e-14 +- 4.9587e-14 [0.0000e+00 - 1.1013e-13] (12#), relativeTol=4.1485e-14 +- 1.9207e-14 [1.4322e-14 - 5.5067e-14] (6#)}
```



### Performance
Code from [StandardLayerTests.java:120](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L120) executed in 0.00 seconds: 
```java
    getPerformanceTester().test(layer, permPrototype);
```
Logging: 
```
    Evaluation performance: 0.000175s +- 0.000035s [0.000140s - 0.000239s]
    Learning performance: 0.000153s +- 0.000005s [0.000146s - 0.000159s]
    
```

