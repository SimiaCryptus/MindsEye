# MaxMetaLayer
## MaxMetaLayerTest
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
      "class": "com.simiacryptus.mindseye.layers.java.MaxMetaLayer",
      "id": "bdc5ac75-bcb3-4259-a954-a8f04e2ae174",
      "isFrozen": false,
      "name": "MaxMetaLayer/bdc5ac75-bcb3-4259-a954-a8f04e2ae174"
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
    [[ -1.16, 0.432, -1.476 ]]
    --------------------
    Output: 
    [ -1.16, 0.432, -1.476 ]
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
    Inputs: [ -1.844, 1.936, -0.92 ],
    [ -0.712, -1.92, 1.404 ],
    [ -0.312, 1.008, -1.076 ]
    Inputs Statistics: {meanExponent=0.17215136567851352, negative=2, min=-0.92, max=-0.92, mean=-0.2760000000000001, count=3.0, positive=1, stdDev=1.6089648846385678, zeros=0},
    {meanExponent=0.09438277671139746, negative=2, min=1.404, max=1.404, mean=-0.40933333333333327, count=3.0, positive=1, stdDev=1.373790215264163, zeros=0},
    {meanExponent=-0.15685753418056012, negative=2, min=-1.076, max=-1.076, mean=-0.1266666666666667, count=3.0, positive=1, stdDev=0.8608233784516363, zeros=0}
    Output: [ -1.844, 1.936, -0.92 ]
    Outputs Statistics: {meanExponent=0.17215136567851352, negative=2, min=-0.92, max=-0.92, mean=-0.2760000000000001, count=3.0, positive=1, stdDev=1.6089648846385678, zeros=0}
    Feedback for input 0
    Inputs Values: [ -1.844, 1.936, -0.92 ]
    Value Statistics: {meanExponent=0.17215136567851352, negative=2, min=-0.92, max=-0.92, mean=-0.2760000000000001, count=3.0, positive=1, stdDev=1.6089648846385678, zeros=0}
    Imple
```
...[skipping 2056 bytes](etc/83.txt)...
```
    nt=3.0, positive=1, stdDev=0.8608233784516363, zeros=0}
    Implemented Feedback: [ [ 1.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ] ]
    Implemented Statistics: {meanExponent=0.0, negative=0, min=0.0, max=0.0, mean=0.1111111111111111, count=9.0, positive=1, stdDev=0.31426968052735443, zeros=8}
    Measured Feedback: [ [ 0.9999999999998899, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ] ]
    Measured Statistics: {meanExponent=-4.7830642341045674E-14, negative=0, min=0.0, max=0.0, mean=0.11111111111109888, count=9.0, positive=1, stdDev=0.31426968052731985, zeros=8}
    Feedback Error: [ [ -1.1013412404281553E-13, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ] ]
    Error Statistics: {meanExponent=-12.958078098036825, negative=1, min=0.0, max=0.0, mean=-1.223712489364617E-14, count=9.0, positive=0, stdDev=3.461181597809566E-14, zeros=8}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 1.2237e-14 +- 3.4612e-14 [0.0000e+00 - 1.1013e-13] (27#)
    relativeTol: 5.5067e-14 +- 0.0000e+00 [5.5067e-14 - 5.5067e-14] (3#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=1.2237e-14 +- 3.4612e-14 [0.0000e+00 - 1.1013e-13] (27#), relativeTol=5.5067e-14 +- 0.0000e+00 [5.5067e-14 - 5.5067e-14] (3#)}
```



### Performance
Code from [StandardLayerTests.java:120](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L120) executed in 0.00 seconds: 
```java
    getPerformanceTester().test(layer, permPrototype);
```
Logging: 
```
    Evaluation performance: 0.000166s +- 0.000059s [0.000099s - 0.000242s]
    Learning performance: 0.000004s +- 0.000001s [0.000002s - 0.000005s]
    
```

