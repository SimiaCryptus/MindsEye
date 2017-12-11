# MaxMetaLayer
## MaxMetaLayerTest
### Json Serialization
Code from [LayerTestBase.java:121](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L121) executed in 0.00 seconds: 
```java
    JsonObject json = layer.getJson();
    NNLayer echo = NNLayer.fromJson(json);
    assert (echo != null) : "Failed to deserialize";
    assert (layer != echo) : "Serialization did not copy";
    Assert.assertEquals("Serialization not equal", layer, echo);
    return new GsonBuilder().setPrettyPrinting().create().toJson(json);
```

Returns: 

```
    {
      "class": "com.simiacryptus.mindseye.layers.java.MaxMetaLayer",
      "id": "e2a3bda5-e7e7-4c05-aeb3-4ede00001ee7",
      "isFrozen": false,
      "name": "MaxMetaLayer/e2a3bda5-e7e7-4c05-aeb3-4ede00001ee7"
    }
```



### Example Input/Output Pair
Code from [LayerTestBase.java:159](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L159) executed in 0.00 seconds: 
```java
    SimpleEval eval = SimpleEval.run(layer, inputPrototype);
    return String.format("--------------------\nInput: \n[%s]\n--------------------\nOutput: \n%s",
      Arrays.stream(inputPrototype).map(t -> t.prettyPrint()).reduce((a, b) -> a + ",\n" + b).get(),
      eval.getOutput().prettyPrint());
```

Returns: 

```
    --------------------
    Input: 
    [[ -0.4, 0.244, 1.668 ]]
    --------------------
    Output: 
    [ -0.4, 0.244, 1.668 ]
```



### Batch Execution
Code from [LayerTestBase.java:178](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L178) executed in 0.00 seconds: 
```java
    BatchingTester batchingTester = getBatchingTester();
    return batchingTester == null ? null : batchingTester.test(layer, inputPrototype);
```

### Differential Validation
Code from [LayerTestBase.java:186](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L186) executed in 0.00 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [ -0.4, 0.244, 1.668 ]
    Inputs Statistics: {meanExponent=-0.2627847120105295, negative=1, min=1.668, max=1.668, mean=0.504, count=3.0, positive=2, stdDev=0.8640432087961034, zeros=0}
    Output: [ -0.4, 0.244, 1.668 ]
    Outputs Statistics: {meanExponent=-0.2627847120105295, negative=1, min=1.668, max=1.668, mean=0.504, count=3.0, positive=2, stdDev=0.8640432087961034, zeros=0}
    Feedback for input 0
    Inputs Values: [ -0.4, 0.244, 1.668 ]
    Value Statistics: {meanExponent=-0.2627847120105295, negative=1, min=1.668, max=1.668, mean=0.504, count=3.0, positive=2, stdDev=0.8640432087961034, zeros=0}
    Implemented Feedback: [ [ 1.0, 0.0, 0.0 ], [ 0.0, 1.0, 0.0 ], [ 0.0, 0.0, 1.0 ] ]
    Implemented Statistics: {meanExponent=0.0, negative=0, min=1.0, max=1.0, mean=0.3333333333333333, count=9.0, positive=3, stdDev=0.4714045207910317, zeros=6}
    Measured Feedback: [ [ 0.9999999999998899, 0.0, 0.0 ], [ 0.0, 0.9999999999998899, 0.0 ], [ 0.0, 0.0, 0.9999999999998899 ] ]
    Measured Statistics: {meanExponent=-4.7830642341045674E-14, negative=0, min=0.9999999999998899, max=0.9999999999998899, mean=0.3333333333332966, count=9.0, positive=3, stdDev=0.4714045207909798, zeros=6}
    Feedback Error: [ [ -1.1013412404281553E-13, 0.0, 0.0 ], [ 0.0, -1.1013412404281553E-13, 0.0 ], [ 0.0, 0.0, -1.1013412404281553E-13 ] ]
    Error Statistics: {meanExponent=-12.958078098036827, negative=3, min=-1.1013412404281553E-13, max=-1.1013412404281553E-13, mean=-3.671137468093851E-14, count=9.0, positive=0, stdDev=5.1917723967143496E-14, zeros=6}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 3.6711e-14 +- 5.1918e-14 [0.0000e+00 - 1.1013e-13] (9#)
    relativeTol: 5.5067e-14 +- 0.0000e+00 [5.5067e-14 - 5.5067e-14] (3#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=3.6711e-14 +- 5.1918e-14 [0.0000e+00 - 1.1013e-13] (9#), relativeTol=5.5067e-14 +- 0.0000e+00 [5.5067e-14 - 5.5067e-14] (3#)}
```



### Performance
Code from [LayerTestBase.java:192](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L192) executed in 0.00 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 0.2150 +- 0.0904 [0.1681 - 0.8720]
    Learning performance: 0.0017 +- 0.0016 [0.0000 - 0.0057]
    
```

