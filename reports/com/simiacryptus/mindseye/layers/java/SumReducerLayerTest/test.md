# SumReducerLayer
## SumReducerLayerTest
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
      "class": "com.simiacryptus.mindseye.layers.java.SumReducerLayer",
      "id": "e2a3bda5-e7e7-4c05-aeb3-4ede0000368c",
      "isFrozen": false,
      "name": "SumReducerLayer/e2a3bda5-e7e7-4c05-aeb3-4ede0000368c"
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
    [[ -1.468, 1.56, 0.676 ]]
    --------------------
    Output: 
    [ 0.7680000000000001 ]
```



### Batch Execution
Code from [LayerTestBase.java:178](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L178) executed in 0.00 seconds: 
```java
    BatchingTester batchingTester = getBatchingTester();
    return batchingTester == null ? null : batchingTester.test(layer, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (40#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (40#)}
```



### Differential Validation
Code from [LayerTestBase.java:186](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L186) executed in 0.00 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [ -1.468, 1.56, 0.676 ]
    Inputs Statistics: {meanExponent=0.06326578329204975, negative=1, min=0.676, max=0.676, mean=0.25600000000000006, count=3.0, positive=2, stdDev=1.2713499387134397, zeros=0}
    Output: [ 0.7680000000000001 ]
    Outputs Statistics: {meanExponent=-0.11463877996848792, negative=0, min=0.7680000000000001, max=0.7680000000000001, mean=0.7680000000000001, count=1.0, positive=1, stdDev=0.0, zeros=0}
    Feedback for input 0
    Inputs Values: [ -1.468, 1.56, 0.676 ]
    Value Statistics: {meanExponent=0.06326578329204975, negative=1, min=0.676, max=0.676, mean=0.25600000000000006, count=3.0, positive=2, stdDev=1.2713499387134397, zeros=0}
    Implemented Feedback: [ [ 1.0 ], [ 1.0 ], [ 1.0 ] ]
    Implemented Statistics: {meanExponent=0.0, negative=0, min=1.0, max=1.0, mean=1.0, count=3.0, positive=3, stdDev=0.0, zeros=0}
    Measured Feedback: [ [ 0.9999999999998899 ], [ 0.9999999999998899 ], [ 0.9999999999998899 ] ]
    Measured Statistics: {meanExponent=-4.7830642341045674E-14, negative=0, min=0.9999999999998899, max=0.9999999999998899, mean=0.9999999999998899, count=3.0, positive=3, stdDev=0.0, zeros=0}
    Feedback Error: [ [ -1.1013412404281553E-13 ], [ -1.1013412404281553E-13 ], [ -1.1013412404281553E-13 ] ]
    Error Statistics: {meanExponent=-12.958078098036827, negative=3, min=-1.1013412404281553E-13, max=-1.1013412404281553E-13, mean=-1.1013412404281553E-13, count=3.0, positive=0, stdDev=0.0, zeros=0}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 1.1013e-13 +- 0.0000e+00 [1.1013e-13 - 1.1013e-13] (3#)
    relativeTol: 5.5067e-14 +- 0.0000e+00 [5.5067e-14 - 5.5067e-14] (3#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=1.1013e-13 +- 0.0000e+00 [1.1013e-13 - 1.1013e-13] (3#), relativeTol=5.5067e-14 +- 0.0000e+00 [5.5067e-14 - 5.5067e-14] (3#)}
```



### Performance
Code from [LayerTestBase.java:192](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L192) executed in 0.00 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 0.1616 +- 0.0650 [0.1055 - 0.6669]
    Learning performance: 0.0021 +- 0.0022 [0.0000 - 0.0171]
    
```

