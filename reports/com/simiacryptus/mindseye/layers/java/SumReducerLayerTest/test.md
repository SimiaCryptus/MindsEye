# SumReducerLayer
## SumReducerLayerTest
### Json Serialization
Code from [LayerTestBase.java:83](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L83) executed in 0.00 seconds: 
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
      "id": "a864e734-2f23-44db-97c1-504000002cbb",
      "isFrozen": false,
      "name": "SumReducerLayer/a864e734-2f23-44db-97c1-504000002cbb"
    }
```



### Example Input/Output Pair
Code from [LayerTestBase.java:120](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L120) executed in 0.00 seconds: 
```java
    SimpleEval eval = SimpleEval.run(layer, inputPrototype);
    return String.format("--------------------\nInput: \n[%s]\n--------------------\nOutput: \n%s",
      Arrays.stream(inputPrototype).map(t->t.prettyPrint()).reduce((a,b)->a+",\n"+b).get(),
      eval.getOutput().prettyPrint());
```

Returns: 

```
    --------------------
    Input: 
    [[ 1.432, 0.244, 0.764 ]]
    --------------------
    Output: 
    [ 2.44 ]
```



### Batch Execution
Code from [LayerTestBase.java:138](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L138) executed in 0.00 seconds: 
```java
    BatchingTester batchingTester = getBatchingTester();
    return batchingTester==null?null:batchingTester.test(layer, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (40#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (40#)}
```



### Differential Validation
Code from [LayerTestBase.java:144](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.00 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [ 1.432, 0.244, 0.764 ]
    Inputs Statistics: {meanExponent=-0.1911912657045813, negative=0, min=0.764, max=0.764, mean=0.8133333333333334, count=3.0, positive=3, stdDev=0.4862518780312205, zeros=0}
    Output: [ 2.44 ]
    Outputs Statistics: {meanExponent=0.3873898263387294, negative=0, min=2.44, max=2.44, mean=2.44, count=1.0, positive=1, stdDev=0.0, zeros=0}
    Feedback for input 0
    Inputs Values: [ 1.432, 0.244, 0.764 ]
    Value Statistics: {meanExponent=-0.1911912657045813, negative=0, min=0.764, max=0.764, mean=0.8133333333333334, count=3.0, positive=3, stdDev=0.4862518780312205, zeros=0}
    Implemented Feedback: [ [ 1.0 ], [ 1.0 ], [ 1.0 ] ]
    Implemented Statistics: {meanExponent=0.0, negative=0, min=1.0, max=1.0, mean=1.0, count=3.0, positive=3, stdDev=0.0, zeros=0}
    Measured Feedback: [ [ 1.0000000000021103 ], [ 1.0000000000021103 ], [ 1.0000000000021103 ] ]
    Measured Statistics: {meanExponent=9.16496824211277E-13, negative=0, min=1.0000000000021103, max=1.0000000000021103, mean=1.0000000000021103, count=3.0, positive=3, stdDev=0.0, zeros=0}
    Feedback Error: [ [ 2.1103119252074976E-12 ], [ 2.1103119252074976E-12 ], [ 2.1103119252074976E-12 ] ]
    Error Statistics: {meanExponent=-11.675653346889904, negative=0, min=2.1103119252074976E-12, max=2.1103119252074976E-12, mean=2.1103119252074976E-12, count=3.0, positive=3, stdDev=0.0, zeros=0}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 2.1103e-12 +- 0.0000e+00 [2.1103e-12 - 2.1103e-12] (3#)
    relativeTol: 1.0552e-12 +- 0.0000e+00 [1.0552e-12 - 1.0552e-12] (3#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=2.1103e-12 +- 0.0000e+00 [2.1103e-12 - 2.1103e-12] (3#), relativeTol=1.0552e-12 +- 0.0000e+00 [1.0552e-12 - 1.0552e-12] (3#)}
```



### Performance
Code from [LayerTestBase.java:149](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L149) executed in 0.00 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 0.1675 +- 0.0682 [0.1339 - 0.8179]
    Learning performance: 0.0021 +- 0.0015 [0.0000 - 0.0086]
    
```

