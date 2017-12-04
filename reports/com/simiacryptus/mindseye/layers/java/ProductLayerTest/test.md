# ProductLayer
## ProductLayerTest
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
      "class": "com.simiacryptus.mindseye.layers.java.ProductLayer",
      "id": "a864e734-2f23-44db-97c1-504000002c7c",
      "isFrozen": false,
      "name": "ProductLayer/a864e734-2f23-44db-97c1-504000002c7c"
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
    [[ 0.652, 1.808, -1.012 ]]
    --------------------
    Output: 
    [ -1.1929617920000002 ]
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
    Inputs: [ 0.652, 1.808, -1.012 ]
    Inputs Statistics: {meanExponent=0.025542178125015014, negative=1, min=-1.012, max=-1.012, mean=0.48266666666666663, count=3.0, positive=2, stdDev=1.1574700380091438, zeros=0}
    Output: [ -1.1929617920000002 ]
    Outputs Statistics: {meanExponent=0.07662653437504507, negative=1, min=-1.1929617920000002, max=-1.1929617920000002, mean=-1.1929617920000002, count=1.0, positive=0, stdDev=0.0, zeros=0}
    Feedback for input 0
    Inputs Values: [ 0.652, 1.808, -1.012 ]
    Value Statistics: {meanExponent=0.025542178125015014, negative=1, min=-1.012, max=-1.012, mean=0.48266666666666663, count=3.0, positive=2, stdDev=1.1574700380091438, zeros=0}
    Implemented Feedback: [ [ -1.8296960000000002 ], [ -0.6598240000000001 ], [ 1.178816 ] ]
    Implemented Statistics: {meanExponent=0.05108435625003005, negative=2, min=1.178816, max=1.178816, mean=-0.43690133333333336, count=3.0, positive=1, stdDev=1.238293718528668, zeros=0}
    Measured Feedback: [ [ -1.8296959999974938 ], [ -0.6598239999977551 ], [ 1.17881600000036 ] ]
    Measured Statistics: {meanExponent=0.05108435624938339, negative=2, min=1.17881600000036, max=1.17881600000036, mean=-0.4369013333316296, count=3.0, positive=1, stdDev=1.23829371852775, zeros=0}
    Feedback Error: [ [ 2.5064395003937534E-12 ], [ 2.244981978094529E-12 ], [ 3.5993430458347575E-13 ] ]
    Error Statistics: {meanExponent=-11.897835558155114, negative=0, min=3.5993430458347575E-13, max=3.5993430458347575E-13, mean=1.7037852610239195E-12, count=3.0, positive=3, stdDev=9.562222732663969E-13, zeros=0}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 1.7038e-12 +- 9.5622e-13 [3.5993e-13 - 2.5064e-12] (3#)
    relativeTol: 8.4627e-13 +- 6.4240e-13 [1.5267e-13 - 1.7012e-12] (3#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=1.7038e-12 +- 9.5622e-13 [3.5993e-13 - 2.5064e-12] (3#), relativeTol=8.4627e-13 +- 6.4240e-13 [1.5267e-13 - 1.7012e-12] (3#)}
```



### Performance
Code from [LayerTestBase.java:149](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L149) executed in 0.00 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 0.1304 +- 0.0344 [0.1054 - 0.3562]
    Learning performance: 0.0021 +- 0.0023 [0.0000 - 0.0199]
    
```

