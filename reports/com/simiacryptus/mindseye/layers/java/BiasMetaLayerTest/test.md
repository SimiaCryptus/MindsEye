# BiasMetaLayer
## BiasMetaLayerTest
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
      "class": "com.simiacryptus.mindseye.layers.java.BiasMetaLayer",
      "id": "e2a3bda5-e7e7-4c05-aeb3-4ede00001e19",
      "isFrozen": false,
      "name": "BiasMetaLayer/e2a3bda5-e7e7-4c05-aeb3-4ede00001e19"
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
    [[ 1.032, 1.352, 0.096 ],
    [ 1.156, -0.408, -0.984 ]]
    --------------------
    Output: 
    [ 2.1879999999999997, 0.9440000000000002, -0.888 ]
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
    Inputs: [ 1.032, 1.352, 0.096 ],
    [ 1.156, -0.408, -0.984 ]
    Inputs Statistics: {meanExponent=-0.29102412602120725, negative=0, min=0.096, max=0.096, mean=0.8266666666666668, count=3.0, positive=3, stdDev=0.5329198397090837, zeros=0},
    {meanExponent=-0.11112896813142277, negative=2, min=-0.984, max=-0.984, mean=-0.07866666666666666, count=3.0, positive=1, stdDev=0.9041553455512437, zeros=0}
    Output: [ 2.1879999999999997, 0.9440000000000002, -0.888 ]
    Outputs Statistics: {meanExponent=0.08781075924602107, negative=1, min=-0.888, max=-0.888, mean=0.7479999999999999, count=3.0, positive=2, stdDev=1.2633964803919104, zeros=0}
    Feedback for input 0
    Inputs Values: [ 1.032, 1.352, 0.096 ]
    Value Statistics: {meanExponent=-0.29102412602120725, negative=0, min=0.096, max=0.096, mean=0.8266666666666668, count=3.0, positive=3, stdDev=0.5329198397090837, zeros=0}
    Implemented Feedback: [ [ 1.0, 0.0, 0.0 ], [ 0.0, 1.0, 0.0 ], [ 0.0, 0.0, 1.0 ] ]
    Implemented Statistics: {meanExponent=0.0, negative=0, min=1.0, max=1.0, mean
```
...[skipping 1080 bytes](etc/47.txt)...
```
    ] ]
    Implemented Statistics: {meanExponent=0.0, negative=0, min=1.0, max=1.0, mean=0.3333333333333333, count=9.0, positive=3, stdDev=0.4714045207910317, zeros=6}
    Measured Feedback: [ [ 1.0000000000021103, 0.0, 0.0 ], [ 0.0, 0.9999999999998899, 0.0 ], [ 0.0, 0.0, 0.9999999999998899 ] ]
    Measured Statistics: {meanExponent=2.7361184650972856E-13, negative=0, min=0.9999999999998899, max=0.9999999999998899, mean=0.3333333333335433, count=9.0, positive=3, stdDev=0.4714045207913287, zeros=6}
    Feedback Error: [ [ 2.1103119252074976E-12, 0.0, 0.0 ], [ 0.0, -1.1013412404281553E-13, 0.0 ], [ 0.0, 0.0, -1.1013412404281553E-13 ] ]
    Error Statistics: {meanExponent=-12.530603180987853, negative=2, min=-1.1013412404281553E-13, max=-1.1013412404281553E-13, mean=2.1000485301354073E-13, count=9.0, positive=1, stdDev=6.733627986644662E-13, zeros=6}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 2.5895e-13 +- 6.5610e-13 [0.0000e+00 - 2.1103e-12] (18#)
    relativeTol: 3.8843e-13 +- 4.7145e-13 [5.5067e-14 - 1.0552e-12] (6#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=2.5895e-13 +- 6.5610e-13 [0.0000e+00 - 2.1103e-12] (18#), relativeTol=3.8843e-13 +- 4.7145e-13 [5.5067e-14 - 1.0552e-12] (6#)}
```



### Performance
Code from [LayerTestBase.java:192](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L192) executed in 0.00 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 0.2401 +- 0.1860 [0.1311 - 1.4363]
    Learning performance: 0.0042 +- 0.0035 [0.0000 - 0.0257]
    
```

