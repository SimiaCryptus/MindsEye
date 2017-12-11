# MonitoringSynapse
## MonitoringSynapseTest
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
      "class": "com.simiacryptus.mindseye.layers.java.MonitoringSynapse",
      "id": "e2a3bda5-e7e7-4c05-aeb3-4ede00001ef2",
      "isFrozen": false,
      "name": "MonitoringSynapse/e2a3bda5-e7e7-4c05-aeb3-4ede00001ef2",
      "totalBatches": 0,
      "totalItems": 0
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
    [[ 1.2, -0.032, -0.588 ]]
    --------------------
    Output: 
    [ 1.2, -0.032, -0.588 ]
```



### Batch Execution
Code from [LayerTestBase.java:178](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L178) executed in 0.00 seconds: 
```java
    BatchingTester batchingTester = getBatchingTester();
    return batchingTester == null ? null : batchingTester.test(layer, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (60#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (60#)}
```



### Differential Validation
Code from [LayerTestBase.java:186](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L186) executed in 0.00 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [ 1.2, -0.032, -0.588 ]
    Inputs Statistics: {meanExponent=-0.5487638165187768, negative=2, min=-0.588, max=-0.588, mean=0.19333333333333333, count=3.0, positive=1, stdDev=0.747135567052965, zeros=0}
    Output: [ 1.2, -0.032, -0.588 ]
    Outputs Statistics: {meanExponent=-0.5487638165187768, negative=2, min=-0.588, max=-0.588, mean=0.19333333333333333, count=3.0, positive=1, stdDev=0.747135567052965, zeros=0}
    Feedback for input 0
    Inputs Values: [ 1.2, -0.032, -0.588 ]
    Value Statistics: {meanExponent=-0.5487638165187768, negative=2, min=-0.588, max=-0.588, mean=0.19333333333333333, count=3.0, positive=1, stdDev=0.747135567052965, zeros=0}
    Implemented Feedback: [ [ 1.0, 0.0, 0.0 ], [ 0.0, 1.0, 0.0 ], [ 0.0, 0.0, 1.0 ] ]
    Implemented Statistics: {meanExponent=0.0, negative=0, min=1.0, max=1.0, mean=0.3333333333333333, count=9.0, positive=3, stdDev=0.4714045207910317, zeros=6}
    Measured Feedback: [ [ 0.9999999999998899, 0.0, 0.0 ], [ 0.0, 1.0000000000000286, 0.0 ], [ 0.0, 0.0, 0.9999999999998899 ] ]
    Measured Statistics: {meanExponent=-2.7740486787851373E-14, negative=0, min=0.9999999999998899, max=0.9999999999998899, mean=0.333333333333312, count=9.0, positive=3, stdDev=0.4714045207910016, zeros=6}
    Feedback Error: [ [ -1.1013412404281553E-13, 0.0, 0.0 ], [ 0.0, 2.864375403532904E-14, 0.0 ], [ 0.0, 0.0, -1.1013412404281553E-13 ] ]
    Error Statistics: {meanExponent=-13.153042086767142, negative=2, min=-1.1013412404281553E-13, max=-1.1013412404281553E-13, mean=-2.1291610450033557E-14, count=9.0, positive=1, stdDev=4.8304038389477654E-14, zeros=6}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 2.7657e-14 +- 4.4963e-14 [0.0000e+00 - 1.1013e-13] (9#)
    relativeTol: 4.1485e-14 +- 1.9207e-14 [1.4322e-14 - 5.5067e-14] (3#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=2.7657e-14 +- 4.4963e-14 [0.0000e+00 - 1.1013e-13] (9#), relativeTol=4.1485e-14 +- 1.9207e-14 [1.4322e-14 - 5.5067e-14] (3#)}
```



### Performance
Code from [LayerTestBase.java:192](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L192) executed in 0.00 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 0.1529 +- 0.0432 [0.1225 - 0.4560]
    Learning performance: 0.0234 +- 0.0227 [0.0114 - 0.1881]
    
```

