# VariableLayer
## VariableLayerTest
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
      "class": "com.simiacryptus.mindseye.layers.java.VariableLayer",
      "id": "a864e734-2f23-44db-97c1-504000002cbe",
      "isFrozen": false,
      "name": "VariableLayer/a864e734-2f23-44db-97c1-504000002cbe",
      "inner": {
        "class": "com.simiacryptus.mindseye.layers.java.MonitoringSynapse",
        "id": "a864e734-2f23-44db-97c1-504000002cbd",
        "isFrozen": false,
        "name": "MonitoringSynapse/a864e734-2f23-44db-97c1-504000002cbd",
        "totalBatches": 0,
        "totalItems": 0
      }
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
    [[ 1.544, 0.032, -1.24 ]]
    --------------------
    Output: 
    [ 1.544, 0.032, -1.24 ]
```



### Batch Execution
Code from [LayerTestBase.java:138](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L138) executed in 0.00 seconds: 
```java
    BatchingTester batchingTester = getBatchingTester();
    return batchingTester==null?null:batchingTester.test(layer, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (60#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (60#)}
```



### Differential Validation
Code from [LayerTestBase.java:144](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.00 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [ 1.544, 0.032, -1.24 ]
    Inputs Statistics: {meanExponent=-0.40426034683938045, negative=1, min=-1.24, max=-1.24, mean=0.11200000000000003, count=3.0, positive=2, stdDev=1.1379701226306427, zeros=0}
    Output: [ 1.544, 0.032, -1.24 ]
    Outputs Statistics: {meanExponent=-0.40426034683938045, negative=1, min=-1.24, max=-1.24, mean=0.11200000000000003, count=3.0, positive=2, stdDev=1.1379701226306427, zeros=0}
    Feedback for input 0
    Inputs Values: [ 1.544, 0.032, -1.24 ]
    Value Statistics: {meanExponent=-0.40426034683938045, negative=1, min=-1.24, max=-1.24, mean=0.11200000000000003, count=3.0, positive=2, stdDev=1.1379701226306427, zeros=0}
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
Code from [LayerTestBase.java:149](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L149) executed in 0.00 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 0.1233 +- 0.0364 [0.0969 - 0.4474]
    Learning performance: 0.0150 +- 0.0290 [0.0057 - 0.2964]
    
```

