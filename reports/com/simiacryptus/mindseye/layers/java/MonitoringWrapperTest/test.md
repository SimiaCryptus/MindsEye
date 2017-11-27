# MonitoringWrapperLayer
## MonitoringWrapperTest
### Json Serialization
Code from [LayerTestBase.java:84](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L84) executed in 0.00 seconds: 
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
      "class": "com.simiacryptus.mindseye.layers.java.MonitoringWrapperLayer",
      "id": "0910987d-3688-428c-a892-e2c400000e68",
      "isFrozen": false,
      "name": "MonitoringSynapse/0910987d-3688-428c-a892-e2c400000e67",
      "inner": {
        "class": "com.simiacryptus.mindseye.layers.java.MonitoringSynapse",
        "id": "0910987d-3688-428c-a892-e2c400000e67",
        "isFrozen": false,
        "name": "MonitoringSynapse/0910987d-3688-428c-a892-e2c400000e67",
        "totalBatches": 0,
        "totalItems": 0
      },
      "totalBatches": 0,
      "totalItems": 0,
      "recordSignalMetrics": true
    }
```



### Example Input/Output Pair
Code from [LayerTestBase.java:121](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L121) executed in 0.00 seconds: 
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
    [[ 0.672, -1.052, 1.512 ]]
    --------------------
    Output: 
    [ 0.672, -1.052, 1.512 ]
```



### Differential Validation
Code from [LayerTestBase.java:139](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L139) executed in 0.00 seconds: 
```java
    getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Finite-Difference Derivative Accuracy:
    absoluteTol: 1.9615e-12 +- 2.8268e-12 [0.0000e+00 - 6.5510e-12] (9#)
    relativeTol: 2.9422e-12 +- 4.7137e-13 [2.2756e-12 - 3.2755e-12] (3#)
    
```

### Performance
Code from [LayerTestBase.java:144](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.00 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 0.0867 +- 0.0453 [0.0684 - 0.4731]
    Learning performance: 0.0513 +- 0.0122 [0.0342 - 0.1169]
    
```

