# MonitoringWrapperLayer
## MonitoringWrapperTest
### Json Serialization
Code from [LayerTestBase.java:75](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L75) executed in 0.00 seconds: 
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
      "id": "b385277b-2d2d-42fe-8250-210c0000ed2a",
      "isFrozen": false,
      "name": "MonitoringSynapse/b385277b-2d2d-42fe-8250-210c0000ed29",
      "inner": {
        "class": "com.simiacryptus.mindseye.layers.java.MonitoringSynapse",
        "id": "b385277b-2d2d-42fe-8250-210c0000ed29",
        "isFrozen": false,
        "name": "MonitoringSynapse/b385277b-2d2d-42fe-8250-210c0000ed29",
        "totalBatches": 0,
        "totalItems": 0
      },
      "totalBatches": 0,
      "totalItems": 0,
      "recordSignalMetrics": true
    }
```



### Example Input/Output Pair
Code from [LayerTestBase.java:112](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L112) executed in 0.00 seconds: 
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
    [[ -0.72, 1.448, -0.712 ]]
    --------------------
    Output: 
    [ -0.72, 1.448, -0.712 ]
```



### Differential Validation
Code from [LayerTestBase.java:130](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L130) executed in 0.00 seconds: 
```java
    getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Finite-Difference Derivative Accuracy:
    absoluteTol: 1.5531e-11 +- 2.6354e-11 [0.0000e+00 - 8.2267e-11] (9#)
    relativeTol: 2.3296e-11 +- 1.2613e-11 [1.4378e-11 - 4.1133e-11] (3#)
    
```

### Performance
Code from [LayerTestBase.java:135](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L135) executed in 0.05 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 0.0197 +- 0.0235 [0.0142 - 1.4363]
    Learning performance: 0.0110 +- 0.0150 [0.0057 - 0.6583]
    
```

