### Json Serialization
Code from [LayerTestBase.java:57](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L57) executed in 0.00 seconds: 
```java
    NNLayer layer = getLayer();
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
      "class": "com.simiacryptus.mindseye.layers.cudnn.f32.SchemaBiasLayer",
      "id": "9d13704a-9a5a-4ecb-a687-5c7c00016384",
      "isFrozen": false,
      "name": "SchemaBiasLayer/9d13704a-9a5a-4ecb-a687-5c7c00016384",
      "selected": [
        "test1",
        "test2"
      ],
      "features": {
        "test2": 0.0,
        "test1": 0.0
      }
    }
```



### Differential Validation
Code from [LayerTestBase.java:74](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L74) executed in 0.06 seconds: 
```java
    getDerivativeTester().test(getLayer(), outputPrototype, inputPrototype);
```
Logging: 
```
    Finite-Difference Derivative Accuracy:
    absoluteTol: 1.4337e-05 +- 4.8032e-05 [0.0000e+00 - 4.3011e-04] (360#)
    relativeTol: 1.4336e-04 +- 6.7628e-05 [1.6928e-05 - 4.3020e-04] (36#)
    
```

### Performance
Code from [LayerTestBase.java:79](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L79) executed in 2.64 seconds: 
```java
    getPerformanceTester().test(getLayer(), outputPrototype, inputPrototype);
```
Logging: 
```
    Forward performance: 140.3421 +- 34.8070 [112.0649 - 244.8506]
    Backward performance: 123.3891 +- 5.9450 [110.7113 - 158.6047]
    
```

### Reference Implementation
