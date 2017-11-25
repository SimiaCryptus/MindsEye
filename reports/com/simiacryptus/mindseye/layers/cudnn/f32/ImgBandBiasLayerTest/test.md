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
      "class": "com.simiacryptus.mindseye.layers.cudnn.f32.ImgBandBiasLayer",
      "id": "9d13704a-9a5a-4ecb-a687-5c7c00016377",
      "isFrozen": false,
      "name": "ImgBandBiasLayer/9d13704a-9a5a-4ecb-a687-5c7c00016377",
      "bias": [
        0.0,
        0.0
      ]
    }
```



### Differential Validation
Code from [LayerTestBase.java:74](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L74) executed in 0.04 seconds: 
```java
    getDerivativeTester().test(getLayer(), outputPrototype, inputPrototype);
```
Logging: 
```
    Finite-Difference Derivative Accuracy:
    absoluteTol: 1.4318e-05 +- 4.8241e-05 [0.0000e+00 - 4.3011e-04] (360#)
    relativeTol: 1.4317e-04 +- 6.9453e-05 [1.6928e-05 - 4.3020e-04] (36#)
    
```

### Performance
Code from [LayerTestBase.java:79](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L79) executed in 2.45 seconds: 
```java
    getPerformanceTester().test(getLayer(), outputPrototype, inputPrototype);
```
Logging: 
```
    Forward performance: 123.2669 +- 6.2514 [112.7774 - 156.2822]
    Backward performance: 121.7155 +- 5.3203 [110.8709 - 136.7497]
    
```

### Reference Implementation
