# ImgCropLayer
## ImgCropLayerTest
### Json Serialization
Code from [LayerTestBase.java:76](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L76) executed in 0.09 seconds: 
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
      "class": "com.simiacryptus.mindseye.layers.java.ImgCropLayer",
      "id": "4d8ea217-c310-4d9a-bc56-248800000001",
      "isFrozen": false,
      "name": "ImgCropLayer/4d8ea217-c310-4d9a-bc56-248800000001",
      "sizeX": 1,
      "sizeY": 1
    }
```



### Differential Validation
Code from [LayerTestBase.java:100](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L100) executed in 0.05 seconds: 
```java
    getDerivativeTester().test(layer, outputPrototype, inputPrototype);
```
Logging: 
```
    Finite-Difference Derivative Accuracy:
    absoluteTol: 9.1407e-12 +- 2.5854e-11 [0.0000e+00 - 8.2267e-11] (9#)
    relativeTol: 8.2267e-11 +- 0.0000e+00 [8.2267e-11 - 8.2267e-11] (1#)
    
```

### Performance
Code from [LayerTestBase.java:105](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L105) executed in 0.06 seconds: 
```java
    getPerformanceTester().test(layer, outputPrototype, inputPrototype);
```
Logging: 
```
    Forward performance: 4.5013 +- 15.9862 [1.5674 - 163.1587]
    
```

### Reference Implementation
