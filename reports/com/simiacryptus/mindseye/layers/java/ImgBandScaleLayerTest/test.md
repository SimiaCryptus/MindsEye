### Json Serialization
Code from [LayerTestBase.java:57](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L57) executed in 0.00 seconds: 
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
      "class": "com.simiacryptus.mindseye.layers.java.ImgBandScaleLayer",
      "id": "9d13704a-9a5a-4ecb-a687-5c7c0002dd20",
      "isFrozen": false,
      "name": "ImgBandScaleLayer/9d13704a-9a5a-4ecb-a687-5c7c0002dd20",
      "bias": [
        3.0
      ]
    }
```



