# BinaryEntropyActivationLayer
## BinaryEntropyActivationLayerTest
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
      "class": "com.simiacryptus.mindseye.layers.java.BinaryEntropyActivationLayer",
      "id": "f4569375-56fe-4e46-925c-95f40000098f",
      "isFrozen": true,
      "name": "BinaryEntropyActivationLayer/f4569375-56fe-4e46-925c-95f40000098f"
    }
```



