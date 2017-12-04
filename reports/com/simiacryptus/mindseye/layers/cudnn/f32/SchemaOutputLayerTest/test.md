# SchemaOutputLayer
## SchemaOutputLayerTest
### Json Serialization
Code from [LayerTestBase.java:83](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L83) executed in 0.00 seconds: 
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
      "class": "com.simiacryptus.mindseye.layers.cudnn.f32.SchemaOutputLayer",
      "id": "370a9587-74a1-4959-b406-fa4500000401",
      "isFrozen": false,
      "name": "SchemaOutputLayer/370a9587-74a1-4959-b406-fa4500000401",
      "inputBands": 2,
      "logWeightInit": -3.0,
      "selected": [
        "test1",
        "test2"
      ],
      "features": {
        "test2": [
          2.303187849077335E-4,
          2.8781302927450084E-4
        ],
        "test1": [
          2.8781302927450084E-4,
          2.8781302927450084E-4
        ]
      }
    }
```



