# SchemaOutputLayer
## SchemaOutputLayerTest
### Json Serialization
Code from [LayerTestBase.java:83](../../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L83) executed in 0.00 seconds: 
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
      "id": "a864e734-2f23-44db-97c1-504000000401",
      "isFrozen": false,
      "name": "SchemaOutputLayer/a864e734-2f23-44db-97c1-504000000401",
      "inputBands": 2,
      "logWeightInit": -3.0,
      "selected": [
        "test1",
        "test2"
      ],
      "features": {
        "test2": [
          3.310386556457525E-4,
          6.165363387301148E-5
        ],
        "test1": [
          4.847953177210106E-4,
          6.165363387301148E-5
        ]
      }
    }
```



