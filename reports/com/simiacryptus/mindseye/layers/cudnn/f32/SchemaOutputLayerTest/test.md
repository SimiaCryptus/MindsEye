# SchemaOutputLayer
## SchemaOutputLayerTest
### Json Serialization
Code from [LayerTestBase.java:84](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L84) executed in 0.00 seconds: 
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
      "id": "f4569375-56fe-4e46-925c-95f40000014a",
      "isFrozen": false,
      "name": "SchemaOutputLayer/f4569375-56fe-4e46-925c-95f40000014a",
      "inputBands": 2,
      "logWeightInit": -3.0,
      "selected": [
        "test1",
        "test2"
      ],
      "features": {
        "test2": [
          4.6016488281229386E-4,
          -3.4700687749460846E-4
        ],
        "test1": [
          1.8344907885767148E-5,
          -3.4700687749460846E-4
        ]
      }
    }
```



