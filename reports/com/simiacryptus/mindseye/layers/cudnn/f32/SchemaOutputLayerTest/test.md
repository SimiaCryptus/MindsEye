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
      "class": "com.simiacryptus.mindseye.layers.cudnn.f32.SchemaOutputLayer",
      "id": "9d13704a-9a5a-4ecb-a687-5c7c0001638a",
      "isFrozen": false,
      "name": "SchemaOutputLayer/9d13704a-9a5a-4ecb-a687-5c7c0001638a",
      "inputBands": 2,
      "logWeightInit": -3.0,
      "selected": [
        "test1",
        "test2"
      ],
      "features": {
        "test2": [
          4.986790958713369E-4,
          4.986790958713369E-4
        ],
        "test1": [
          -4.947233626362164E-4,
          -4.947233626362164E-4
        ]
      }
    }
```



