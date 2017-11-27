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
      "id": "79cb4e2c-4a9a-4706-8f49-c2d500000063",
      "isFrozen": false,
      "name": "SchemaOutputLayer/79cb4e2c-4a9a-4706-8f49-c2d500000063",
      "inputBands": 2,
      "logWeightInit": -3.0,
      "selected": [
        "test1",
        "test2"
      ],
      "features": {
        "test2": [
          -2.8363853626127547E-4,
          1.781967229902446E-4
        ],
        "test1": [
          4.426168027372197E-4,
          4.426168027372197E-4
        ]
      }
    }
```



