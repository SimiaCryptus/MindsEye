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
      "id": "c88cbdf1-1c2a-4a5e-b964-8909000002f7",
      "isFrozen": false,
      "name": "SchemaOutputLayer/c88cbdf1-1c2a-4a5e-b964-8909000002f7",
      "inputBands": 2,
      "logWeightInit": -3.0,
      "selected": [
        "test1",
        "test2"
      ],
      "features": {
        "test2": [
          4.999400373278935E-4,
          -4.997601636936622E-4
        ],
        "test1": [
          -4.961672190492596E-4,
          -4.990408848604641E-4
        ]
      }
    }
```



