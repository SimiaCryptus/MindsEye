### Json Serialization
Code from [LayerTestBase.java:57](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L57) executed in 0.04 seconds: 
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
      "class": "com.simiacryptus.mindseye.layers.aparapi.ConvolutionLayer",
      "id": "9d13704a-9a5a-4ecb-a687-5c7c00000001",
      "isFrozen": false,
      "name": "ConvolutionLayer/9d13704a-9a5a-4ecb-a687-5c7c00000001",
      "filter": {
        "dimensions": [
          3,
          3,
          4
        ]
      },
      "skip": {
        "dimensions": [
          1,
          1
        ]
      },
      "simple": true
    }
```



### Differential Validation
Code from [LayerTestBase.java:74](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L74) executed in 0.77 seconds: 
```java
    getDerivativeTester().test(getLayer(), outputPrototype, inputPrototype);
```
Logging: 
```
    Finite-Difference Derivative Accuracy:
    absoluteTol: 2.1131e-18 +- 1.4324e-17 [0.0000e+00 - 1.1102e-16] (972#)
    relativeTol: 2.2098e-17 +- 5.0341e-17 [0.0000e+00 - 1.4384e-16] (196#)
    
```

### Performance
Code from [LayerTestBase.java:79](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L79) executed in 20.10 seconds: 
```java
    getPerformanceTester().test(getLayer(), outputPrototype, inputPrototype);
```
Logging: 
```
    Forward performance: 1198.3191 +- 86.4272 [1110.7118 - 1556.5520]
    Backward performance: 810.7284 +- 69.5233 [752.0352 - 1155.0317]
    
```

### Reference Implementation
