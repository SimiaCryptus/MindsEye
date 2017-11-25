### Json Serialization
Code from [LayerTestBase.java:74](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L74) executed in 0.06 seconds: 
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
      "class": "com.simiacryptus.mindseye.layers.aparapi.ConvolutionLayer",
      "id": "bdd6bbba-380b-47fe-a761-c24100000001",
      "isFrozen": false,
      "name": "ConvolutionLayer/bdd6bbba-380b-47fe-a761-c24100000001",
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
Code from [LayerTestBase.java:98](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L98) executed in 2.00 seconds: 
```java
    getDerivativeTester().test(layer, outputPrototype, inputPrototype);
```
Logging: 
```
    Finite-Difference Derivative Accuracy:
    absoluteTol: 8.3952e-18 +- 3.9030e-17 [0.0000e+00 - 2.2204e-16] (972#)
    relativeTol: 5.7287e-17 +- 7.9585e-17 [0.0000e+00 - 2.0132e-16] (196#)
    
```

### Performance
Code from [LayerTestBase.java:103](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L103) executed in 69.47 seconds: 
```java
    getPerformanceTester().test(layer, outputPrototype, inputPrototype);
```
Logging: 
```
    Forward performance: 4169.8569 +- 111.4655 [3875.3869 - 4603.6777]
    Backward performance: 2775.4182 +- 59.5984 [2602.0322 - 2923.1829]
    
```

### Reference Implementation
