### Json Serialization
Code from [LayerTestBase.java:74](../../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L74) executed in 0.00 seconds: 
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
      "class": "com.simiacryptus.mindseye.layers.cudnn.f32.ConvolutionLayer",
      "id": "bdd6bbba-380b-47fe-a761-c24100008a01",
      "isFrozen": false,
      "name": "ConvolutionLayer/bdd6bbba-380b-47fe-a761-c24100008a01",
      "filter": {
        "dimensions": [
          3,
          3,
          10
        ]
      },
      "strideX": 1,
      "strideY": 1
    }
```



### Differential Validation
Code from [LayerTestBase.java:98](../../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L98) executed in 3.31 seconds: 
```java
    getDerivativeTester().test(layer, outputPrototype, inputPrototype);
```
Logging: 
```
    Finite-Difference Derivative Accuracy:
    absoluteTol: 5.6801e-09 +- 2.3221e-08 [0.0000e+00 - 2.2724e-07] (5832#)
    relativeTol: 2.6549e-01 +- 6.7859e-01 [0.0000e+00 - 2.0000e+00] (565#)
    
```

### Performance
Code from [LayerTestBase.java:103](../../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L103) executed in 242.39 seconds: 
```java
    getPerformanceTester().test(layer, outputPrototype, inputPrototype);
```
Logging: 
```
    Forward performance: 13650.1767 +- 2331.6298 [12291.8076 - 22959.3467]
    Backward performance: 10588.8949 +- 1906.7934 [9510.1564 - 18244.9634]
    
```

### Reference Implementation
Code from [LayerTestBase.java:122](../../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L122) executed in 4.54 seconds: 
```java
  
```
Logging: 
```
    {
      "class": "com.simiacryptus.mindseye.layers.aparapi.ConvolutionLayer",
      "id": "bdd6bbba-380b-47fe-a761-c24100016139",
      "isFrozen": false,
      "name": "ConvolutionLayer/bdd6bbba-380b-47fe-a761-c24100016139",
      "filter": {
        "dimensions": [
          3,
          3,
          10
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
    Reference Layer Accuracy:
    absoluteTol: 1.4623e-01 +- 4.1789e-01 [0.0000e+00 - 1.9657e+00] (5832#)
    relativeTol: 1.7778e+00 +- 6.2854e-01 [1.0356e-09 - 2.0000e+00] (882#)
    
```

