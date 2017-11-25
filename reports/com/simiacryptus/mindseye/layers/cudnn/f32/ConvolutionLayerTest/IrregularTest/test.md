### Json Serialization
Code from [LayerTestBase.java:57](../../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L57) executed in 0.00 seconds: 
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
      "class": "com.simiacryptus.mindseye.layers.cudnn.f32.ConvolutionLayer",
      "id": "9d13704a-9a5a-4ecb-a687-5c7c00008a15",
      "isFrozen": false,
      "name": "ConvolutionLayer/9d13704a-9a5a-4ecb-a687-5c7c00008a15",
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
Code from [LayerTestBase.java:74](../../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L74) executed in 0.63 seconds: 
```java
    getDerivativeTester().test(getLayer(), outputPrototype, inputPrototype);
```
Logging: 
```
    Finite-Difference Derivative Accuracy:
    absoluteTol: 3.9954e-09 +- 1.4801e-08 [0.0000e+00 - 1.2410e-07] (5832#)
    relativeTol: 4.3200e-01 +- 8.2303e-01 [3.7973e-09 - 2.0000e+00] (625#)
    
```

### Performance
Code from [LayerTestBase.java:79](../../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L79) executed in 49.72 seconds: 
```java
    getPerformanceTester().test(getLayer(), outputPrototype, inputPrototype);
```
Logging: 
```
    Forward performance: 2707.9869 +- 452.1681 [2481.1729 - 4417.1451]
    Backward performance: 2264.1189 +- 416.7957 [2026.7977 - 4151.3714]
    
```

### Reference Implementation
Code from [LayerTestBase.java:86](../../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L86) executed in 1.05 seconds: 
```java
    System.out.println(new GsonBuilder().setPrettyPrinting().create().toJson(referenceLayer.getJson()));
    getEquivalencyTester().test(referenceLayer, getLayer(), outputPrototype, inputPrototype);
```
Logging: 
```
    {
      "class": "com.simiacryptus.mindseye.layers.aparapi.ConvolutionLayer",
      "id": "9d13704a-9a5a-4ecb-a687-5c7c00016150",
      "isFrozen": false,
      "name": "ConvolutionLayer/9d13704a-9a5a-4ecb-a687-5c7c00016150",
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
    absoluteTol: 6.5096e-02 +- 2.0119e-01 [0.0000e+00 - 9.9985e-01] (5832#)
    relativeTol: 1.7778e+00 +- 6.2854e-01 [2.6969e-09 - 2.0000e+00] (882#)
    
```

