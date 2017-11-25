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
      "class": "com.simiacryptus.mindseye.layers.cudnn.f32.SimpleConvolutionLayer",
      "id": "9d13704a-9a5a-4ecb-a687-5c7c0001638c",
      "isFrozen": false,
      "name": "SimpleConvolutionLayer/9d13704a-9a5a-4ecb-a687-5c7c0001638c",
      "filter": {
        "dimensions": [
          3,
          3,
          1
        ]
      },
      "simple": false,
      "strideX": 1,
      "strideY": 1
    }
```



### Differential Validation
Code from [LayerTestBase.java:74](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L74) executed in 0.04 seconds: 
```java
    getDerivativeTester().test(getLayer(), outputPrototype, inputPrototype);
```
Logging: 
```
    Finite-Difference Derivative Accuracy:
    absoluteTol: 1.2216e-08 +- 1.9893e-08 [0.0000e+00 - 8.2946e-08] (162#)
    relativeTol: 4.1935e-01 +- 8.1416e-01 [6.3865e-09 - 2.0000e+00] (62#)
    
```

### Performance
Code from [LayerTestBase.java:79](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L79) executed in 2.33 seconds: 
```java
    getPerformanceTester().test(getLayer(), outputPrototype, inputPrototype);
```
Logging: 
```
    Forward performance: 130.7871 +- 9.9996 [118.8588 - 196.2390]
    Backward performance: 102.3046 +- 4.1887 [95.1258 - 114.4445]
    
```

### Reference Implementation
Code from [LayerTestBase.java:86](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L86) executed in 0.13 seconds: 
```java
    System.out.println(new GsonBuilder().setPrettyPrinting().create().toJson(referenceLayer.getJson()));
    getEquivalencyTester().test(referenceLayer, getLayer(), outputPrototype, inputPrototype);
```
Logging: 
```
    {
      "class": "com.simiacryptus.mindseye.layers.aparapi.ConvolutionLayer",
      "id": "9d13704a-9a5a-4ecb-a687-5c7c00016390",
      "isFrozen": false,
      "name": "ConvolutionLayer/9d13704a-9a5a-4ecb-a687-5c7c00016390",
      "filter": {
        "dimensions": [
          3,
          3,
          1
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
    absoluteTol: 3.0365e-09 +- 6.0127e-09 [0.0000e+00 - 2.2609e-08] (162#)
    relativeTol: 2.2785e-08 +- 7.3390e-09 [7.0846e-09 - 3.3352e-08] (49#)
    
```

