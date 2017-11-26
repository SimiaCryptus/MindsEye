# SimpleConvolutionLayer
## SimpleConvolutionLayerTest
### Json Serialization
Code from [LayerTestBase.java:76](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L76) executed in 0.07 seconds: 
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
      "class": "com.simiacryptus.mindseye.layers.cudnn.f64.SimpleConvolutionLayer",
      "id": "912c09f6-07ff-413f-b76b-c60300000001",
      "isFrozen": false,
      "name": "SimpleConvolutionLayer/912c09f6-07ff-413f-b76b-c60300000001",
      "filter": {
        "dimensions": [
          3,
          3,
          1
        ],
        "data": [
          0.8763615443904953,
          -0.7616221305132784,
          -0.782228848938302,
          0.1431678160236316,
          -0.8932794017500711,
          -0.3696802287210845,
          0.44418625152980384,
          -0.9734581081848948,
          -0.5523595677502988
        ]
      },
      "strideX": 1,
      "strideY": 1,
      "simple": false
    }
```



### Differential Validation
Code from [LayerTestBase.java:100](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L100) executed in 0.28 seconds: 
```java
    getDerivativeTester().test(layer, outputPrototype, inputPrototype);
```
Logging: 
```
    Finite-Difference Derivative Accuracy:
    absoluteTol: 8.0824e-11 +- 1.0175e-10 [0.0000e+00 - 4.3733e-10] (162#)
    relativeTol: 1.0641e-10 +- 1.3606e-10 [1.0504e-12 - 8.5653e-10] (98#)
    
```

### Performance
Code from [LayerTestBase.java:105](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L105) executed in 8.48 seconds: 
```java
    getPerformanceTester().test(layer, outputPrototype, inputPrototype);
```
Logging: 
```
    Forward performance: 485.9235 +- 32.7920 [400.2749 - 604.3679]
    Backward performance: 360.7159 +- 19.1671 [318.2496 - 410.7223]
    
```

### Reference Implementation
Code from [LayerTestBase.java:124](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L124) executed in 1.72 seconds: 
```java
    System.out.println(new GsonBuilder().setPrettyPrinting().create().toJson(referenceLayer.getJson()));
    getEquivalencyTester().test(referenceLayer, layer, outputPrototype, inputPrototype);
```
Logging: 
```
    {
      "class": "com.simiacryptus.mindseye.layers.aparapi.ConvolutionLayer",
      "id": "912c09f6-07ff-413f-b76b-c60300000002",
      "isFrozen": false,
      "name": "ConvolutionLayer/912c09f6-07ff-413f-b76b-c60300000002",
      "filter": {
        "dimensions": [
          3,
          3,
          1
        ],
        "data": [
          0.8763615443904953,
          -0.7616221305132784,
          -0.782228848938302,
          0.1431678160236316,
          -0.8932794017500711,
          -0.3696802287210845,
          0.44418625152980384,
          -0.9734581081848948,
          -0.5523595677502988
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
    absoluteTol: 0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (162#)
    relativeTol: 0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (98#)
    
```

