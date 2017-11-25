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
      "class": "com.simiacryptus.mindseye.layers.cudnn.f64.ConvolutionLayer",
      "id": "9d13704a-9a5a-4ecb-a687-5c7c0002038b",
      "isFrozen": false,
      "name": "ConvolutionLayer/9d13704a-9a5a-4ecb-a687-5c7c0002038b",
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
Code from [LayerTestBase.java:74](../../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L74) executed in 0.75 seconds: 
```java
    getDerivativeTester().test(getLayer(), outputPrototype, inputPrototype);
```
Logging: 
```
    Finite-Difference Derivative Accuracy:
    absoluteTol: 1.7133e-18 +- 1.1652e-17 [0.0000e+00 - 1.1102e-16] (5832#)
    relativeTol: 4.0424e-17 +- 5.6830e-17 [0.0000e+00 - 1.2893e-16] (490#)
    
```

### Performance
Code from [LayerTestBase.java:79](../../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L79) executed in 48.95 seconds: 
```java
    getPerformanceTester().test(getLayer(), outputPrototype, inputPrototype);
```
Logging: 
```
    Forward performance: 2661.4726 +- 582.8282 [2329.8180 - 4815.6418]
    Backward performance: 2233.1502 +- 439.8924 [1989.8787 - 3843.0048]
    
```

### Reference Implementation
Code from [LayerTestBase.java:86](../../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L86) executed in 1.03 seconds: 
```java
    System.out.println(new GsonBuilder().setPrettyPrinting().create().toJson(referenceLayer.getJson()));
    getEquivalencyTester().test(referenceLayer, getLayer(), outputPrototype, inputPrototype);
```
Logging: 
```
    {
      "class": "com.simiacryptus.mindseye.layers.aparapi.ConvolutionLayer",
      "id": "9d13704a-9a5a-4ecb-a687-5c7c0002dac6",
      "isFrozen": false,
      "name": "ConvolutionLayer/9d13704a-9a5a-4ecb-a687-5c7c0002dac6",
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
    absoluteTol: 5.6223e-02 +- 1.7150e-01 [0.0000e+00 - 9.1020e-01] (5832#)
    relativeTol: 1.7778e+00 +- 6.2854e-01 [0.0000e+00 - 2.0000e+00] (882#)
    
```

