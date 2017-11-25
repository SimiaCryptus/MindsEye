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
      "id": "9d13704a-9a5a-4ecb-a687-5c7c00018f51",
      "isFrozen": false,
      "name": "ConvolutionLayer/9d13704a-9a5a-4ecb-a687-5c7c00018f51",
      "filter": {
        "dimensions": [
          3,
          3,
          8
        ]
      },
      "strideX": 1,
      "strideY": 1
    }
```



### Differential Validation
Code from [LayerTestBase.java:74](../../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L74) executed in 0.43 seconds: 
```java
    getDerivativeTester().test(getLayer(), outputPrototype, inputPrototype);
```
Logging: 
```
    Finite-Difference Derivative Accuracy:
    absoluteTol: 2.1245e-18 +- 1.3662e-17 [0.0000e+00 - 1.1102e-16] (3240#)
    relativeTol: 2.9706e-17 +- 5.2348e-17 [0.0000e+00 - 1.3879e-16] (392#)
    
```

### Performance
Code from [LayerTestBase.java:79](../../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L79) executed in 22.73 seconds: 
```java
    getPerformanceTester().test(getLayer(), outputPrototype, inputPrototype);
```
Logging: 
```
    Forward performance: 1226.0679 +- 325.1726 [1079.4440 - 2637.1017]
    Backward performance: 1046.9799 +- 243.6737 [925.8834 - 2126.4689]
    
```

### Reference Implementation
Code from [LayerTestBase.java:86](../../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L86) executed in 0.60 seconds: 
```java
    System.out.println(new GsonBuilder().setPrettyPrinting().create().toJson(referenceLayer.getJson()));
    getEquivalencyTester().test(referenceLayer, getLayer(), outputPrototype, inputPrototype);
```
Logging: 
```
    {
      "class": "com.simiacryptus.mindseye.layers.aparapi.ConvolutionLayer",
      "id": "9d13704a-9a5a-4ecb-a687-5c7c00020269",
      "isFrozen": false,
      "name": "ConvolutionLayer/9d13704a-9a5a-4ecb-a687-5c7c00020269",
      "filter": {
        "dimensions": [
          3,
          3,
          8
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
    absoluteTol: 1.1837e-01 +- 2.7406e-01 [0.0000e+00 - 9.8339e-01] (3240#)
    relativeTol: 1.7143e+00 +- 6.9985e-01 [0.0000e+00 - 2.0000e+00] (686#)
    
```

