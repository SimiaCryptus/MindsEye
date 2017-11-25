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
      "id": "9d13704a-9a5a-4ecb-a687-5c7c000015db",
      "isFrozen": false,
      "name": "ConvolutionLayer/9d13704a-9a5a-4ecb-a687-5c7c000015db",
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
Code from [LayerTestBase.java:74](../../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L74) executed in 0.49 seconds: 
```java
    getDerivativeTester().test(getLayer(), outputPrototype, inputPrototype);
```
Logging: 
```
    Finite-Difference Derivative Accuracy:
    absoluteTol: 3.2131e-09 +- 9.5126e-09 [0.0000e+00 - 7.2760e-08] (3240#)
    relativeTol: 4.4444e-01 +- 8.3148e-01 [6.8731e-10 - 2.0000e+00] (504#)
    
```

### Performance
Code from [LayerTestBase.java:79](../../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L79) executed in 23.96 seconds: 
```java
    getPerformanceTester().test(getLayer(), outputPrototype, inputPrototype);
```
Logging: 
```
    Forward performance: 1324.8161 +- 257.1034 [1201.0528 - 2430.4211]
    Backward performance: 1071.0451 +- 201.6486 [948.6760 - 1923.0057]
    
```

### Reference Implementation
Code from [LayerTestBase.java:86](../../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L86) executed in 0.62 seconds: 
```java
    System.out.println(new GsonBuilder().setPrettyPrinting().create().toJson(referenceLayer.getJson()));
    getEquivalencyTester().test(referenceLayer, getLayer(), outputPrototype, inputPrototype);
```
Logging: 
```
    {
      "class": "com.simiacryptus.mindseye.layers.aparapi.ConvolutionLayer",
      "id": "9d13704a-9a5a-4ecb-a687-5c7c000088f3",
      "isFrozen": false,
      "name": "ConvolutionLayer/9d13704a-9a5a-4ecb-a687-5c7c000088f3",
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
    absoluteTol: 7.8732e-02 +- 2.0598e-01 [0.0000e+00 - 9.9560e-01] (3240#)
    relativeTol: 1.7143e+00 +- 6.9985e-01 [1.5410e-10 - 2.0000e+00] (686#)
    
```

