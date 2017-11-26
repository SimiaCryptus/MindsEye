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
      "class": "com.simiacryptus.mindseye.layers.cudnn.f32.SimpleConvolutionLayer",
      "id": "8154aa62-3b26-48f3-9c49-7caa00000001",
      "isFrozen": false,
      "name": "SimpleConvolutionLayer/8154aa62-3b26-48f3-9c49-7caa00000001",
      "filter": {
        "dimensions": [
          3,
          3,
          1
        ],
        "data": [
          0.826719004920013,
          -0.0523600102172761,
          0.5582696403883334,
          0.2888468217099134,
          0.9555271248594714,
          0.8684667789978364,
          -0.12921644653837983,
          -0.22026405805944038,
          0.6044340630476626
        ]
      },
      "simple": false,
      "strideX": 1,
      "strideY": 1
    }
```



### Differential Validation
Code from [LayerTestBase.java:100](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L100) executed in 0.32 seconds: 
```java
    getDerivativeTester().test(layer, outputPrototype, inputPrototype);
```
Logging: 
```
    Finite-Difference Derivative Accuracy:
    absoluteTol: 1.2411e-03 +- 1.2979e-03 [0.0000e+00 - 6.5726e-03] (162#)
    relativeTol: 2.1878e-01 +- 4.1017e-01 [3.2029e-05 - 1.0000e+00] (125#)
    
```

### Performance
Code from [LayerTestBase.java:105](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L105) executed in 11.08 seconds: 
```java
    getPerformanceTester().test(layer, outputPrototype, inputPrototype);
```
Logging: 
```
    Forward performance: 637.1407 +- 44.2622 [537.4465 - 809.2047]
    Backward performance: 469.4678 +- 26.9065 [418.6019 - 574.5478]
    
```

### Reference Implementation
Code from [LayerTestBase.java:124](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L124) executed in 1.85 seconds: 
```java
    System.out.println(new GsonBuilder().setPrettyPrinting().create().toJson(referenceLayer.getJson()));
    getEquivalencyTester().test(referenceLayer, layer, outputPrototype, inputPrototype);
```
Logging: 
```
    {
      "class": "com.simiacryptus.mindseye.layers.aparapi.ConvolutionLayer",
      "id": "8154aa62-3b26-48f3-9c49-7caa00000002",
      "isFrozen": false,
      "name": "ConvolutionLayer/8154aa62-3b26-48f3-9c49-7caa00000002",
      "filter": {
        "dimensions": [
          3,
          3,
          1
        ],
        "data": [
          0.826719004920013,
          -0.0523600102172761,
          0.5582696403883334,
          0.2888468217099134,
          0.9555271248594714,
          0.8684667789978364,
          -0.12921644653837983,
          -0.22026405805944038,
          0.6044340630476626
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
    absoluteTol: 1.3552e-08 +- 2.0907e-08 [0.0000e+00 - 1.3966e-07] (162#)
    relativeTol: 2.4289e-08 +- 3.0494e-08 [9.2102e-10 - 1.2508e-07] (98#)
    
```

