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
      "id": "bdd6bbba-380b-47fe-a761-c241000015cb",
      "isFrozen": false,
      "name": "ConvolutionLayer/bdd6bbba-380b-47fe-a761-c241000015cb",
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
Code from [LayerTestBase.java:98](../../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L98) executed in 1.74 seconds: 
```java
    getDerivativeTester().test(layer, outputPrototype, inputPrototype);
```
Logging: 
```
    Finite-Difference Derivative Accuracy:
    absoluteTol: 1.4255e-08 +- 4.4358e-08 [0.0000e+00 - 3.1572e-07] (3240#)
    relativeTol: 4.3200e-01 +- 8.2303e-01 [7.5493e-09 - 2.0000e+00] (500#)
    
```

### Performance
Code from [LayerTestBase.java:103](../../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L103) executed in 115.72 seconds: 
```java
    getPerformanceTester().test(layer, outputPrototype, inputPrototype);
```
Logging: 
```
    Forward performance: 6427.3511 +- 1182.6463 [5514.9531 - 11768.6500]
    Backward performance: 5144.6467 +- 806.7592 [4573.5156 - 8972.6814]
    
```

### Reference Implementation
Code from [LayerTestBase.java:122](../../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L122) executed in 2.59 seconds: 
```java
    System.out.println(new GsonBuilder().setPrettyPrinting().create().toJson(referenceLayer.getJson()));
    getEquivalencyTester().test(referenceLayer, layer, outputPrototype, inputPrototype);
```
Logging: 
```
    {
      "class": "com.simiacryptus.mindseye.layers.aparapi.ConvolutionLayer",
      "id": "bdd6bbba-380b-47fe-a761-c241000088e0",
      "isFrozen": false,
      "name": "ConvolutionLayer/bdd6bbba-380b-47fe-a761-c241000088e0",
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
    absoluteTol: 1.8880e-01 +- 4.8537e-01 [0.0000e+00 - 1.9779e+00] (3240#)
    relativeTol: 1.7143e+00 +- 6.9985e-01 [5.4195e-10 - 2.0000e+00] (686#)
    
```

