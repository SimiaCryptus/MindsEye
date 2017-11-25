### Json Serialization
Code from [LayerTestBase.java:74](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L74) executed in 0.00 seconds: 
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
      "id": "bdd6bbba-380b-47fe-a761-c24100000006",
      "isFrozen": false,
      "name": "ConvolutionLayer/bdd6bbba-380b-47fe-a761-c24100000006",
      "filter": {
        "dimensions": [
          3,
          3,
          1
        ]
      },
      "strideX": 1,
      "strideY": 1
    }
```



### Differential Validation
Code from [LayerTestBase.java:98](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L98) executed in 0.31 seconds: 
```java
    getDerivativeTester().test(layer, outputPrototype, inputPrototype);
```
Logging: 
```
    Finite-Difference Derivative Accuracy:
    absoluteTol: 2.2472e-08 +- 3.5625e-08 [0.0000e+00 - 2.1257e-07] (162#)
    relativeTol: 4.1935e-01 +- 8.1416e-01 [2.3192e-08 - 2.0000e+00] (62#)
    
```

### Performance
Code from [LayerTestBase.java:103](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L103) executed in 17.61 seconds: 
```java
    getPerformanceTester().test(layer, outputPrototype, inputPrototype);
```
Logging: 
```
    Forward performance: 941.0582 +- 186.4544 [752.8075 - 2323.0811]
    Backward performance: 819.9077 +- 175.8304 [689.5822 - 2036.6209]
    
```

### Reference Implementation
Code from [LayerTestBase.java:122](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L122) executed in 0.57 seconds: 
```java
    System.out.println(new GsonBuilder().setPrettyPrinting().create().toJson(referenceLayer.getJson()));
    getEquivalencyTester().test(referenceLayer, layer, outputPrototype, inputPrototype);
```
Logging: 
```
    {
      "class": "com.simiacryptus.mindseye.layers.aparapi.ConvolutionLayer",
      "id": "bdd6bbba-380b-47fe-a761-c24100001594",
      "isFrozen": false,
      "name": "ConvolutionLayer/bdd6bbba-380b-47fe-a761-c24100001594",
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
    absoluteTol: 5.9997e-09 +- 1.1456e-08 [0.0000e+00 - 4.6772e-08] (162#)
    relativeTol: 1.7419e-08 +- 1.1256e-08 [1.7917e-09 - 3.7665e-08] (49#)
    
```

