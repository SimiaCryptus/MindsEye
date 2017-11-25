### Json Serialization
Code from [LayerTestBase.java:74](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L74) executed in 0.00 seconds: 
```java
  
```

Returns: 

```
    {
      "class": "com.simiacryptus.mindseye.layers.cudnn.f32.ImgBandBiasLayer",
      "id": "bdd6bbba-380b-47fe-a761-c24100016359",
      "isFrozen": false,
      "name": "ImgBandBiasLayer/bdd6bbba-380b-47fe-a761-c24100016359",
      "bias": [
        0.0,
        0.0
      ]
    }
```



### Differential Validation
Code from [LayerTestBase.java:98](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L98) executed in 0.19 seconds: 
```java
  
```
Logging: 
```
    Finite-Difference Derivative Accuracy:
    absoluteTol: 1.8231e-05 +- 7.1032e-05 [0.0000e+00 - 1.0262e-03] (360#)
    relativeTol: 1.8232e-04 +- 1.4341e-04 [1.3209e-04 - 1.0267e-03] (36#)
    
```

### Performance
Code from [LayerTestBase.java:103](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L103) executed in 14.54 seconds: 
```java
  
```
Logging: 
```
    Forward performance: 685.3711 +- 29.5893 [600.6205 - 759.0998]
    Backward performance: 768.4105 +- 176.5743 [638.3659 - 2009.1176]
    
```

### Reference Implementation
