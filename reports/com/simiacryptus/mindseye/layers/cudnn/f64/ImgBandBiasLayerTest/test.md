### Json Serialization
Code from [LayerTestBase.java:74](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L74) executed in 0.00 seconds: 
```java
  
```

Returns: 

```
    {
      "class": "com.simiacryptus.mindseye.layers.cudnn.f64.ImgBandBiasLayer",
      "id": "bdd6bbba-380b-47fe-a761-c2410002dca5",
      "isFrozen": false,
      "name": "ImgBandBiasLayer/bdd6bbba-380b-47fe-a761-c2410002dca5",
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
    absoluteTol: 5.1872e-12 +- 1.8454e-11 [0.0000e+00 - 8.2267e-11] (360#)
    relativeTol: 5.1872e-11 +- 3.1370e-11 [1.0001e-12 - 8.2267e-11] (36#)
    
```

### Performance
Code from [LayerTestBase.java:103](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L103) executed in 13.36 seconds: 
```java
  
```
Logging: 
```
    Forward performance: 660.6403 +- 27.7904 [583.9036 - 731.1007]
    Backward performance: 675.2279 +- 31.0617 [612.2248 - 741.5708]
    
```

### Reference Implementation
