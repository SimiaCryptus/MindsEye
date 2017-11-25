### Json Serialization
Code from [LayerTestBase.java:74](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L74) executed in 0.00 seconds: 
```java
  
```

Returns: 

```
    {
      "class": "com.simiacryptus.mindseye.layers.cudnn.f32.SimpleConvolutionLayer",
      "id": "bdd6bbba-380b-47fe-a761-c24100016361",
      "isFrozen": false,
      "name": "SimpleConvolutionLayer/bdd6bbba-380b-47fe-a761-c24100016361",
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
Code from [LayerTestBase.java:98](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L98) executed in 0.17 seconds: 
```java
  
```
Logging: 
```
    Finite-Difference Derivative Accuracy:
    absoluteTol: 2.6175e-08 +- 4.4841e-08 [0.0000e+00 - 2.2154e-07] (162#)
    relativeTol: 4.1936e-01 +- 8.1416e-01 [1.1947e-08 - 2.0000e+00] (62#)
    
```

### Performance
Code from [LayerTestBase.java:103](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L103) executed in 11.02 seconds: 
```java
  
```
Logging: 
```
    Forward performance: 616.3893 +- 49.7808 [518.4412 - 748.3048]
    Backward performance: 485.0295 +- 37.9650 [403.2473 - 605.5335]
    
```

### Reference Implementation
Code from [LayerTestBase.java:122](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L122) executed in 0.52 seconds: 
```java
  
```
Logging: 
```
    {
      "class": "com.simiacryptus.mindseye.layers.aparapi.ConvolutionLayer",
      "id": "bdd6bbba-380b-47fe-a761-c24100016362",
      "isFrozen": false,
      "name": "ConvolutionLayer/bdd6bbba-380b-47fe-a761-c24100016362",
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
    absoluteTol: 2.4105e-09 +- 6.5100e-09 [0.0000e+00 - 3.0063e-08] (162#)
    relativeTol: 1.0525e-08 +- 8.2995e-09 [8.8123e-10 - 2.2521e-08] (49#)
    
```

