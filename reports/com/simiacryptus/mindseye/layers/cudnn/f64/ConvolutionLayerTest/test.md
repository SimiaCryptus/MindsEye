### Json Serialization
Code from [LayerTestBase.java:74](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L74) executed in 0.00 seconds: 
```java
  
```

Returns: 

```
    {
      "class": "com.simiacryptus.mindseye.layers.cudnn.f64.ConvolutionLayer",
      "id": "bdd6bbba-380b-47fe-a761-c24100016365",
      "isFrozen": false,
      "name": "ConvolutionLayer/bdd6bbba-380b-47fe-a761-c24100016365",
      "filter": {
        "dimensions": [
          3,
          3,
          4
        ]
      },
      "strideX": 1,
      "strideY": 1
    }
```



### Differential Validation
Code from [LayerTestBase.java:98](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L98) executed in 1.10 seconds: 
```java
  
```
Logging: 
```
    Finite-Difference Derivative Accuracy:
    absoluteTol: 8.6808e-18 +- 3.8363e-17 [0.0000e+00 - 2.2204e-16] (972#)
    relativeTol: 4.8907e-17 +- 7.9936e-17 [0.0000e+00 - 2.1636e-16] (196#)
    
```

### Performance
Code from [LayerTestBase.java:103](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L103) executed in 32.73 seconds: 
```java
  
```
Logging: 
```
    Forward performance: 1785.7385 +- 437.3376 [1463.7146 - 3036.5587]
    Backward performance: 1487.3774 +- 226.9126 [1137.8788 - 2724.8494]
    
```

### Reference Implementation
Code from [LayerTestBase.java:122](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L122) executed in 1.02 seconds: 
```java
  
```
Logging: 
```
    {
      "class": "com.simiacryptus.mindseye.layers.aparapi.ConvolutionLayer",
      "id": "bdd6bbba-380b-47fe-a761-c24100018ead",
      "isFrozen": false,
      "name": "ConvolutionLayer/bdd6bbba-380b-47fe-a761-c24100018ead",
      "filter": {
        "dimensions": [
          3,
          3,
          4
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
    absoluteTol: 1.6886e-01 +- 4.2860e-01 [0.0000e+00 - 1.9644e+00] (972#)
    relativeTol: 1.3333e+00 +- 9.4281e-01 [0.0000e+00 - 2.0000e+00] (294#)
    
```

