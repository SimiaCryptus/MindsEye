### Json Serialization
Code from [LayerTestBase.java:74](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L74) executed in 0.00 seconds: 
```java
  
```

Returns: 

```
    {
      "class": "com.simiacryptus.mindseye.layers.cudnn.f64.SimpleConvolutionLayer",
      "id": "bdd6bbba-380b-47fe-a761-c2410002dca7",
      "isFrozen": false,
      "name": "SimpleConvolutionLayer/bdd6bbba-380b-47fe-a761-c2410002dca7",
      "filter": {
        "dimensions": [
          3,
          3,
          1
        ]
      },
      "strideX": 1,
      "strideY": 1,
      "simple": false
    }
```



### Differential Validation
Code from [LayerTestBase.java:98](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L98) executed in 0.13 seconds: 
```java
  
```
Logging: 
```
    Finite-Difference Derivative Accuracy:
    absoluteTol: 1.2336e-17 +- 5.0862e-17 [0.0000e+00 - 2.2204e-16] (162#)
    relativeTol: 3.7991e-17 +- 8.0092e-17 [0.0000e+00 - 2.0684e-16] (49#)
    
```

### Performance
Code from [LayerTestBase.java:103](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L103) executed in 8.21 seconds: 
```java
  
```
Logging: 
```
    Forward performance: 458.9104 +- 29.2025 [385.3791 - 564.2772]
    Backward performance: 361.8256 +- 21.2405 [300.8602 - 430.1948]
    
```

### Reference Implementation
Code from [LayerTestBase.java:122](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L122) executed in 0.49 seconds: 
```java
  
```
Logging: 
```
    {
      "class": "com.simiacryptus.mindseye.layers.aparapi.ConvolutionLayer",
      "id": "bdd6bbba-380b-47fe-a761-c2410002dca8",
      "isFrozen": false,
      "name": "ConvolutionLayer/bdd6bbba-380b-47fe-a761-c2410002dca8",
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
    absoluteTol: 0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (162#)
    relativeTol: 0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (49#)
    
```

