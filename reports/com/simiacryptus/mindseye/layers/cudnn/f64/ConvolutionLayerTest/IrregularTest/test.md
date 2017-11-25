### Json Serialization
Code from [LayerTestBase.java:74](../../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L74) executed in 0.00 seconds: 
```java
  
```

Returns: 

```
    {
      "class": "com.simiacryptus.mindseye.layers.cudnn.f64.ConvolutionLayer",
      "id": "bdd6bbba-380b-47fe-a761-c24100020350",
      "isFrozen": false,
      "name": "ConvolutionLayer/bdd6bbba-380b-47fe-a761-c24100020350",
      "filter": {
        "dimensions": [
          3,
          3,
          10
        ]
      },
      "strideX": 1,
      "strideY": 1
    }
```



### Differential Validation
Code from [LayerTestBase.java:98](../../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L98) executed in 3.41 seconds: 
```java
  
```
Logging: 
```
    Finite-Difference Derivative Accuracy:
    absoluteTol: 4.2833e-18 +- 2.9752e-17 [0.0000e+00 - 2.2204e-16] (5832#)
    relativeTol: 3.6774e-17 +- 6.0514e-17 [0.0000e+00 - 1.6736e-16] (490#)
    
```

### Performance
Code from [LayerTestBase.java:103](../../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L103) executed in 223.56 seconds: 
```java
  
```
Logging: 
```
    Forward performance: 12235.7720 +- 2324.2690 [11038.2390 - 20588.0130]
    Backward performance: 10120.2374 +- 1978.8529 [8092.7821 - 17315.8626]
    
```

### Reference Implementation
Code from [LayerTestBase.java:122](../../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L122) executed in 4.28 seconds: 
```java
  
```
Logging: 
```
    {
      "class": "com.simiacryptus.mindseye.layers.aparapi.ConvolutionLayer",
      "id": "bdd6bbba-380b-47fe-a761-c2410002da88",
      "isFrozen": false,
      "name": "ConvolutionLayer/bdd6bbba-380b-47fe-a761-c2410002da88",
      "filter": {
        "dimensions": [
          3,
          3,
          10
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
    absoluteTol: 1.3771e-01 +- 4.0850e-01 [0.0000e+00 - 1.8988e+00] (5832#)
    relativeTol: 1.7778e+00 +- 6.2854e-01 [0.0000e+00 - 2.0000e+00] (882#)
    
```

