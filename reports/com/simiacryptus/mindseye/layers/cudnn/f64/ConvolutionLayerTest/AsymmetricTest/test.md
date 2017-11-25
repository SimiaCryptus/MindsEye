### Json Serialization
Code from [LayerTestBase.java:74](../../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L74) executed in 0.00 seconds: 
```java
  
```

Returns: 

```
    {
      "class": "com.simiacryptus.mindseye.layers.cudnn.f64.ConvolutionLayer",
      "id": "bdd6bbba-380b-47fe-a761-c24100018f1a",
      "isFrozen": false,
      "name": "ConvolutionLayer/bdd6bbba-380b-47fe-a761-c24100018f1a",
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
Code from [LayerTestBase.java:98](../../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L98) executed in 1.90 seconds: 
```java
  
```
Logging: 
```
    Finite-Difference Derivative Accuracy:
    absoluteTol: 5.1399e-18 +- 3.1408e-17 [0.0000e+00 - 2.2204e-16] (3240#)
    relativeTol: 4.3345e-17 +- 6.7491e-17 [0.0000e+00 - 2.0056e-16] (392#)
    
```

### Performance
Code from [LayerTestBase.java:103](../../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L103) executed in 106.81 seconds: 
```java
  
```
Logging: 
```
    Forward performance: 5879.6102 +- 1078.1680 [4488.0335 - 10131.3095]
    Backward performance: 4800.8442 +- 993.6161 [3592.3606 - 8817.8783]
    
```

### Reference Implementation
Code from [LayerTestBase.java:122](../../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L122) executed in 2.36 seconds: 
```java
  
```
Logging: 
```
    {
      "class": "com.simiacryptus.mindseye.layers.aparapi.ConvolutionLayer",
      "id": "bdd6bbba-380b-47fe-a761-c2410002022f",
      "isFrozen": false,
      "name": "ConvolutionLayer/bdd6bbba-380b-47fe-a761-c2410002022f",
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
    absoluteTol: 1.7177e-01 +- 4.6025e-01 [0.0000e+00 - 1.9746e+00] (3240#)
    relativeTol: 1.7143e+00 +- 6.9985e-01 [0.0000e+00 - 2.0000e+00] (686#)
    
```

