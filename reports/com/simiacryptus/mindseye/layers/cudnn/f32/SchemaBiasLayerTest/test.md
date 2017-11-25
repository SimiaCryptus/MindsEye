### Json Serialization
Code from [LayerTestBase.java:74](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L74) executed in 0.00 seconds: 
```java
  
```

Returns: 

```
    {
      "class": "com.simiacryptus.mindseye.layers.cudnn.f32.SchemaBiasLayer",
      "id": "bdd6bbba-380b-47fe-a761-c2410001635d",
      "isFrozen": false,
      "name": "SchemaBiasLayer/bdd6bbba-380b-47fe-a761-c2410001635d",
      "selected": [
        "test1",
        "test2"
      ],
      "features": {
        "test2": 0.0,
        "test1": 0.0
      }
    }
```



### Differential Validation
Code from [LayerTestBase.java:98](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L98) executed in 0.19 seconds: 
```java
  
```
Logging: 
```
    Finite-Difference Derivative Accuracy:
    absoluteTol: 1.6491e-05 +- 7.3509e-05 [0.0000e+00 - 1.0262e-03] (360#)
    relativeTol: 1.6492e-04 +- 1.7201e-04 [1.6928e-05 - 1.0267e-03] (36#)
    
```

### Performance
Code from [LayerTestBase.java:103](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L103) executed in 13.73 seconds: 
```java
  
```
Logging: 
```
    Forward performance: 690.0588 +- 31.8486 [605.5449 - 789.3873]
    Backward performance: 683.0819 +- 54.3581 [390.5686 - 810.5469]
    
```

### Reference Implementation
