### Json Serialization
Code from [LayerTestBase.java:74](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L74) executed in 0.00 seconds: 
```java
  
```

Returns: 

```
    {
      "class": "com.simiacryptus.mindseye.layers.java.ReLuActivationLayer",
      "id": "bdd6bbba-380b-47fe-a761-c2410002dce0",
      "isFrozen": true,
      "name": "ReLuActivationLayer/bdd6bbba-380b-47fe-a761-c2410002dce0",
      "weights": {
        "dimensions": [
          1
        ],
        "data": [
          1.0
        ]
      }
    }
```



### Differential Validation
Code from [LayerTestBase.java:98](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L98) executed in 0.01 seconds: 
```java
  
```
Logging: 
```
    Finite-Difference Derivative Accuracy:
    absoluteTol: 4.2553e-11 +- 6.3550e-11 [0.0000e+00 - 1.8564e-10] (12#)
    relativeTol: 9.0866e-11 +- 5.8905e-11 [1.0001e-12 - 1.8190e-10] (6#)
    
```

### Performance
Code from [LayerTestBase.java:103](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L103) executed in 0.03 seconds: 
```java
  
```
Logging: 
```
    Forward performance: 1.7817 +- 0.5330 [0.8806 - 3.9498]
    Backward performance: 1.5538 +- 0.6753 [0.6241 - 4.2462]
    
```

### Reference Implementation
### Function Plots
Code from [ActivationLayerTestBase.java:73](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/java/ActivationLayerTestBase.java#L73) executed in 0.01 seconds: 
```java
    return plot("Value Plot", plotData, x -> new double[]{x[0], x[1]});
```

Returns: 

![Result](etc/test.1.png)



Code from [ActivationLayerTestBase.java:77](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/java/ActivationLayerTestBase.java#L77) executed in 0.01 seconds: 
```java
    return plot("Derivative Plot", plotData, x -> new double[]{x[0], x[2]});
```

Returns: 

![Result](etc/test.2.png)



