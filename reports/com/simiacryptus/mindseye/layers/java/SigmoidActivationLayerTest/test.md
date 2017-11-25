### Json Serialization
Code from [LayerTestBase.java:74](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L74) executed in 0.00 seconds: 
```java
  
```

Returns: 

```
    {
      "class": "com.simiacryptus.mindseye.layers.java.SigmoidActivationLayer",
      "id": "bdd6bbba-380b-47fe-a761-c2410002dce3",
      "isFrozen": true,
      "name": "SigmoidActivationLayer/bdd6bbba-380b-47fe-a761-c2410002dce3",
      "balanced": true
    }
```



### Differential Validation
Code from [LayerTestBase.java:98](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L98) executed in 0.00 seconds: 
```java
  
```
Logging: 
```
    Finite-Difference Derivative Accuracy:
    absoluteTol: 3.1060e-08 +- 4.3984e-08 [0.0000e+00 - 9.6073e-08] (9#)
    relativeTol: 3.0350e-07 +- 3.7135e-08 [2.7593e-07 - 3.5600e-07] (3#)
    
```

### Performance
Code from [LayerTestBase.java:103](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L103) executed in 0.01 seconds: 
```java
  
```
Logging: 
```
    Forward performance: 1.1363 +- 0.6432 [0.6355 - 5.0498]
    
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



