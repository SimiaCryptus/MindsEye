### Json Serialization
Code from [LayerTestBase.java:74](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L74) executed in 0.00 seconds: 
```java
  
```

Returns: 

```
    {
      "class": "com.simiacryptus.mindseye.layers.java.NthPowerActivationLayer",
      "id": "bdd6bbba-380b-47fe-a761-c2410002dcd9",
      "isFrozen": false,
      "name": "NthPowerActivationLayer/bdd6bbba-380b-47fe-a761-c2410002dcd9",
      "power": 2.5
    }
```



### Differential Validation
Code from [LayerTestBase.java:98](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L98) executed in 0.00 seconds: 
```java
  
```
Logging: 
```
    Finite-Difference Derivative Accuracy:
    absoluteTol: 4.3902e-07 +- 8.5677e-07 [0.0000e+00 - 2.4930e-06] (9#)
    relativeTol: 8.3198e-07 +- 4.0813e-07 [4.2384e-07 - 1.2401e-06] (2#)
    
```

### Performance
Code from [LayerTestBase.java:103](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L103) executed in 0.01 seconds: 
```java
  
```
Logging: 
```
    Forward performance: 1.0572 +- 0.3242 [0.6640 - 2.5534]
    
```

### Reference Implementation
### Function Plots
Code from [ActivationLayerTestBase.java:73](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/java/ActivationLayerTestBase.java#L73) executed in 0.01 seconds: 
```java
    return plot("Value Plot", plotData, x -> new double[]{x[0], x[1]});
```

Returns: 

![Result](etc/test.1.png)



Code from [ActivationLayerTestBase.java:77](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/java/ActivationLayerTestBase.java#L77) executed in 0.01 seconds: 
```java
    return plot("Derivative Plot", plotData, x -> new double[]{x[0], x[2]});
```

Returns: 

![Result](etc/test.2.png)



