### Json Serialization
Code from [LayerTestBase.java:74](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L74) executed in 0.00 seconds: 
```java
  
```

Returns: 

```
    {
      "class": "com.simiacryptus.mindseye.layers.java.LinearActivationLayer",
      "id": "bdd6bbba-380b-47fe-a761-c2410002dcc4",
      "isFrozen": false,
      "name": "LinearActivationLayer/bdd6bbba-380b-47fe-a761-c2410002dcc4",
      "weights": {
        "dimensions": [
          2
        ],
        "data": [
          1.0,
          0.0
        ]
      }
    }
```



### Differential Validation
Code from [LayerTestBase.java:98](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L98) executed in 0.00 seconds: 
```java
  
```
Logging: 
```
    Finite-Difference Derivative Accuracy:
    absoluteTol: 2.6225e-11 +- 3.1998e-11 [0.0000e+00 - 1.2281e-10] (15#)
    relativeTol: 4.9900e-11 +- 3.8078e-11 [2.8756e-11 - 1.3854e-10] (9#)
    
```

### Performance
Code from [LayerTestBase.java:103](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L103) executed in 0.03 seconds: 
```java
  
```
Logging: 
```
    Forward performance: 1.5391 +- 0.5885 [0.7979 - 4.4770]
    Backward performance: 1.2996 +- 1.2981 [0.5871 - 13.0178]
    
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



