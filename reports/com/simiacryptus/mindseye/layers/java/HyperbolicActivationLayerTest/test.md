### Json Serialization
Code from [LayerTestBase.java:74](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L74) executed in 0.00 seconds: 
```java
  
```

Returns: 

```
    {
      "class": "com.simiacryptus.mindseye.layers.java.HyperbolicActivationLayer",
      "id": "bdd6bbba-380b-47fe-a761-c2410002dcbb",
      "isFrozen": false,
      "name": "HyperbolicActivationLayer/bdd6bbba-380b-47fe-a761-c2410002dcbb",
      "weights": {
        "dimensions": [
          2
        ],
        "data": [
          1.0,
          1.0
        ]
      },
      "negativeMode": 1
    }
```



### Differential Validation
Code from [LayerTestBase.java:98](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L98) executed in 0.00 seconds: 
```java
  
```
Logging: 
```
    Finite-Difference Derivative Accuracy:
    absoluteTol: 2.6441e-07 +- 3.8128e-07 [0.0000e+00 - 9.9532e-07] (15#)
    relativeTol: 9.7188e-07 +- 2.7541e-07 [4.4493e-07 - 1.2586e-06] (6#)
    
```

### Performance
Code from [LayerTestBase.java:103](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L103) executed in 0.03 seconds: 
```java
  
```
Logging: 
```
    Forward performance: 1.7965 +- 0.8319 [1.0003 - 7.1729]
    Backward performance: 1.1870 +- 0.5193 [0.5700 - 4.1407]
    
```

### Reference Input/Output Pairs
Code from [LayerTestBase.java:111](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L111) executed in 0.00 seconds: 
```java
  
```

Returns: 

```
    Input: [Lcom.simiacryptus.mindseye.lang.Tensor;@41a6d121
    Output: [ 0.0 ]
    Error: 0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (1#)
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



