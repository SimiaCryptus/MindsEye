### Json Serialization
Code from [LayerTestBase.java:74](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L74) executed in 0.00 seconds: 
```java
  
```

Returns: 

```
    {
      "class": "com.simiacryptus.mindseye.layers.java.FullyConnectedLayer",
      "id": "bdd6bbba-380b-47fe-a761-c2410002dcb8",
      "isFrozen": false,
      "name": "FullyConnectedLayer/bdd6bbba-380b-47fe-a761-c2410002dcb8",
      "outputDims": [
        3
      ],
      "inputDims": [
        3
      ],
      "weights": {
        "dimensions": [
          3,
          3
        ],
        "data": [
          0.25669730332498525,
          0.26036702968649494,
          -0.4223764149243788,
          0.2572318477154033,
          -0.4547870130561086,
          -0.6089738171089158,
          0.2841177742503492,
          -0.42830138276893776,
          0.5751474563785285
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
    absoluteTol: 3.0098e-11 +- 4.0122e-11 [0.0000e+00 - 1.3354e-10] (36#)
    relativeTol: 1.2190e-10 +- 1.3611e-10 [5.1832e-12 - 5.2023e-10] (18#)
    
```

### Performance
Code from [LayerTestBase.java:103](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L103) executed in 0.07 seconds: 
```java
  
```
Logging: 
```
    Forward performance: 4.5133 +- 2.4423 [1.7555 - 14.3772]
    Backward performance: 2.7977 +- 1.8076 [1.4648 - 18.8570]
    
```

### Reference Implementation
