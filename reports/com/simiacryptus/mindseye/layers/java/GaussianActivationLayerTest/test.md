# GaussianActivationLayer
## GaussianActivationLayerTest
### Json Serialization
Code from [LayerTestBase.java:84](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L84) executed in 0.00 seconds: 
```java
    JsonObject json = layer.getJson();
    NNLayer echo = NNLayer.fromJson(json);
    assert (echo != null) : "Failed to deserialize";
    assert (layer != echo) : "Serialization did not copy";
    Assert.assertEquals("Serialization not equal", layer, echo);
    return new GsonBuilder().setPrettyPrinting().create().toJson(json);
```

Returns: 

```
    {
      "class": "com.simiacryptus.mindseye.layers.java.GaussianActivationLayer",
      "id": "f4569375-56fe-4e46-925c-95f4000009b7",
      "isFrozen": true,
      "name": "GaussianActivationLayer/f4569375-56fe-4e46-925c-95f4000009b7",
      "mean": 0.0,
      "stddev": 1.0
    }
```



### Example Input/Output Pair
Code from [LayerTestBase.java:121](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L121) executed in 0.00 seconds: 
```java
    SimpleEval eval = SimpleEval.run(layer, inputPrototype);
    return String.format("--------------------\nInput: \n[%s]\n--------------------\nOutput: \n%s",
      Arrays.stream(inputPrototype).map(t->t.prettyPrint()).reduce((a,b)->a+",\n"+b).get(),
      eval.getOutput().prettyPrint());
```

Returns: 

```
    --------------------
    Input: 
    [[ 0.868, 1.3, -1.488 ]]
    --------------------
    Output: 
    [ 0.2737197426193348, 0.17136859204780736, 0.1318605263983428 ]
```



### Differential Validation
Code from [LayerTestBase.java:139](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L139) executed in 0.00 seconds: 
```java
    getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Feedback for input 0
    Inputs: [ 0.868, 1.3, -1.488 ]
    Output: [ 0.2737197426193348, 0.17136859204780736, 0.1318605263983428 ]
    Measured: [ [ -0.23759211033980154, 0.0, 0.0 ], [ 0.0, -0.22277325695901906, 0.0 ], [ 0.0, 0.0, 0.1962164679072953 ] ]
    Implemented: [ [ -0.23758873659358262, 0.0, 0.0 ], [ 0.0, -0.22277916966214956, 0.0 ], [ 0.0, 0.0, 0.1962084632807341 ] ]
    Error: [ [ -3.3737462189209477E-6, 0.0, 0.0 ], [ 0.0, 5.912703130495078E-6, 0.0 ], [ 0.0, 0.0, 8.004626561197714E-6 ] ]
    Finite-Difference Derivative Accuracy:
    absoluteTol: 1.9212e-06 +- 2.9287e-06 [0.0000e+00 - 8.0046e-06] (9#)
    relativeTol: 1.3589e-05 +- 5.4335e-06 [7.0999e-06 - 2.0398e-05] (3#)
    
```

### Performance
Code from [LayerTestBase.java:144](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.00 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 0.1543 +- 0.0331 [0.1140 - 0.3391]
    Learning performance: 0.0026 +- 0.0026 [0.0000 - 0.0171]
    
```

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



