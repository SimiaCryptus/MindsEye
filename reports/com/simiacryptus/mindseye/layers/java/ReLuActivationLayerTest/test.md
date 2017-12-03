# ReLuActivationLayer
## ReLuActivationLayerTest
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
      "class": "com.simiacryptus.mindseye.layers.java.ReLuActivationLayer",
      "id": "e2d0bffa-47dc-4875-864f-3d3d0000154f",
      "isFrozen": true,
      "name": "ReLuActivationLayer/e2d0bffa-47dc-4875-864f-3d3d0000154f",
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
    [[ -0.684, -0.48, 0.692 ]]
    --------------------
    Output: 
    [ 0.0, 0.0, 0.692 ]
```



### Differential Validation
Code from [LayerTestBase.java:139](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L139) executed in 0.00 seconds: 
```java
    getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Feedback for input 0
    Inputs: [ -0.684, -0.48, 0.692 ]
    Output: [ 0.0, 0.0, 0.692 ]
    Measured: [ [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.9999999999998899 ] ]
    Implemented: [ [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 1.0 ] ]
    Error: [ [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, -1.1013412404281553E-13 ] ]
    Learning Gradient for weight set 0
    Inputs: [ -0.684, -0.48, 0.692 ]
    Outputs: [ 0.0, 0.0, 0.692 ]
    Measured Gradient: [ [ 0.0, 0.0, 0.6919999999999149 ] ]
    Implemented Gradient: [ [ 0.0, 0.0, 0.692 ] ]
    Error: [ [ 0.0, 0.0, -8.504308368628699E-14 ] ]
    Finite-Difference Derivative Accuracy:
    absoluteTol: 1.6265e-14 +- 3.6728e-14 [0.0000e+00 - 1.1013e-13] (12#)
    relativeTol: 5.8257e-14 +- 3.1901e-15 [5.5067e-14 - 6.1447e-14] (2#)
    
```

### Performance
Code from [LayerTestBase.java:144](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.00 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 0.2020 +- 0.0504 [0.1396 - 0.4332]
    Learning performance: 0.0503 +- 0.0157 [0.0370 - 0.1254]
    
```

### Function Plots
Code from [ActivationLayerTestBase.java:73](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/java/ActivationLayerTestBase.java#L73) executed in 0.00 seconds: 
```java
    return plot("Value Plot", plotData, x -> new double[]{x[0], x[1]});
```

Returns: 

![Result](etc/test.1.png)



Code from [ActivationLayerTestBase.java:77](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/java/ActivationLayerTestBase.java#L77) executed in 0.00 seconds: 
```java
    return plot("Derivative Plot", plotData, x -> new double[]{x[0], x[2]});
```

Returns: 

![Result](etc/test.2.png)



