# LogActivationLayer
## LogActivationLayerTest
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
      "class": "com.simiacryptus.mindseye.layers.java.LogActivationLayer",
      "id": "e2d0bffa-47dc-4875-864f-3d3d000014fd",
      "isFrozen": true,
      "name": "LogActivationLayer/e2d0bffa-47dc-4875-864f-3d3d000014fd"
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
    [[ -1.428, -1.712, -1.068 ]]
    --------------------
    Output: 
    [ 0.3562748639173926, 0.5376622777195503, 0.06578774053800315 ]
```



### Differential Validation
Code from [LayerTestBase.java:139](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L139) executed in 0.00 seconds: 
```java
    getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Feedback for input 0
    Inputs: [ -1.428, -1.712, -1.068 ]
    Output: [ 0.3562748639173926, 0.5376622777195503, 0.06578774053800315 ]
    Measured: [ [ -0.7002801116762214, 0.0, 0.0 ], [ 0.0, -0.5841121475391731, 0.0 ], [ 0.0, 0.0, -0.9363295860875809 ] ]
    Implemented: [ [ -0.700280112044818, 0.0, 0.0 ], [ 0.0, -0.5841121495327103, 0.0 ], [ 0.0, 0.0, -0.9363295880149812 ] ]
    Error: [ [ 3.685965976885086E-10, 0.0, 0.0 ], [ 0.0, 1.9935372241874916E-9, 0.0 ], [ 0.0, 0.0, 1.9274003504321513E-9 ] ]
    Finite-Difference Derivative Accuracy:
    absoluteTol: 4.7661e-10 +- 8.0142e-10 [0.0000e+00 - 1.9935e-09] (9#)
    relativeTol: 9.9963e-10 +- 5.8959e-10 [2.6318e-10 - 1.7065e-09] (3#)
    
```

### Performance
Code from [LayerTestBase.java:144](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.00 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 0.1472 +- 0.0416 [0.0941 - 0.3619]
    Learning performance: 0.0052 +- 0.0183 [0.0000 - 0.1852]
    
```

### Function Plots
Code from [ActivationLayerTestBase.java:73](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/java/ActivationLayerTestBase.java#L73) executed in 0.01 seconds: 
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



