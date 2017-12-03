# NthPowerActivationLayer
## SquarePowerTest
### Json Serialization
Code from [LayerTestBase.java:84](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L84) executed in 0.00 seconds: 
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
      "class": "com.simiacryptus.mindseye.layers.java.NthPowerActivationLayer",
      "id": "e2d0bffa-47dc-4875-864f-3d3d00001537",
      "isFrozen": false,
      "name": "NthPowerActivationLayer/e2d0bffa-47dc-4875-864f-3d3d00001537",
      "power": 2.0
    }
```



### Example Input/Output Pair
Code from [LayerTestBase.java:121](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L121) executed in 0.00 seconds: 
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
    [[ 1.412, 0.54, 1.756 ]]
    --------------------
    Output: 
    [ 1.9937439999999997, 0.2916, 3.083536 ]
```



### Differential Validation
Code from [LayerTestBase.java:139](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L139) executed in 0.00 seconds: 
```java
    getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Feedback for input 0
    Inputs: [ 1.412, 0.54, 1.756 ]
    Output: [ 1.9937439999999997, 0.2916, 3.083536 ]
    Measured: [ [ 2.824100000000662, 0.0, 0.0 ], [ 0.0, 1.0800999999999172, 0.0 ], [ 0.0, 0.0, 3.5120999999982416 ] ]
    Implemented: [ [ 2.824, 0.0, 0.0 ], [ 0.0, 1.08, 0.0 ], [ 0.0, 0.0, 3.512 ] ]
    Error: [ [ 1.0000000066234804E-4, 0.0, 0.0 ], [ 0.0, 9.999999991716635E-5, 0.0 ], [ 0.0, 0.0, 9.999999824161776E-5 ] ]
    Finite-Difference Derivative Accuracy:
    absoluteTol: 3.3333e-05 +- 4.7140e-05 [0.0000e+00 - 1.0000e-04] (9#)
    relativeTol: 2.6079e-05 +- 1.4364e-05 [1.4237e-05 - 4.6294e-05] (3#)
    
```

### Performance
Code from [LayerTestBase.java:144](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.00 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 0.1286 +- 0.0342 [0.0997 - 0.3078]
    Learning performance: 0.0010 +- 0.0014 [0.0000 - 0.0057]
    
```

### Function Plots
Code from [ActivationLayerTestBase.java:73](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/java/ActivationLayerTestBase.java#L73) executed in 0.10 seconds: 
```java
    return plot("Value Plot", plotData, x -> new double[]{x[0], x[1]});
```

Returns: 

![Result](etc/test.1.png)



Code from [ActivationLayerTestBase.java:77](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/java/ActivationLayerTestBase.java#L77) executed in 0.00 seconds: 
```java
    return plot("Derivative Plot", plotData, x -> new double[]{x[0], x[2]});
```

Returns: 

![Result](etc/test.2.png)



