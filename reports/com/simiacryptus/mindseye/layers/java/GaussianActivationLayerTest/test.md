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
      "id": "e2d0bffa-47dc-4875-864f-3d3d000014a3",
      "isFrozen": true,
      "name": "GaussianActivationLayer/e2d0bffa-47dc-4875-864f-3d3d000014a3",
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
    [[ 0.268, -1.868, -0.024 ]]
    --------------------
    Output: 
    [ 0.3848696654896559, 0.06969333895067566, 0.39882740156802315 ]
```



### Differential Validation
Code from [LayerTestBase.java:139](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L139) executed in 0.00 seconds: 
```java
    getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Feedback for input 0
    Inputs: [ 0.268, -1.868, -0.024 ]
    Output: [ 0.3848696654896559, 0.06969333895067566, 0.39882740156802315 ]
    Measured: [ [ -0.10316293118761699, 0.0, 0.0 ], [ 0.0, 0.1301958320797103, 0.0 ], [ 0.0, 0.0, 0.009551927706019647 ] ]
    Implemented: [ [ -0.10314507035122777, 0.0, 0.0 ], [ 0.0, 0.13018715715986212, 0.0 ], [ 0.0, 0.0, 0.009571857637632556 ] ]
    Error: [ [ -1.7860836389216317E-5, 0.0, 0.0 ], [ 0.0, 8.674919848183915E-6, 0.0 ], [ 0.0, 0.0, -1.9929931612908472E-5 ] ]
    Finite-Difference Derivative Accuracy:
    absoluteTol: 5.1629e-06 +- 7.8285e-06 [0.0000e+00 - 1.9930e-05] (9#)
    relativeTol: 3.8735e-04 +- 4.6353e-04 [3.3316e-05 - 1.0422e-03] (3#)
    
```

### Performance
Code from [LayerTestBase.java:144](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.00 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 0.2092 +- 0.2325 [0.1055 - 2.1972]
    Learning performance: 0.0032 +- 0.0065 [0.0000 - 0.0513]
    
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



