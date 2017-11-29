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
      "id": "c88cbdf1-1c2a-4a5e-b964-890900000f7d",
      "isFrozen": true,
      "name": "LogActivationLayer/c88cbdf1-1c2a-4a5e-b964-890900000f7d"
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
    [[ -0.148, -1.64, 1.428 ]]
    --------------------
    Output: 
    [ -1.9105430052180221, 0.494696241836107, 0.3562748639173926 ]
```



### Differential Validation
Code from [LayerTestBase.java:139](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L139) executed in 0.00 seconds: 
```java
    getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Component: LogActivationLayer/c88cbdf1-1c2a-4a5e-b964-890900000f7d
    Inputs: [ -0.148, -1.64, 1.428 ]
    output=[ -1.9105430052180221, 0.494696241836107, 0.3562748639173926 ]
    measured/actual: [ [ -6.756756976145084, 0.0, 0.0 ], [ 0.0, -0.6097560956153103, 0.0 ], [ 0.0, 0.0, 0.7002801061251063 ] ]
    implemented/expected: [ [ -6.756756756756757, -0.0, -0.0 ], [ -0.0, -0.6097560975609756, -0.0 ], [ 0.0, 0.0, 0.700280112044818 ] ]
    error: [ [ -2.1938832706069888E-7, 0.0, 0.0 ], [ 0.0, 1.9456652955440745E-9, 0.0 ], [ 0.0, 0.0, -5.919711720814291E-9 ] ]
    Finite-Difference Derivative Accuracy:
    absoluteTol: 2.5250e-08 +- 6.8663e-08 [0.0000e+00 - 2.1939e-07] (9#)
    relativeTol: 7.3523e-09 +- 6.3720e-09 [1.5954e-09 - 1.6235e-08] (3#)
    
```

### Performance
Code from [LayerTestBase.java:144](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.00 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 0.1275 +- 0.0429 [0.0912 - 0.3277]
    Learning performance: 0.0028 +- 0.0035 [0.0000 - 0.0256]
    
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



