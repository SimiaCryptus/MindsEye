# SqActivationLayer
## SqActivationLayerTest
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
      "class": "com.simiacryptus.mindseye.layers.java.SqActivationLayer",
      "id": "c88cbdf1-1c2a-4a5e-b964-890900000f9c",
      "isFrozen": true,
      "name": "SqActivationLayer/c88cbdf1-1c2a-4a5e-b964-890900000f9c"
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
    [[ -1.072, 1.352, 1.232 ]]
    --------------------
    Output: 
    [ 1.1491840000000002, 1.8279040000000002, 1.517824 ]
```



### Differential Validation
Code from [LayerTestBase.java:139](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L139) executed in 0.00 seconds: 
```java
    getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Component: SqActivationLayer/c88cbdf1-1c2a-4a5e-b964-890900000f9c
    Inputs: [ -1.072, 1.352, 1.232 ]
    output=[ 1.1491840000000002, 1.8279040000000002, 1.517824 ]
    measured/actual: [ [ -2.1439000000000874, 0.0, 0.0 ], [ 0.0, 2.704099999999432, 0.0 ], [ 0.0, 0.0, 2.464099999999192 ] ]
    implemented/expected: [ [ -2.144, -0.0, -0.0 ], [ 0.0, 2.704, 0.0 ], [ 0.0, 0.0, 2.464 ] ]
    error: [ [ 9.999999991272546E-5, 0.0, 0.0 ], [ 0.0, 9.999999943177684E-5, 0.0 ], [ 0.0, 0.0, 9.999999919196867E-5 ] ]
    Finite-Difference Derivative Accuracy:
    absoluteTol: 3.3333e-05 +- 4.7140e-05 [0.0000e+00 - 1.0000e-04] (9#)
    relativeTol: 2.0701e-05 +- 1.9933e-06 [1.8491e-05 - 2.3321e-05] (3#)
    
```

### Performance
Code from [LayerTestBase.java:144](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.00 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 0.1291 +- 0.0393 [0.0912 - 0.4104]
    Learning performance: 0.0015 +- 0.0016 [0.0000 - 0.0086]
    
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



