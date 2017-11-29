# SigmoidActivationLayer
## SigmoidActivationLayerTest
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
      "class": "com.simiacryptus.mindseye.layers.java.SigmoidActivationLayer",
      "id": "c88cbdf1-1c2a-4a5e-b964-890900000f99",
      "isFrozen": true,
      "name": "SigmoidActivationLayer/c88cbdf1-1c2a-4a5e-b964-890900000f99",
      "balanced": true
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
    [[ -1.58, 0.664, -0.568 ]]
    --------------------
    Output: 
    [ -0.658409035955251, 0.3203167194475147, -0.27660311525848746 ]
```



### Differential Validation
Code from [LayerTestBase.java:139](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L139) executed in 0.00 seconds: 
```java
    getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Component: SigmoidActivationLayer/c88cbdf1-1c2a-4a5e-b964-890900000f99
    Inputs: [ -1.58, 0.664, -0.568 ]
    output=[ -0.658409035955251, 0.3203167194475147, -0.27660311525848746 ]
    measured/actual: [ [ 0.28325809543483516, 0.0, 0.0 ], [ 0.0, 0.44869141307968263, 0.0 ], [ 0.0, 0.0, 0.4617517440275165 ] ]
    implemented/expected: [ [ 0.2832487706862385, 0.0, 0.0 ], [ 0.0, 0.4486985996211911, 0.0 ], [ 0.0, 0.0, 0.46174535831465 ] ]
    error: [ [ 9.32474859666188E-6, 0.0, 0.0 ], [ 0.0, -7.186541508441557E-6, 0.0 ], [ 0.0, 0.0, 6.385712866485704E-6 ] ]
    Finite-Difference Derivative Accuracy:
    absoluteTol: 2.5441e-06 +- 3.6685e-06 [0.0000e+00 - 9.3247e-06] (9#)
    relativeTol: 1.0461e-05 +- 4.2654e-06 [6.9147e-06 - 1.6460e-05] (3#)
    
```

### Performance
Code from [LayerTestBase.java:144](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.00 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 0.1202 +- 0.0380 [0.0940 - 0.3591]
    Learning performance: 0.0014 +- 0.0023 [0.0000 - 0.0171]
    
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



