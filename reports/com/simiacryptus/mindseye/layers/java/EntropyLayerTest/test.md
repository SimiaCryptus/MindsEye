# EntropyLayer
## EntropyLayerTest
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
      "class": "com.simiacryptus.mindseye.layers.java.EntropyLayer",
      "id": "c88cbdf1-1c2a-4a5e-b964-890900000f6c",
      "isFrozen": true,
      "name": "EntropyLayer/c88cbdf1-1c2a-4a5e-b964-890900000f6c"
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
    [[ 0.652, 1.6, -1.576 ]]
    --------------------
    Output: 
    [ 0.27886738752017565, -0.7520058067931771, 0.7169066265026434 ]
```



### Differential Validation
Code from [LayerTestBase.java:139](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L139) executed in 0.00 seconds: 
```java
    getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Component: EntropyLayer/c88cbdf1-1c2a-4a5e-b964-890900000f6c
    Inputs: [ 0.652, 1.6, -1.576 ]
    output=[ 0.27886738752017565, -0.7520058067931771, 0.7169066265026434 ]
    measured/actual: [ [ -0.5723659661410752, 0.0, 0.0 ], [ 0.0, -1.4700348785945394, 0.0 ], [ 0.0, 0.0, -1.4548582648754582 ] ]
    implemented/expected: [ [ -0.5722892829445159, -0.0, -0.0 ], [ -0.0, -1.4700036292457357, -0.0 ], [ -0.0, -0.0, -1.4548899914356874 ] ]
    error: [ [ -7.668319655929068E-5, 0.0, 0.0 ], [ 0.0, -3.12493488037191E-5, 0.0 ], [ 0.0, 0.0, 3.172656022920606E-5 ] ]
    Finite-Difference Derivative Accuracy:
    absoluteTol: 1.5518e-05 +- 2.5158e-05 [0.0000e+00 - 7.6683e-05] (9#)
    relativeTol: 2.9508e-05 +- 2.6506e-05 [1.0629e-05 - 6.6992e-05] (3#)
    
```

### Performance
Code from [LayerTestBase.java:144](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.00 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 0.1600 +- 0.0355 [0.1026 - 0.2650]
    Learning performance: 0.0032 +- 0.0025 [0.0000 - 0.0171]
    
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



