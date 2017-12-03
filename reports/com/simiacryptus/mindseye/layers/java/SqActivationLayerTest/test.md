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
      "id": "e2d0bffa-47dc-4875-864f-3d3d00001568",
      "isFrozen": true,
      "name": "SqActivationLayer/e2d0bffa-47dc-4875-864f-3d3d00001568"
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
    [[ -0.436, -1.624, -1.196 ]]
    --------------------
    Output: 
    [ 0.190096, 2.637376, 1.430416 ]
```



### Differential Validation
Code from [LayerTestBase.java:139](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L139) executed in 0.00 seconds: 
```java
    getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Feedback for input 0
    Inputs: [ -0.436, -1.624, -1.196 ]
    Output: [ 0.190096, 2.637376, 1.430416 ]
    Measured: [ [ -0.87189999999987, 0.0, 0.0 ], [ 0.0, -3.2478999999963065, 0.0 ], [ 0.0, 0.0, -2.3919000000005575 ] ]
    Implemented: [ [ -0.872, 0.0, 0.0 ], [ 0.0, -3.248, 0.0 ], [ 0.0, 0.0, -2.392 ] ]
    Error: [ [ 1.000000001299961E-4, 0.0, 0.0 ], [ 0.0, 1.0000000369370099E-4, 0.0 ], [ 0.0, 0.0, 9.999999944243498E-5 ] ]
    Finite-Difference Derivative Accuracy:
    absoluteTol: 3.3333e-05 +- 4.7140e-05 [0.0000e+00 - 1.0000e-04] (9#)
    relativeTol: 3.1214e-05 +- 1.8613e-05 [1.5394e-05 - 5.7343e-05] (3#)
    
```

### Performance
Code from [LayerTestBase.java:144](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.00 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 0.1434 +- 0.0655 [0.0884 - 0.7210]
    Learning performance: 0.0011 +- 0.0018 [0.0000 - 0.0114]
    
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



