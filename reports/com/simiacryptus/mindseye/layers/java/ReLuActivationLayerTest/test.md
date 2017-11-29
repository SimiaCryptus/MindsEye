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
      "id": "c88cbdf1-1c2a-4a5e-b964-890900000f96",
      "isFrozen": true,
      "name": "ReLuActivationLayer/c88cbdf1-1c2a-4a5e-b964-890900000f96",
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
    [[ -1.424, -0.936, 0.492 ]]
    --------------------
    Output: 
    [ 0.0, 0.0, 0.492 ]
```



### Differential Validation
Code from [LayerTestBase.java:139](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L139) executed in 0.00 seconds: 
```java
    getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Component: ReLuActivationLayer/c88cbdf1-1c2a-4a5e-b964-890900000f96
    Inputs: [ -1.424, -0.936, 0.492 ]
    output=[ 0.0, 0.0, 0.492 ]
    measured/actual: [ [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.9999999999998899 ] ]
    implemented/expected: [ [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 1.0 ] ]
    error: [ [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, -1.1013412404281553E-13 ] ]
    Component: ReLuActivationLayer/c88cbdf1-1c2a-4a5e-b964-890900000f96
    Inputs: [ -1.424, -0.936, 0.492 ]
    Outputs: [ 0.0, 0.0, 0.492 ]
    Measured Gradient: [ [ 0.0, 0.0, 0.4919999999997149 ] ]
    Implemented Gradient: [ [ 0.0, 0.0, 0.492 ] ]
    Error: [ [ 0.0, 0.0, -2.851052727237402E-13 ] ]
    Finite-Difference Derivative Accuracy:
    absoluteTol: 3.2937e-14 +- 8.1852e-14 [0.0000e+00 - 2.8511e-13] (12#)
    relativeTol: 1.7240e-13 +- 1.1734e-13 [5.5067e-14 - 2.8974e-13] (2#)
    
```

### Performance
Code from [LayerTestBase.java:144](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.00 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 0.1895 +- 0.0460 [0.1396 - 0.3306]
    Learning performance: 0.0541 +- 0.0173 [0.0371 - 0.1567]
    
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



