# HyperbolicActivationLayer
## HyperbolicActivationLayerTest
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
      "class": "com.simiacryptus.mindseye.layers.java.HyperbolicActivationLayer",
      "id": "c88cbdf1-1c2a-4a5e-b964-890900000f71",
      "isFrozen": false,
      "name": "HyperbolicActivationLayer/c88cbdf1-1c2a-4a5e-b964-890900000f71",
      "weights": {
        "dimensions": [
          2
        ],
        "data": [
          1.0,
          1.0
        ]
      },
      "negativeMode": 1
    }
```



### Reference Input/Output Pairs
Code from [LayerTestBase.java:111](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L111) executed in 0.00 seconds: 
```java
    SimpleEval eval = SimpleEval.run(layer, input);
    DoubleStatistics error = new DoubleStatistics().accept(eval.getOutput().add(output.scale(-1)).getData());
    return String.format("--------------------\nInput: \n[%s]\n--------------------\nOutput: \n%s\nError: %s",
      Arrays.stream(input).map(t->t.prettyPrint()).reduce((a,b)->a+",\n"+b).get(),
      eval.getOutput().prettyPrint(), error);
```

Returns: 

```
    --------------------
    Input: 
    [[ 0.0 ]]
    --------------------
    Output: 
    [ 0.0 ]
    Error: 0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (1#)
```



### Differential Validation
Code from [LayerTestBase.java:139](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L139) executed in 0.00 seconds: 
```java
    getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Component: HyperbolicActivationLayer/c88cbdf1-1c2a-4a5e-b964-890900000f71
    Inputs: [ 1.808, 1.936, -1.68 ]
    output=[ 1.0661229392269957, 1.179012620431557, 0.9550959055759898 ]
    measured/actual: [ [ 0.8750746036945145, 0.0, 0.0 ], [ 0.0, 0.8884806411613155, 0.0 ], [ 0.0, 0.0, -0.8592861935041718 ] ]
    implemented/expected: [ [ 0.8750689349959165, 0.0, 0.0 ], [ 0.0, 0.888475808651614, 0.0 ], [ -0.0, -0.0, -0.8592928844097067 ] ]
    error: [ [ 5.668698597971478E-6, 0.0, 0.0 ], [ 0.0, 4.832509701513388E-6, 0.0 ], [ 0.0, 0.0, 6.690905534956215E-6 ] ]
    Component: HyperbolicActivationLayer/c88cbdf1-1c2a-4a5e-b964-890900000f71
    Inputs: [ 1.808, 1.936, -1.68 ]
    Outputs: [ 1.0661229392269957, 1.179012620431557, 0.9550959055759898 ]
    Measured Gradient: [ [ -0.48393138205593544, -0.4588594569066018, 0.0 ], [ 0.0, 0.0, -0.5114138361972387 ] ]
    Implemented Gradient: [ [ -0.4839983047543786, -0.458923454882032, 0.0 ], [ 0.0, 0.0, -0.5114838597676826 ] ]
    Error: [ [ 6.692269844316145E-5, 6.39979754302189E-5, 0.0 ], [ 0.0, 0.0, 7.002357044394447E-5 ] ]
    Finite-Difference Derivative Accuracy:
    absoluteTol: 1.4542e-05 +- 2.6338e-05 [0.0000e+00 - 7.0024e-05] (15#)
    relativeTol: 3.6196e-05 +- 3.2916e-05 [2.7195e-06 - 6.9731e-05] (6#)
    
```

### Performance
Code from [LayerTestBase.java:144](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.00 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 0.1693 +- 0.0388 [0.1140 - 0.2736]
    Learning performance: 0.0574 +- 0.0190 [0.0399 - 0.1795]
    
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



