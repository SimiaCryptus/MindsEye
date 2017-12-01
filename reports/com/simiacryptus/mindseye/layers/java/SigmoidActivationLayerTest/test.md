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
      "id": "f4569375-56fe-4e46-925c-95f400000a72",
      "isFrozen": true,
      "name": "SigmoidActivationLayer/f4569375-56fe-4e46-925c-95f400000a72",
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
    [[ -1.084, -1.528, -1.752 ]]
    --------------------
    Output: 
    [ -0.4945003991661382, -0.6434270037385942, -0.7044097657869619 ]
```



### Differential Validation
Code from [LayerTestBase.java:139](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L139) executed in 0.00 seconds: 
```java
    getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Feedback for input 0
    Inputs: [ -1.084, -1.528, -1.752 ]
    Output: [ -0.4945003991661382, -0.6434270037385942, -0.7044097657869619 ]
    Measured: [ [ 0.3777440170260604, 0.0, 0.0 ], [ 0.0, 0.2930102717224603, 0.0 ], [ 0.0, 0.0, 0.25191231319610097 ] ]
    Implemented: [ [ 0.377734677612265, 0.0, 0.0 ], [ 0.0, 0.29300084542998756, 0.0 ], [ 0.0, 0.0, 0.25190344093197864 ] ]
    Error: [ [ 9.339413795439455E-6, 0.0, 0.0 ], [ 0.0, 9.426292472747289E-6, 0.0 ], [ 0.0, 0.0, 8.872264122328222E-6 ] ]
    Finite-Difference Derivative Accuracy:
    absoluteTol: 3.0709e-06 +- 4.3452e-06 [0.0000e+00 - 9.4263e-06] (9#)
    relativeTol: 1.5353e-05 +- 2.2042e-06 [1.2362e-05 - 1.7610e-05] (3#)
    
```

### Performance
Code from [LayerTestBase.java:144](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.00 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 0.1535 +- 0.0780 [0.0969 - 0.8179]
    Learning performance: 0.0036 +- 0.0142 [0.0000 - 0.1425]
    
```

### Function Plots
Code from [ActivationLayerTestBase.java:73](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/java/ActivationLayerTestBase.java#L73) executed in 0.00 seconds: 
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



