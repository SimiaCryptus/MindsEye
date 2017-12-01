# NthPowerActivationLayer
## SqrtPowerTest
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
      "id": "f4569375-56fe-4e46-925c-95f400000a4a",
      "isFrozen": false,
      "name": "NthPowerActivationLayer/f4569375-56fe-4e46-925c-95f400000a4a",
      "power": 0.5
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
    [[ 1.236, 1.92, -0.136 ]]
    --------------------
    Output: 
    [ 1.1117553687749837, 1.3856406460551018, 0.0 ]
```



### Differential Validation
Code from [LayerTestBase.java:139](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L139) executed in 0.00 seconds: 
```java
    getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Feedback for input 0
    Inputs: [ 1.236, 1.92, -0.136 ]
    Output: [ 1.1117553687749837, 1.3856406460551018, 0.0 ]
    Measured: [ [ 0.4497301305517176, 0.0, 0.0 ], [ 0.0, 0.36083921987728473, 0.0 ], [ 0.0, 0.0, 0.0 ] ]
    Implemented: [ [ 0.44973922685072154, 0.0, 0.0 ], [ 0.0, 0.3608439182435161, 0.0 ], [ 0.0, 0.0, 0.0 ] ]
    Error: [ [ -9.09629900391895E-6, 0.0, 0.0 ], [ 0.0, -4.698366231348228E-6, 0.0 ], [ 0.0, 0.0, 0.0 ] ]
    Finite-Difference Derivative Accuracy:
    absoluteTol: 1.5327e-06 +- 3.0491e-06 [0.0000e+00 - 9.0963e-06] (9#)
    relativeTol: 8.3116e-06 +- 1.8013e-06 [6.5103e-06 - 1.0113e-05] (2#)
    
```

### Performance
Code from [LayerTestBase.java:144](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.00 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 0.1911 +- 0.0477 [0.1368 - 0.4417]
    Learning performance: 0.0032 +- 0.0036 [0.0000 - 0.0314]
    
```

### Function Plots
Code from [ActivationLayerTestBase.java:73](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/java/ActivationLayerTestBase.java#L73) executed in 0.01 seconds: 
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



