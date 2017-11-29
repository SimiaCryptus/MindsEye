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
      "id": "c88cbdf1-1c2a-4a5e-b964-890900000f6f",
      "isFrozen": true,
      "name": "GaussianActivationLayer/c88cbdf1-1c2a-4a5e-b964-890900000f6f",
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
    [[ -1.688, 0.564, -1.24 ]]
    --------------------
    Output: 
    [ 0.09598047120031673, 0.3402799787902737, 0.1849372809633053 ]
```



### Differential Validation
Code from [LayerTestBase.java:139](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L139) executed in 0.00 seconds: 
```java
    getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Component: GaussianActivationLayer/c88cbdf1-1c2a-4a5e-b964-890900000f6f
    Inputs: [ -1.688, 0.564, -1.24 ]
    output=[ 0.09598047120031673, 0.3402799787902737, 0.1849372809633053 ]
    measured/actual: [ [ 0.16202391039082942, 0.0, 0.0 ], [ 0.0, -0.1919295090940798, 0.0 ], [ 0.0, 0.0, 0.2293271989495249 ] ]
    implemented/expected: [ [ 0.16201503538613465, 0.0, 0.0 ], [ -0.0, -0.19191790803771436, -0.0 ], [ 0.0, 0.0, 0.2293222283944986 ] ]
    error: [ [ 8.875004694763877E-6, 0.0, 0.0 ], [ 0.0, -1.1601056365440066E-5, 0.0 ], [ 0.0, 0.0, 4.9705550263234866E-6 ] ]
    Finite-Difference Derivative Accuracy:
    absoluteTol: 2.8274e-06 +- 4.2961e-06 [0.0000e+00 - 1.1601e-05] (9#)
    relativeTol: 2.2816e-05 +- 8.5491e-06 [1.0837e-05 - 3.0223e-05] (3#)
    
```

### Performance
Code from [LayerTestBase.java:144](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.00 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 0.2258 +- 0.4739 [0.1111 - 4.8674]
    Learning performance: 0.0027 +- 0.0049 [0.0000 - 0.0427]
    
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



