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
      "id": "e2d0bffa-47dc-4875-864f-3d3d00001498",
      "isFrozen": true,
      "name": "EntropyLayer/e2d0bffa-47dc-4875-864f-3d3d00001498"
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
    [[ 1.524, 1.516, -0.504 ]]
    --------------------
    Output: 
    [ -0.6421198088710288, -0.6307701354257929, -0.34533022149902726 ]
```



### Differential Validation
Code from [LayerTestBase.java:139](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L139) executed in 0.00 seconds: 
```java
    getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Feedback for input 0
    Inputs: [ 1.524, 1.516, -0.504 ]
    Output: [ -0.6421198088710288, -0.6307701354257929, -0.34533022149902726 ]
    Measured: [ [ -1.421371264944682, 0.0, 0.0 ], [ 0.0, -1.4161082680241854, 0.0 ], [ 0.0, 0.0, -0.3147217761784171 ] ]
    Implemented: [ [ -1.4213384572644545, 0.0, 0.0 ], [ 0.0, -1.4160752872201798, 0.0 ], [ 0.0, 0.0, -0.3148209890892316 ] ]
    Error: [ [ -3.2807680227397995E-5, 0.0, 0.0 ], [ 0.0, -3.2980804005600106E-5, 0.0 ], [ 0.0, 0.0, 9.921291081449457E-5 ] ]
    Finite-Difference Derivative Accuracy:
    absoluteTol: 1.8333e-05 +- 3.1592e-05 [0.0000e+00 - 9.9213e-05] (9#)
    relativeTol: 6.0260e-05 +- 6.8826e-05 [1.1541e-05 - 1.5760e-04] (3#)
    
```

### Performance
Code from [LayerTestBase.java:144](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.00 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 0.1272 +- 0.0389 [0.0969 - 0.3391]
    Learning performance: 0.0015 +- 0.0025 [0.0000 - 0.0171]
    
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



