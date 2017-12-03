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
      "id": "e2d0bffa-47dc-4875-864f-3d3d0000155b",
      "isFrozen": true,
      "name": "SigmoidActivationLayer/e2d0bffa-47dc-4875-864f-3d3d0000155b",
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
    [[ 1.228, 0.4, -1.608 ]]
    --------------------
    Output: 
    [ 0.5469366716116726, 0.197375320224904, -0.6662670550794461 ]
```



### Differential Validation
Code from [LayerTestBase.java:139](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L139) executed in 0.00 seconds: 
```java
    getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Feedback for input 0
    Inputs: [ 1.228, 0.4, -1.608 ]
    Output: [ 0.5469366716116726, 0.197375320224904, -0.6662670550794461 ]
    Measured: [ [ 0.3504205554372142, 0.0, 0.0 ], [ 0.0, 0.4805167489752016, 0.0 ], [ 0.0, 0.0, 0.2780533683166553 ] ]
    Implemented: [ [ 0.35043013862317257, 0.0, 0.0 ], [ 0.0, 0.48052149148305834, 0.0 ], [ 0.0, 0.0, 0.2780441056578812 ] ]
    Error: [ [ -9.583185958395024E-6, 0.0, 0.0 ], [ 0.0, -4.742507856758671E-6, 0.0 ], [ 0.0, 0.0, 9.262658774100707E-6 ] ]
    Finite-Difference Derivative Accuracy:
    absoluteTol: 2.6209e-06 +- 3.9201e-06 [0.0000e+00 - 9.5832e-06] (9#)
    relativeTol: 1.1755e-05 +- 4.9740e-06 [4.9348e-06 - 1.6657e-05] (3#)
    
```

### Performance
Code from [LayerTestBase.java:144](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.00 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 0.1325 +- 0.0244 [0.1054 - 0.2565]
    Learning performance: 0.0007 +- 0.0013 [0.0000 - 0.0057]
    
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



