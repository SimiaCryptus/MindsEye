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
      "id": "e2d0bffa-47dc-4875-864f-3d3d000014ab",
      "isFrozen": false,
      "name": "HyperbolicActivationLayer/e2d0bffa-47dc-4875-864f-3d3d000014ab",
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
    Feedback for input 0
    Inputs: [ -1.348, -0.388, -0.54 ]
    Output: [ 0.6784230694315425, 0.07263414079545316, 0.13648581161402973 ]
    Measured: [ [ -0.8031242390416082, 0.0, 0.0 ], [ 0.0, -0.3616858032717829, 0.0 ], [ 0.0, 0.0, -0.47511485069762216 ] ]
    Implemented: [ [ -0.8031348141899337, 0.0, 0.0 ], [ 0.0, -0.3617263195745976, 0.0 ], [ 0.0, 0.0, -0.47514891473488396 ] ]
    Error: [ [ 1.057514832547568E-5, 0.0, 0.0 ], [ 0.0, 4.0516302814708194E-5, 0.0 ], [ 0.0, 0.0, 3.406403726180507E-5 ] ]
    Learning Gradient for weight set 0
    Inputs: [ -1.348, -0.388, -0.54 ]
    Outputs: [ 0.6784230694315425, 0.07263414079545316, 0.13648581161402973 ]
    Measured Gradient: [ [ 0.0, 0.0, 0.0 ], [ -0.5957185540650389, -0.9321850110985408, -0.8798074844729165 ] ]
    Implemented Gradient: [ [ 0.0, 0.0, 0.0 ], [ -0.5957973399035116, -0.9322843288005093, -0.8799053976571926 ] ]
    Error: [ [ 0.0, 0.0, 0.0 ], [ 7.87858384726503E-5, 9.9317701968471E-5, 9.791318427609941E-5 ] ]
    Finite-Difference Derivative Accuracy:
    absoluteTol: 2.4078e-05 +- 3.6401e-05 [0.0000e+00 - 9.9318e-05] (15#)
    relativeTol: 4.5578e-05 +- 1.9607e-05 [6.5837e-06 - 6.6122e-05] (6#)
    
```

### Performance
Code from [LayerTestBase.java:144](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.00 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 0.1686 +- 0.0473 [0.1111 - 0.3477]
    Learning performance: 0.0574 +- 0.0205 [0.0370 - 0.1824]
    
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



