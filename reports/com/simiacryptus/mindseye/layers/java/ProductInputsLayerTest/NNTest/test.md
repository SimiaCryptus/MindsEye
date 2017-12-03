# ProductInputsLayer
## NNTest
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
      "class": "com.simiacryptus.mindseye.layers.java.ProductInputsLayer",
      "id": "e2d0bffa-47dc-4875-864f-3d3d00001546",
      "isFrozen": false,
      "name": "ProductInputsLayer/e2d0bffa-47dc-4875-864f-3d3d00001546"
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
    [[ 0.328, -0.86, 1.46 ],
    [ -0.348, -0.268, -0.708 ]]
    --------------------
    Output: 
    [ -0.114144, 0.23048000000000002, -1.03368 ]
```



### Differential Validation
Code from [LayerTestBase.java:139](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L139) executed in 0.00 seconds: 
```java
    getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Feedback for input 0
    Inputs: [ 0.328, -0.86, 1.46 ],
    [ -0.348, -0.268, -0.708 ]
    Output: [ -0.114144, 0.23048000000000002, -1.03368 ]
    Measured: [ [ -0.34800000000001496, 0.0, 0.0 ], [ 0.0, -0.26799999999993496, 0.0 ], [ 0.0, 0.0, -0.708000000000375 ] ]
    Implemented: [ [ -0.348, 0.0, 0.0 ], [ 0.0, -0.268, 0.0 ], [ 0.0, 0.0, -0.708 ] ]
    Error: [ [ -1.4988010832439613E-14, 0.0, 0.0 ], [ 0.0, 6.505906924303417E-14, 0.0 ], [ 0.0, 0.0, -3.750333377183779E-13 ] ]
    Feedback for input 1
    Inputs: [ 0.328, -0.86, 1.46 ],
    [ -0.348, -0.268, -0.708 ]
    Output: [ -0.114144, 0.23048000000000002, -1.03368 ]
    Measured: [ [ 0.32799999999999496, 0.0, 0.0 ], [ 0.0, -0.8600000000000274, 0.0 ], [ 0.0, 0.0, 1.4599999999997948 ] ]
    Implemented: [ [ 0.328, 0.0, 0.0 ], [ 0.0, -0.86, 0.0 ], [ 0.0, 0.0, 1.46 ] ]
    Error: [ [ -5.051514762044462E-15, 0.0, 0.0 ], [ 0.0, -2.7422508708241367E-14, 0.0 ], [ 0.0, 0.0, -2.0516921495072893E-13 ] ]
    Finite-Difference Derivative Accuracy:
    absoluteTol: 3.8485e-14 +- 9.4669e-14 [0.0000e+00 - 3.7503e-13] (18#)
    relativeTol: 8.3612e-14 +- 9.0045e-14 [7.7005e-15 - 2.6485e-13] (6#)
    
```

### Performance
Code from [LayerTestBase.java:144](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.00 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 0.2170 +- 0.0756 [0.1624 - 0.7295]
    Learning performance: 0.0177 +- 0.0106 [0.0086 - 0.0769]
    
```

