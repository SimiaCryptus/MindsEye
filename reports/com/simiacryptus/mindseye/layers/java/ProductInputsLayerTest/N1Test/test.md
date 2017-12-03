# ProductInputsLayer
## N1Test
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
      "id": "e2d0bffa-47dc-4875-864f-3d3d0000153f",
      "isFrozen": false,
      "name": "ProductInputsLayer/e2d0bffa-47dc-4875-864f-3d3d0000153f"
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
    [[ -1.34, -0.892, -1.44 ],
    [ -0.212 ]]
    --------------------
    Output: 
    [ 0.28408, 0.189104, 0.30528 ]
```



### Differential Validation
Code from [LayerTestBase.java:139](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L139) executed in 0.00 seconds: 
```java
    getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Feedback for input 0
    Inputs: [ -1.34, -0.892, -1.44 ],
    [ -0.212 ]
    Output: [ 0.28408, 0.189104, 0.30528 ]
    Measured: [ [ -0.21199999999998997, 0.0, 0.0 ], [ 0.0, -0.21199999999998997, 0.0 ], [ 0.0, 0.0, -0.21199999999998997 ] ]
    Implemented: [ [ -0.212, 0.0, 0.0 ], [ 0.0, -0.212, 0.0 ], [ 0.0, 0.0, -0.212 ] ]
    Error: [ [ 1.0019762797242038E-14, 0.0, 0.0 ], [ 0.0, 1.0019762797242038E-14, 0.0 ], [ 0.0, 0.0, 1.0019762797242038E-14 ] ]
    Feedback for input 1
    Inputs: [ -1.34, -0.892, -1.44 ],
    [ -0.212 ]
    Output: [ 0.28408, 0.189104, 0.30528 ]
    Measured: [ [ -1.3399999999996748, -0.8919999999998374, -1.4399999999997748 ] ]
    Implemented: [ [ -1.34, -0.892, -1.44 ] ]
    Error: [ [ 3.2529534621517087E-13, 1.6264767310758543E-13, 2.2515322939398175E-13 ] ]
    Finite-Difference Derivative Accuracy:
    absoluteTol: 6.1930e-14 +- 1.0694e-13 [0.0000e+00 - 3.2530e-13] (12#)
    relativeTol: 6.0270e-14 +- 3.8809e-14 [2.3632e-14 - 1.2138e-13] (6#)
    
```

### Performance
Code from [LayerTestBase.java:144](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.01 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 0.2489 +- 0.0653 [0.1995 - 0.5244]
    Learning performance: 0.0185 +- 0.0097 [0.0114 - 0.0741]
    
```

