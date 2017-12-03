# AvgSubsampleLayer
## AvgSubsampleLayerTest
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
      "class": "com.simiacryptus.mindseye.layers.java.AvgSubsampleLayer",
      "id": "e2d0bffa-47dc-4875-864f-3d3d00001468",
      "isFrozen": false,
      "name": "AvgSubsampleLayer/e2d0bffa-47dc-4875-864f-3d3d00001468",
      "inner": [
        2,
        2,
        1
      ]
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
    [[
    	[ [ -0.28, 1.024, -0.364 ], [ -0.628, 0.604, 1.112 ] ],
    	[ [ -0.876, -0.528, 0.1 ], [ 0.308, 0.26, -0.624 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ -0.36900000000000005, 0.34, 0.05600000000000002 ] ]
    ]
```



### Differential Validation
Code from [LayerTestBase.java:139](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L139) executed in 0.00 seconds: 
```java
    getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Feedback for input 0
    Inputs: [
    	[ [ -0.28, 1.024, -0.364 ], [ -0.628, 0.604, 1.112 ] ],
    	[ [ -0.876, -0.528, 0.1 ], [ 0.308, 0.26, -0.624 ] ]
    ]
    Output: [
    	[ [ -0.36900000000000005, 0.34, 0.05600000000000002 ] ]
    ]
    Measured: [ [ 0.2500000000005276, 0.0, 0.0 ], [ 0.2500000000005276, 0.0, 0.0 ], [ 0.2500000000005276, 0.0, 0.0 ], [ 0.24999999999997247, 0.0, 0.0 ], [ 0.0, 0.24999999999941735, 0.0 ], [ 0.0, 0.24999999999941735, 0.0 ], [ 0.0, 0.24999999999941735, 0.0 ], [ 0.0, 0.24999999999997247, 0.0 ], [ 0.0, 0.0, 0.24999999999997247 ], [ 0.0, 0.0, 0.24999999999997247 ] ]
    Implemented: [ [ 0.25, 0.0, 0.0 ], [ 0.25, 0.0, 0.0 ], [ 0.25, 0.0, 0.0 ], [ 0.25, 0.0, 0.0 ], [ 0.0, 0.25, 0.0 ], [ 0.0, 0.25, 0.0 ], [ 0.0, 0.25, 0.0 ], [ 0.0, 0.25, 0.0 ], [ 0.0, 0.0, 0.25 ], [ 0.0, 0.0, 0.25 ] ]
    Error: [ [ 5.275779813018744E-13, 0.0, 0.0 ], [ 5.275779813018744E-13, 0.0, 0.0 ], [ 5.275779813018744E-13, 0.0, 0.0 ], [ -2.7533531010703882E-14, 0.0, 0.0 ], [ 0.0, -5.826450433232822E-13, 0.0 ], [ 0.0, -5.826450433232822E-13, 0.0 ], [ 0.0, -5.826450433232822E-13, 0.0 ], [ 0.0, -2.7533531010703882E-14, 0.0 ], [ 0.0, 0.0, -2.7533531010703882E-14 ], [ 0.0, 0.0, -2.7533531010703882E-14 ] ]
    Finite-Difference Derivative Accuracy:
    absoluteTol: 9.7108e-14 +- 2.0538e-13 [0.0000e+00 - 5.8265e-13] (36#)
    relativeTol: 5.8265e-13 +- 5.2901e-13 [5.5067e-14 - 1.1653e-12] (12#)
    
```

### Performance
Code from [LayerTestBase.java:144](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.01 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 0.3113 +- 0.1180 [0.2137 - 0.9575]
    Learning performance: 0.0037 +- 0.0041 [0.0000 - 0.0314]
    
```

