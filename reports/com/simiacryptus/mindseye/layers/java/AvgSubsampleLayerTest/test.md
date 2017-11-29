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
      "id": "c88cbdf1-1c2a-4a5e-b964-890900000f63",
      "isFrozen": false,
      "name": "AvgSubsampleLayer/c88cbdf1-1c2a-4a5e-b964-890900000f63",
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
    	[ [ -1.468, 0.104, -0.568 ], [ -0.248, -1.68, -1.06 ] ],
    	[ [ -1.812, 1.608, -0.956 ], [ -1.436, -1.888, 1.62 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ -1.241, -0.4639999999999999, -0.241 ] ]
    ]
```



### Differential Validation
Code from [LayerTestBase.java:139](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L139) executed in 0.00 seconds: 
```java
    getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Component: AvgSubsampleLayer/c88cbdf1-1c2a-4a5e-b964-890900000f63
    Inputs: [
    	[ [ -1.468, 0.104, -0.568 ], [ -0.248, -1.68, -1.06 ] ],
    	[ [ -1.812, 1.608, -0.956 ], [ -1.436, -1.888, 1.62 ] ]
    ]
    output=[
    	[ [ -1.241, -0.4639999999999999, -0.241 ] ]
    ]
    measured/actual: [ [ 0.2500000000016378, 0.0, 0.0 ], [ 0.2500000000016378, 0.0, 0.0 ], [ 0.24999999999941735, 0.0, 0.0 ], [ 0.24999999999941735, 0.0, 0.0 ], [ 0.0, 0.24999999999997247, 0.0 ], [ 0.0, 0.24999999999997247, 0.0 ], [ 0.0, 0.24999999999997247, 0.0 ], [ 0.0, 0.24999999999997247, 0.0 ], [ 0.0, 0.0, 0.2500000000005276 ], [ 0.0, 0.0, 0.2500000000005276 ] ]
    implemented/expected: [ [ 0.25, 0.0, 0.0 ], [ 0.25, 0.0, 0.0 ], [ 0.25, 0.0, 0.0 ], [ 0.25, 0.0, 0.0 ], [ 0.0, 0.25, 0.0 ], [ 0.0, 0.25, 0.0 ], [ 0.0, 0.25, 0.0 ], [ 0.0, 0.25, 0.0 ], [ 0.0, 0.0, 0.25 ], [ 0.0, 0.0, 0.25 ] ]
    error: [ [ 1.637801005927031E-12, 0.0, 0.0 ], [ 1.637801005927031E-12, 0.0, 0.0 ], [ -5.826450433232822E-13, 0.0, 0.0 ], [ -5.826450433232822E-13, 0.0, 0.0 ], [ 0.0, -2.7533531010703882E-14, 0.0 ], [ 0.0, -2.7533531010703882E-14, 0.0 ], [ 0.0, -2.7533531010703882E-14, 0.0 ], [ 0.0, -2.7533531010703882E-14, 0.0 ], [ 0.0, 0.0, 5.275779813018744E-13 ], [ 0.0, 0.0, 5.275779813018744E-13 ] ]
    Finite-Difference Derivative Accuracy:
    absoluteTol: 1.7115e-13 +- 4.0236e-13 [0.0000e+00 - 1.6378e-12] (36#)
    relativeTol: 1.0269e-12 +- 1.1134e-12 [5.5067e-14 - 3.2756e-12] (12#)
    
```

### Performance
Code from [LayerTestBase.java:144](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.01 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 0.2510 +- 0.0490 [0.2137 - 0.5415]
    Learning performance: 0.0038 +- 0.0032 [0.0000 - 0.0200]
    
```

