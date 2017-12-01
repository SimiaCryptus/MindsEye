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
      "id": "f4569375-56fe-4e46-925c-95f40000097c",
      "isFrozen": false,
      "name": "AvgSubsampleLayer/f4569375-56fe-4e46-925c-95f40000097c",
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
    	[ [ -1.56, -1.532, 0.704 ], [ -1.724, -0.612, 1.184 ] ],
    	[ [ 0.5, -2.0, 0.256 ], [ -0.14, -0.448, -1.588 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ -0.731, -1.1480000000000001, 0.139 ] ]
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
    	[ [ -1.56, -1.532, 0.704 ], [ -1.724, -0.612, 1.184 ] ],
    	[ [ 0.5, -2.0, 0.256 ], [ -0.14, -0.448, -1.588 ] ]
    ]
    Output: [
    	[ [ -0.731, -1.1480000000000001, 0.139 ] ]
    ]
    Measured: [ [ 0.24999999999941735, 0.0, 0.0 ], [ 0.24999999999941735, 0.0, 0.0 ], [ 0.24999999999941735, 0.0, 0.0 ], [ 0.2500000000005276, 0.0, 0.0 ], [ 0.0, 0.24999999999941735, 0.0 ], [ 0.0, 0.24999999999941735, 0.0 ], [ 0.0, 0.24999999999941735, 0.0 ], [ 0.0, 0.2500000000016378, 0.0 ], [ 0.0, 0.0, 0.24999999999941735 ], [ 0.0, 0.0, 0.24999999999941735 ] ]
    Implemented: [ [ 0.25, 0.0, 0.0 ], [ 0.25, 0.0, 0.0 ], [ 0.25, 0.0, 0.0 ], [ 0.25, 0.0, 0.0 ], [ 0.0, 0.25, 0.0 ], [ 0.0, 0.25, 0.0 ], [ 0.0, 0.25, 0.0 ], [ 0.0, 0.25, 0.0 ], [ 0.0, 0.0, 0.25 ], [ 0.0, 0.0, 0.25 ] ]
    Error: [ [ -5.826450433232822E-13, 0.0, 0.0 ], [ -5.826450433232822E-13, 0.0, 0.0 ], [ -5.826450433232822E-13, 0.0, 0.0 ], [ 5.275779813018744E-13, 0.0, 0.0 ], [ 0.0, -5.826450433232822E-13, 0.0 ], [ 0.0, -5.826450433232822E-13, 0.0 ], [ 0.0, -5.826450433232822E-13, 0.0 ], [ 0.0, 1.637801005927031E-12, 0.0 ], [ 0.0, 0.0, -5.826450433232822E-13 ], [ 0.0, 0.0, -5.826450433232822E-13 ] ]
    Finite-Difference Derivative Accuracy:
    absoluteTol: 2.0658e-13 +- 3.5279e-13 [0.0000e+00 - 1.6378e-12] (36#)
    relativeTol: 1.2395e-12 +- 6.8509e-13 [5.5067e-14 - 3.2756e-12] (12#)
    
```

### Performance
Code from [LayerTestBase.java:144](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.01 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 0.2746 +- 0.1022 [0.2194 - 0.8692]
    Learning performance: 0.0030 +- 0.0021 [0.0000 - 0.0199]
    
```

