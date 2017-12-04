# AvgSubsampleLayer
## AvgSubsampleLayerTest
### Json Serialization
Code from [LayerTestBase.java:83](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L83) executed in 0.00 seconds: 
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
      "id": "a864e734-2f23-44db-97c1-504000002b82",
      "isFrozen": false,
      "name": "AvgSubsampleLayer/a864e734-2f23-44db-97c1-504000002b82",
      "inner": [
        2,
        2,
        1
      ]
    }
```



### Example Input/Output Pair
Code from [LayerTestBase.java:120](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L120) executed in 0.00 seconds: 
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
    	[ [ -0.224, -0.82, -0.976 ], [ -1.82, 1.664, -0.504 ] ],
    	[ [ -1.264, 0.336, 1.928 ], [ -1.536, -1.028, -0.06 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ -1.2109999999999999, 0.03799999999999998, 0.09699999999999999 ] ]
    ]
```



### Batch Execution
Code from [LayerTestBase.java:138](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L138) executed in 0.00 seconds: 
```java
    BatchingTester batchingTester = getBatchingTester();
    return batchingTester==null?null:batchingTester.test(layer, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (150#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (150#)}
```



### Differential Validation
Code from [LayerTestBase.java:144](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.00 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ -0.224, -0.82, -0.976 ], [ -1.82, 1.664, -0.504 ] ],
    	[ [ -1.264, 0.336, 1.928 ], [ -1.536, -1.028, -0.06 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.13942534196572562, negative=9, min=-0.06, max=-0.06, mean=-0.35866666666666663, count=12.0, positive=3, stdDev=1.129683534840129, zeros=0}
    Output: [
    	[ [ -1.2109999999999999, 0.03799999999999998, 0.09699999999999999 ] ]
    ]
    Outputs Statistics: {meanExponent=-0.7834335086579644, negative=1, min=0.09699999999999999, max=0.09699999999999999, mean=-0.35866666666666663, count=3.0, positive=2, stdDev=0.6031718015807952, zeros=0}
    Feedback for input 0
    Inputs Values: [
    	[ [ -0.224, -0.82, -0.976 ], [ -1.82, 1.664, -0.504 ] ],
    	[ [ -1.264, 0.336, 1.928 ], [ -1.536, -1.028, -0.06 ] ]
    ]
    Value Statistics: {meanExponent=-0.13942534196572562, negative=9, min=-0.06, max=-0.06, mean=-0.35866666666666663, count=12.0, positive=3, stdDev=1.129683534840129, zeros=0}
    Implemented Feedback: [ [ 0.25, 0.0, 0.0 ], [ 0.25, 0.0, 0.0 ], [ 0.25, 0.0, 0.0 ], [ 0.25, 0.0, 0.0 ], [ 0.0, 0.25, 0.0 ], [ 0.0, 0.25, 0.0 ], [ 0.0, 0.25, 0.0 ], [ 0.0, 0.25, 0.0 ], ... ]
    Implemented Statistics: {meanExponent=-0.6020599913279624, negative=0, min=0.25, max=0.25, mean=0.08333333333333333, count=36.0, positive=12, stdDev=0.11785113019775792, zeros=24}
    Measured Feedback: [ [ 0.24999999999941735, 0.0, 0.0 ], [ 0.24999999999941735, 0.0, 0.0 ], [ 0.24999999999941735, 0.0, 0.0 ], [ 0.24999999999941735, 0.0, 0.0 ], [ 0.0, 0.24999999999997247, 0.0 ], [ 0.0, 0.24999999999997247, 0.0 ], [ 0.0, 0.24999999999997247, 0.0 ], [ 0.0, 0.24999999999997247, 0.0 ], ... ]
    Measured Statistics: {meanExponent=-0.6020599913283319, negative=0, min=0.24999999999997247, max=0.24999999999997247, mean=0.08333333333326248, count=36.0, positive=12, stdDev=0.11785113019765771, zeros=24}
    Feedback Error: [ [ -5.826450433232822E-13, 0.0, 0.0 ], [ -5.826450433232822E-13, 0.0, 0.0 ], [ -5.826450433232822E-13, 0.0, 0.0 ], [ -5.826450433232822E-13, 0.0, 0.0 ], [ 0.0, -2.7533531010703882E-14, 0.0 ], [ 0.0, -2.7533531010703882E-14, 0.0 ], [ 0.0, -2.7533531010703882E-14, 0.0 ], [ 0.0, -2.7533531010703882E-14, 0.0 ], ... ]
    Error Statistics: {meanExponent=-13.118290707517657, negative=12, min=-2.7533531010703882E-14, max=-2.7533531010703882E-14, mean=-7.085690059385443E-14, count=36.0, positive=0, stdDev=1.8129323512325971E-13, zeros=24}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 7.0857e-14 +- 1.8129e-13 [0.0000e+00 - 5.8265e-13] (36#)
    relativeTol: 4.2514e-13 +- 5.2336e-13 [5.5067e-14 - 1.1653e-12] (12#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=7.0857e-14 +- 1.8129e-13 [0.0000e+00 - 5.8265e-13] (36#), relativeTol=4.2514e-13 +- 5.2336e-13 [5.5067e-14 - 1.1653e-12] (12#)}
```



### Performance
Code from [LayerTestBase.java:149](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L149) executed in 0.01 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 0.2635 +- 0.0990 [0.1881 - 0.8321]
    Learning performance: 0.0037 +- 0.0074 [0.0000 - 0.0741]
    
```

