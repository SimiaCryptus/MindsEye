# AvgSubsampleLayer
## AvgSubsampleLayerTest
### Json Serialization
Code from [LayerTestBase.java:83](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L83) executed in 0.00 seconds: 
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
      "id": "370a9587-74a1-4959-b406-fa4500002b82",
      "isFrozen": false,
      "name": "AvgSubsampleLayer/370a9587-74a1-4959-b406-fa4500002b82",
      "inner": [
        2,
        2,
        1
      ]
    }
```



### Example Input/Output Pair
Code from [LayerTestBase.java:120](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L120) executed in 0.00 seconds: 
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
    	[ [ -0.068, 0.832, -1.64 ], [ -1.556, -0.296, 1.496 ] ],
    	[ [ -1.484, -1.888, 0.504 ], [ 1.076, 0.936, -0.464 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ -0.508, -0.10400000000000001, -0.02599999999999998 ] ]
    ]
```



### Batch Execution
Code from [LayerTestBase.java:138](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L138) executed in 0.00 seconds: 
```java
    BatchingTester batchingTester = getBatchingTester();
    return batchingTester==null?null:batchingTester.test(layer, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (150#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (150#)}
```



### Differential Validation
Code from [LayerTestBase.java:144](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.00 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ -0.068, 0.832, -1.64 ], [ -1.556, -0.296, 1.496 ] ],
    	[ [ -1.484, -1.888, 0.504 ], [ 1.076, 0.936, -0.464 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.11456820731169587, negative=7, min=-0.464, max=-0.464, mean=-0.21266666666666667, count=12.0, positive=5, stdDev=1.1496548274252678, zeros=0}
    Output: [
    	[ [ -0.508, -0.10400000000000001, -0.02599999999999998 ] ]
    ]
    Outputs Statistics: {meanExponent=-0.9540432001488276, negative=3, min=-0.02599999999999998, max=-0.02599999999999998, mean=-0.21266666666666667, count=3.0, positive=0, stdDev=0.21124603875312997, zeros=0}
    Feedback for input 0
    Inputs Values: [
    	[ [ -0.068, 0.832, -1.64 ], [ -1.556, -0.296, 1.496 ] ],
    	[ [ -1.484, -1.888, 0.504 ], [ 1.076, 0.936, -0.464 ] ]
    ]
    Value Statistics: {meanExponent=-0.11456820731169587, negative=7, min=-0.464, max=-0.464, mean=-0.21266666666666667, count=12.0, positive=5, stdDev=1.1496548274252678, zeros=0}
    Implemented Feedback: [ [ 0.25, 0.0, 0.0 ], [ 0.25, 0.0, 0.0 ], [ 0.25, 0.0, 0.0 ], [ 0.25, 0.0, 0.0 ], [ 0.0, 0.25, 0.0 ], [ 0.0, 0.25, 0.0 ], [ 0.0, 0.25, 0.0 ], [ 0.0, 0.25, 0.0 ], ... ]
    Implemented Statistics: {meanExponent=-0.6020599913279624, negative=0, min=0.25, max=0.25, mean=0.08333333333333333, count=36.0, positive=12, stdDev=0.11785113019775792, zeros=24}
    Measured Feedback: [ [ 0.2500000000005276, 0.0, 0.0 ], [ 0.2500000000005276, 0.0, 0.0 ], [ 0.2500000000005276, 0.0, 0.0 ], [ 0.24999999999941735, 0.0, 0.0 ], [ 0.0, 0.2500000000005276, 0.0 ], [ 0.0, 0.2500000000005276, 0.0 ], [ 0.0, 0.24999999999997247, 0.0 ], [ 0.0, 0.24999999999997247, 0.0 ], ... ]
    Measured Statistics: {meanExponent=-0.6020599913276888, negative=0, min=0.24999999999997247, max=0.24999999999997247, mean=0.08333333333338583, count=36.0, positive=12, stdDev=0.11785113019783218, zeros=24}
    Feedback Error: [ [ 5.275779813018744E-13, 0.0, 0.0 ], [ 5.275779813018744E-13, 0.0, 0.0 ], [ 5.275779813018744E-13, 0.0, 0.0 ], [ -5.826450433232822E-13, 0.0, 0.0 ], [ 0.0, 5.275779813018744E-13, 0.0 ], [ 0.0, 5.275779813018744E-13, 0.0 ], [ 0.0, -2.7533531010703882E-14, 0.0 ], [ 0.0, -2.7533531010703882E-14, 0.0 ], ... ]
    Error Statistics: {meanExponent=-12.915332597591787, negative=7, min=-2.7533531010703882E-14, max=-2.7533531010703882E-14, mean=5.2501213253385183E-14, count=36.0, positive=5, stdDev=2.1320874383458044E-13, zeros=24}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 9.4048e-14 +- 1.9842e-13 [0.0000e+00 - 5.8265e-13] (36#)
    relativeTol: 5.6429e-13 +- 5.1005e-13 [5.5067e-14 - 1.1653e-12] (12#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=9.4048e-14 +- 1.9842e-13 [0.0000e+00 - 5.8265e-13] (36#), relativeTol=5.6429e-13 +- 5.1005e-13 [5.5067e-14 - 1.1653e-12] (12#)}
```



### Performance
Code from [LayerTestBase.java:149](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L149) executed in 0.01 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 0.2527 +- 0.0827 [0.1966 - 0.6897]
    Learning performance: 0.0041 +- 0.0052 [0.0000 - 0.0485]
    
```

