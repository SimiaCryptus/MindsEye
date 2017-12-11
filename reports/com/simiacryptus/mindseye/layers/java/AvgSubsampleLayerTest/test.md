# AvgSubsampleLayer
## AvgSubsampleLayerTest
### Json Serialization
Code from [LayerTestBase.java:121](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L121) executed in 0.00 seconds: 
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
      "id": "e2a3bda5-e7e7-4c05-aeb3-4ede00001e0d",
      "isFrozen": false,
      "name": "AvgSubsampleLayer/e2a3bda5-e7e7-4c05-aeb3-4ede00001e0d",
      "inner": [
        2,
        2,
        1
      ]
    }
```



### Example Input/Output Pair
Code from [LayerTestBase.java:159](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L159) executed in 0.00 seconds: 
```java
    SimpleEval eval = SimpleEval.run(layer, inputPrototype);
    return String.format("--------------------\nInput: \n[%s]\n--------------------\nOutput: \n%s",
      Arrays.stream(inputPrototype).map(t -> t.prettyPrint()).reduce((a, b) -> a + ",\n" + b).get(),
      eval.getOutput().prettyPrint());
```

Returns: 

```
    --------------------
    Input: 
    [[
    	[ [ 1.128, 0.22, 0.308 ], [ 0.184, -0.644, 1.184 ] ],
    	[ [ 1.92, 0.344, -1.94 ], [ 1.304, 0.656, -0.312 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 1.1340000000000001, 0.144, -0.19 ] ]
    ]
```



### Batch Execution
Code from [LayerTestBase.java:178](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L178) executed in 0.00 seconds: 
```java
    BatchingTester batchingTester = getBatchingTester();
    return batchingTester == null ? null : batchingTester.test(layer, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (150#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (150#)}
```



### Differential Validation
Code from [LayerTestBase.java:186](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L186) executed in 0.00 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ 1.128, 0.22, 0.308 ], [ 0.184, -0.644, 1.184 ] ],
    	[ [ 1.92, 0.344, -1.94 ], [ 1.304, 0.656, -0.312 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.20297205702320553, negative=3, min=-0.312, max=-0.312, mean=0.36266666666666664, count=12.0, positive=9, stdDev=0.978272400146753, zeros=0}
    Output: [
    	[ [ 1.1340000000000001, 0.144, -0.19 ] ]
    ]
    Outputs Statistics: {meanExponent=-0.5027569507983446, negative=1, min=-0.19, max=-0.19, mean=0.3626666666666667, count=3.0, positive=2, stdDev=0.5622012292962568, zeros=0}
    Feedback for input 0
    Inputs Values: [
    	[ [ 1.128, 0.22, 0.308 ], [ 0.184, -0.644, 1.184 ] ],
    	[ [ 1.92, 0.344, -1.94 ], [ 1.304, 0.656, -0.312 ] ]
    ]
    Value Statistics: {meanExponent=-0.20297205702320553, negative=3, min=-0.312, max=-0.312, mean=0.36266666666666664, count=12.0, positive=9, stdDev=0.978272400146753, zeros=0}
    Implemented Feedback: [ [ 0.25, 0.0, 0.0 ], [ 0.25, 0.0, 0.0 ], [ 0.25, 0.0, 0.0 ], [ 0.25, 0.0, 0.0 ], [ 0.0, 0.25, 0.0 ], [ 0.0, 0.25, 0.0 ], [ 0.0, 0.25, 0.0 ], [ 0.0
```
...[skipping 429 bytes](etc/44.txt)...
```
    0 ], [ 0.0, 0.24999999999997247, 0.0 ], [ 0.0, 0.24999999999997247, 0.0 ], ... ]
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
Code from [LayerTestBase.java:192](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L192) executed in 0.01 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 0.3572 +- 0.3906 [0.2023 - 3.1348]
    Learning performance: 0.0027 +- 0.0028 [0.0000 - 0.0200]
    
```

