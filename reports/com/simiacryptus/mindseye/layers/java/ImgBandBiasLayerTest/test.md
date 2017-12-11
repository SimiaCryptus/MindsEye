# ImgBandBiasLayer
## ImgBandBiasLayerTest
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
      "class": "com.simiacryptus.mindseye.layers.java.ImgBandBiasLayer",
      "id": "e2a3bda5-e7e7-4c05-aeb3-4ede00001e6c",
      "isFrozen": false,
      "name": "ImgBandBiasLayer/e2a3bda5-e7e7-4c05-aeb3-4ede00001e6c",
      "bias": [
        0.0,
        0.0,
        0.0
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
    	[ [ -1.76, -1.992, 0.864 ], [ -1.74, 1.548, -1.32 ] ],
    	[ [ -1.88, -1.288, -1.928 ], [ -0.24, -0.368, -0.452 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ -1.76, -1.992, 0.864 ], [ -1.74, 1.548, -1.32 ] ],
    	[ [ -1.88, -1.288, -1.928 ], [ -0.24, -0.368, -0.452 ] ]
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
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (240#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (239#)}
```



### Differential Validation
Code from [LayerTestBase.java:186](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L186) executed in 0.01 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ -1.76, -1.992, 0.864 ], [ -1.74, 1.548, -1.32 ] ],
    	[ [ -1.88, -1.288, -1.928 ], [ -0.24, -0.368, -0.452 ] ]
    ]
    Inputs Statistics: {meanExponent=0.0252156764389031, negative=10, min=-0.452, max=-0.452, mean=-0.8796666666666667, count=12.0, positive=2, stdDev=1.118054808833429, zeros=0}
    Output: [
    	[ [ -1.76, -1.992, 0.864 ], [ -1.74, 1.548, -1.32 ] ],
    	[ [ -1.88, -1.288, -1.928 ], [ -0.24, -0.368, -0.452 ] ]
    ]
    Outputs Statistics: {meanExponent=0.0252156764389031, negative=10, min=-0.452, max=-0.452, mean=-0.8796666666666667, count=12.0, positive=2, stdDev=1.118054808833429, zeros=0}
    Feedback for input 0
    Inputs Values: [
    	[ [ -1.76, -1.992, 0.864 ], [ -1.74, 1.548, -1.32 ] ],
    	[ [ -1.88, -1.288, -1.928 ], [ -0.24, -0.368, -0.452 ] ]
    ]
    Value Statistics: {meanExponent=0.0252156764389031, negative=10, min=-0.452, max=-0.452, mean=-0.8796666666666667, count=12.0, positive=2, stdDev=1.118054808833429, zeros=0}
    Implemented Feedback: [ [ 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 1.0, 0
```
...[skipping 2649 bytes](etc/59.txt)...
```
    99998899, 0.9999999999998899, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ] ]
    Measured Statistics: {meanExponent=-4.783064234104566E-14, negative=0, min=0.9999999999998899, max=0.9999999999998899, mean=0.3333333333332966, count=36.0, positive=12, stdDev=0.4714045207909798, zeros=24}
    Gradient Error: [ [ -1.1013412404281553E-13, -1.1013412404281553E-13, -1.1013412404281553E-13, -1.1013412404281553E-13, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, -1.1013412404281553E-13, -1.1013412404281553E-13, -1.1013412404281553E-13, -1.1013412404281553E-13, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ] ]
    Error Statistics: {meanExponent=-12.958078098036824, negative=12, min=-1.1013412404281553E-13, max=-1.1013412404281553E-13, mean=-3.671137468093851E-14, count=36.0, positive=0, stdDev=5.1917723967143496E-14, zeros=24}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 1.4685e-14 +- 3.7438e-14 [0.0000e+00 - 1.1013e-13] (180#)
    relativeTol: 5.5067e-14 +- 0.0000e+00 [5.5067e-14 - 5.5067e-14] (24#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=1.4685e-14 +- 3.7438e-14 [0.0000e+00 - 1.1013e-13] (180#), relativeTol=5.5067e-14 +- 0.0000e+00 [5.5067e-14 - 5.5067e-14] (24#)}
```



### Performance
Code from [LayerTestBase.java:192](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L192) executed in 0.01 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 0.2139 +- 0.0907 [0.1567 - 0.7580]
    Learning performance: 0.0490 +- 0.0158 [0.0370 - 0.1653]
    
```

