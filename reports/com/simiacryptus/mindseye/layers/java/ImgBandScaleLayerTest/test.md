# ImgBandScaleLayer
## ImgBandScaleLayerTest
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
      "class": "com.simiacryptus.mindseye.layers.java.ImgBandScaleLayer",
      "id": "e2a3bda5-e7e7-4c05-aeb3-4ede00001e79",
      "isFrozen": false,
      "name": "ImgBandScaleLayer/e2a3bda5-e7e7-4c05-aeb3-4ede00001e79",
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
    	[ [ 0.308, 1.676, 0.688 ], [ -1.86, 0.004, -0.108 ] ],
    	[ [ 1.104, -1.068, 0.66 ], [ 1.476, -1.588, 1.188 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 0.0, 0.0, 0.0 ], [ -0.0, 0.0, -0.0 ] ],
    	[ [ 0.0, -0.0, 0.0 ], [ 0.0, -0.0, 0.0 ] ]
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
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (240#), relativeTol=0.0000e+00 +- 0.0000e+00 [Infinity - -Infinity] (0#)}
```



### Differential Validation
Code from [LayerTestBase.java:186](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L186) executed in 0.01 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ 0.308, 1.676, 0.688 ], [ -1.86, 0.004, -0.108 ] ],
    	[ [ 1.104, -1.068, 0.66 ], [ 1.476, -1.588, 1.188 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.2673960485071747, negative=4, min=1.188, max=1.188, mean=0.20666666666666664, count=12.0, positive=8, stdDev=1.12704017477442, zeros=0}
    Output: [
    	[ [ 0.0, 0.0, 0.0 ], [ -0.0, 0.0, -0.0 ] ],
    	[ [ 0.0, -0.0, 0.0 ], [ 0.0, -0.0, 0.0 ] ]
    ]
    Outputs Statistics: {meanExponent=NaN, negative=0, min=0.0, max=0.0, mean=0.0, count=12.0, positive=0, stdDev=0.0, zeros=12}
    Feedback for input 0
    Inputs Values: [
    	[ [ 0.308, 1.676, 0.688 ], [ -1.86, 0.004, -0.108 ] ],
    	[ [ 1.104, -1.068, 0.66 ], [ 1.476, -1.588, 1.188 ] ]
    ]
    Value Statistics: {meanExponent=-0.2673960485071747, negative=4, min=1.188, max=1.188, mean=0.20666666666666664, count=12.0, positive=8, stdDev=1.12704017477442, zeros=0}
    Implemented Feedback: [ [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0
```
...[skipping 1904 bytes](etc/60.txt)...
```
    24}
    Measured Gradient: [ [ 0.30800000000000005, 1.104, -1.8600000000000003, 1.4760000000000002, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 1.6760000000000002, -1.068, 0.004, -1.588, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ] ]
    Measured Statistics: {meanExponent=-0.2673960485071747, negative=4, min=1.188, max=1.188, mean=0.06888888888888887, count=36.0, positive=8, stdDev=0.6579497522936003, zeros=24}
    Gradient Error: [ [ 5.551115123125783E-17, 0.0, -2.220446049250313E-16, 2.220446049250313E-16, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 2.220446049250313E-16, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ] ]
    Error Statistics: {meanExponent=-15.804074772359012, negative=1, min=0.0, max=0.0, mean=7.709882115452476E-18, count=36.0, positive=3, stdDev=6.430245059256404E-17, zeros=32}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 4.0091e-18 +- 2.8684e-17 [0.0000e+00 - 2.2204e-16] (180#)
    relativeTol: 2.4272e-17 +- 3.4952e-17 [0.0000e+00 - 9.0116e-17] (12#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=4.0091e-18 +- 2.8684e-17 [0.0000e+00 - 2.2204e-16] (180#), relativeTol=2.4272e-17 +- 3.4952e-17 [0.0000e+00 - 9.0116e-17] (12#)}
```



### Performance
Code from [LayerTestBase.java:192](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L192) executed in 0.01 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 0.2111 +- 0.0706 [0.1482 - 0.5700]
    Learning performance: 0.0508 +- 0.0216 [0.0370 - 0.1938]
    
```

