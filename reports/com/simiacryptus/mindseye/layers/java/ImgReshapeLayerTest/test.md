# ImgReshapeLayer
## ImgReshapeLayerTest
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
      "class": "com.simiacryptus.mindseye.layers.java.ImgReshapeLayer",
      "id": "e2a3bda5-e7e7-4c05-aeb3-4ede00001ea2",
      "isFrozen": false,
      "name": "ImgReshapeLayer/e2a3bda5-e7e7-4c05-aeb3-4ede00001ea2",
      "kernelSizeX": 2,
      "kernelSizeY": 2,
      "expand": false
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
    	[ [ -0.004, 0.272, 0.8 ], [ 0.456, -1.748, -0.54 ] ],
    	[ [ -1.66, 0.696, -0.012 ], [ -0.864, -0.516, 1.16 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ -0.004, 0.272, 0.8, 0.456, -1.748, -0.54, -1.66, 0.696, ... ] ]
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
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (240#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (240#)}
```



### Differential Validation
Code from [LayerTestBase.java:186](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L186) executed in 0.00 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ -0.004, 0.272, 0.8 ], [ 0.456, -1.748, -0.54 ] ],
    	[ [ -1.66, 0.696, -0.012 ], [ -0.864, -0.516, 1.16 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.4642384233190702, negative=7, min=1.16, max=1.16, mean=-0.16333333333333339, count=12.0, positive=5, stdDev=0.8919743394415683, zeros=0}
    Output: [
    	[ [ -0.004, 0.272, 0.8, 0.456, -1.748, -0.54, -1.66, 0.696, ... ] ]
    ]
    Outputs Statistics: {meanExponent=-0.4642384233190702, negative=7, min=1.16, max=1.16, mean=-0.16333333333333336, count=12.0, positive=5, stdDev=0.8919743394415682, zeros=0}
    Feedback for input 0
    Inputs Values: [
    	[ [ -0.004, 0.272, 0.8 ], [ 0.456, -1.748, -0.54 ] ],
    	[ [ -1.66, 0.696, -0.012 ], [ -0.864, -0.516, 1.16 ] ]
    ]
    Value Statistics: {meanExponent=-0.4642384233190702, negative=7, min=1.16, max=1.16, mean=-0.16333333333333339, count=12.0, positive=5, stdDev=0.8919743394415683, zeros=0}
    Implemented Feedback: [ [ 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 1.0,
```
...[skipping 1081 bytes](etc/63.txt)...
```
    8333333333332564, count=144.0, positive=12, stdDev=0.27638539919625776, zeros=132}
    Feedback Error: [ [ -1.6653345369377348E-15, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.1013412404281553E-13, 0.0, ... ], [ 0.0, 0.0, 0.0, -1.1013412404281553E-13, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, -1.1013412404281553E-13, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.1013412404281553E-13, ... ], [ 0.0, 0.0, 0.0, 0.0, -1.1013412404281553E-13, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], ... ]
    Error Statistics: {meanExponent=-13.215122958489298, negative=12, min=-1.1013412404281553E-13, max=-1.1013412404281553E-13, mean=-7.701401245125478E-15, count=144.0, positive=0, stdDev=2.798723678812316E-14, zeros=132}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 7.7014e-15 +- 2.7987e-14 [0.0000e+00 - 1.1013e-13] (144#)
    relativeTol: 4.6208e-14 +- 1.9813e-14 [8.3267e-16 - 5.5067e-14] (12#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=7.7014e-15 +- 2.7987e-14 [0.0000e+00 - 1.1013e-13] (144#), relativeTol=4.6208e-14 +- 1.9813e-14 [8.3267e-16 - 5.5067e-14] (12#)}
```



### Performance
Code from [LayerTestBase.java:192](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L192) executed in 0.00 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 0.1902 +- 0.0998 [0.1368 - 1.0915]
    Learning performance: 0.0031 +- 0.0053 [0.0000 - 0.0485]
    
```

