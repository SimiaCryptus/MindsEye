# ImgReshapeLayer
## ImgReshapeLayerTest
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
      "class": "com.simiacryptus.mindseye.layers.java.ImgReshapeLayer",
      "id": "a864e734-2f23-44db-97c1-504000002bf9",
      "isFrozen": false,
      "name": "ImgReshapeLayer/a864e734-2f23-44db-97c1-504000002bf9",
      "kernelSizeX": 2,
      "kernelSizeY": 2,
      "expand": false
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
    	[ [ -1.956, -0.472, 1.324 ], [ -0.004, 1.356, 0.788 ] ],
    	[ [ 1.464, -0.892, 1.152 ], [ -0.952, 0.832, -1.264 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ -1.956, -0.472, 1.324, -0.004, 1.356, 0.788, 1.464, -0.892, ... ] ]
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
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (240#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (240#)}
```



### Differential Validation
Code from [LayerTestBase.java:144](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.00 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ -1.956, -0.472, 1.324 ], [ -0.004, 1.356, 0.788 ] ],
    	[ [ 1.464, -0.892, 1.152 ], [ -0.952, 0.832, -1.264 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.17534079238790268, negative=6, min=-1.264, max=-1.264, mean=0.11466666666666671, count=12.0, positive=6, stdDev=1.1389285413151353, zeros=0}
    Output: [
    	[ [ -1.956, -0.472, 1.324, -0.004, 1.356, 0.788, 1.464, -0.892, ... ] ]
    ]
    Outputs Statistics: {meanExponent=-0.17534079238790265, negative=6, min=-1.264, max=-1.264, mean=0.11466666666666664, count=12.0, positive=6, stdDev=1.1389285413151355, zeros=0}
    Feedback for input 0
    Inputs Values: [
    	[ [ -1.956, -0.472, 1.324 ], [ -0.004, 1.356, 0.788 ] ],
    	[ [ 1.464, -0.892, 1.152 ], [ -0.952, 0.832, -1.264 ] ]
    ]
    Value Statistics: {meanExponent=-0.17534079238790268, negative=6, min=-1.264, max=-1.264, mean=0.11466666666666671, count=12.0, positive=6, stdDev=1.1389285413151353, zeros=0}
    Implemented Feedback: [ [ 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], ... ]
    Implemented Statistics: {meanExponent=0.0, negative=0, min=1.0, max=1.0, mean=0.08333333333333333, count=144.0, positive=12, stdDev=0.2763853991962833, zeros=132}
    Measured Feedback: [ [ 0.9999999999998899, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9999999999998899, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.9999999999999983, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.9999999999998899, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9999999999998899, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.9999999999998899, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], ... ]
    Measured Statistics: {meanExponent=-4.3905025945951446E-14, negative=0, min=0.9999999999998899, max=0.9999999999998899, mean=0.0833333333333249, count=144.0, positive=12, stdDev=0.2763853991962554, zeros=132}
    Feedback Error: [ [ -1.1013412404281553E-13, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.1013412404281553E-13, 0.0, ... ], [ 0.0, 0.0, 0.0, -1.6653345369377348E-15, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, -1.1013412404281553E-13, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.1013412404281553E-13, ... ], [ 0.0, 0.0, 0.0, 0.0, -1.1013412404281553E-13, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], ... ]
    Error Statistics: {meanExponent=-13.109779799128367, negative=12, min=-1.1013412404281553E-13, max=-1.1013412404281553E-13, mean=-8.42458818755492E-15, count=144.0, positive=0, stdDev=2.925075267334976E-14, zeros=132}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 8.4246e-15 +- 2.9251e-14 [0.0000e+00 - 1.1013e-13] (144#)
    relativeTol: 5.0548e-14 +- 1.4990e-14 [8.3267e-16 - 5.5067e-14] (12#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=8.4246e-15 +- 2.9251e-14 [0.0000e+00 - 1.1013e-13] (144#), relativeTol=5.0548e-14 +- 1.4990e-14 [8.3267e-16 - 5.5067e-14] (12#)}
```



### Performance
Code from [LayerTestBase.java:149](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L149) executed in 0.00 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 0.2073 +- 0.0554 [0.1681 - 0.5529]
    Learning performance: 0.0034 +- 0.0096 [0.0000 - 0.0883]
    
```

