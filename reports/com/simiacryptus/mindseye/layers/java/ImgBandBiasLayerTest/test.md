# ImgBandBiasLayer
## ImgBandBiasLayerTest
### Json Serialization
Code from [StandardLayerTests.java:68](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L68) executed in 0.00 seconds: 
```java
    JsonObject json = layer.getJson();
    NNLayer echo = NNLayer.fromJson(json);
    if ((echo == null)) throw new AssertionError("Failed to deserialize");
    if ((layer == echo)) throw new AssertionError("Serialization did not copy");
    if ((!layer.equals(echo))) throw new AssertionError("Serialization not equal");
    return new GsonBuilder().setPrettyPrinting().create().toJson(json);
```

Returns: 

```
    {
      "class": "com.simiacryptus.mindseye.layers.java.ImgBandBiasLayer",
      "id": "43fdbca8-fbdd-4296-bb55-a74fe5d7205c",
      "isFrozen": false,
      "name": "ImgBandBiasLayer/43fdbca8-fbdd-4296-bb55-a74fe5d7205c",
      "bias": [
        0.0,
        0.0,
        0.0
      ]
    }
```



### Example Input/Output Pair
Code from [StandardLayerTests.java:152](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L152) executed in 0.00 seconds: 
```java
    SimpleEval eval = SimpleEval.run(layer, inputPrototype);
    return String.format("--------------------\nInput: \n[%s]\n--------------------\nOutput: \n%s\n--------------------\nDerivative: \n%s",
      Arrays.stream(inputPrototype).map(t -> t.prettyPrint()).reduce((a, b) -> a + ",\n" + b).get(),
      eval.getOutput().prettyPrint(),
      Arrays.stream(eval.getDerivative()).map(t -> t.prettyPrint()).reduce((a, b) -> a + ",\n" + b).get());
```

Returns: 

```
    --------------------
    Input: 
    [[
    	[ [ 0.872, -0.932, 0.996 ], [ -1.276, -0.952, -0.632 ] ],
    	[ [ 0.524, 0.116, -0.472 ], [ 0.992, 1.644, 0.604 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 0.872, -0.932, 0.996 ], [ -1.276, -0.952, -0.632 ] ],
    	[ [ 0.524, 0.116, -0.472 ], [ 0.992, 1.644, 0.604 ] ]
    ]
    --------------------
    Derivative: 
    [
    	[ [ 1.0, 1.0, 1.0 ], [ 1.0, 1.0, 1.0 ] ],
    	[ [ 1.0, 1.0, 1.0 ], [ 1.0, 1.0, 1.0 ] ]
    ]
```



### Batch Execution
Code from [StandardLayerTests.java:101](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L101) executed in 0.00 seconds: 
```java
    return getBatchingTester().test(layer, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (240#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (239#)}
```



### Differential Validation
Code from [StandardLayerTests.java:109](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L109) executed in 0.00 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ -1.212, 1.444, 0.792 ], [ 0.26, 1.968, -1.636 ] ],
    	[ [ 1.724, 0.032, -0.476 ], [ 1.272, 1.484, -0.832 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.11000706107249615, negative=4, min=-0.832, max=-0.832, mean=0.4016666666666666, count=12.0, positive=8, stdDev=1.1752542514518105, zeros=0}
    Output: [
    	[ [ -1.212, 1.444, 0.792 ], [ 0.26, 1.968, -1.636 ] ],
    	[ [ 1.724, 0.032, -0.476 ], [ 1.272, 1.484, -0.832 ] ]
    ]
    Outputs Statistics: {meanExponent=-0.11000706107249615, negative=4, min=-0.832, max=-0.832, mean=0.4016666666666666, count=12.0, positive=8, stdDev=1.1752542514518105, zeros=0}
    Feedback for input 0
    Inputs Values: [
    	[ [ -1.212, 1.444, 0.792 ], [ 0.26, 1.968, -1.636 ] ],
    	[ [ 1.724, 0.032, -0.476 ], [ 1.272, 1.484, -0.832 ] ]
    ]
    Value Statistics: {meanExponent=-0.11000706107249615, negative=4, min=-0.832, max=-0.832, mean=0.4016666666666666, count=12.0, positive=8, stdDev=1.1752542514518105, zeros=0}
    Implemented Feedback: [ [ 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 1.0, 0.0,
```
...[skipping 2639 bytes](etc/113.txt)...
```
    99999998899, 0.9999999999998899, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ] ]
    Measured Statistics: {meanExponent=-4.28081034527471E-14, negative=0, min=0.9999999999998899, max=0.9999999999998899, mean=0.33333333333330045, count=36.0, positive=12, stdDev=0.4714045207909852, zeros=24}
    Gradient Error: [ [ -1.1013412404281553E-13, -1.1013412404281553E-13, -1.1013412404281553E-13, -1.1013412404281553E-13, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, -1.1013412404281553E-13, 2.864375403532904E-14, -1.1013412404281553E-13, -1.1013412404281553E-13, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ] ]
    Error Statistics: {meanExponent=-13.006819095219404, negative=11, min=-1.1013412404281553E-13, max=-1.1013412404281553E-13, mean=-3.285643362321227E-14, count=36.0, positive=1, stdDev=5.147319000513805E-14, zeros=24}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 1.3779e-14 +- 3.6080e-14 [0.0000e+00 - 1.1013e-13] (180#)
    relativeTol: 5.1672e-14 +- 1.1261e-14 [1.4322e-14 - 5.5067e-14] (24#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=1.3779e-14 +- 3.6080e-14 [0.0000e+00 - 1.1013e-13] (180#), relativeTol=5.1672e-14 +- 1.1261e-14 [1.4322e-14 - 5.5067e-14] (24#)}
```



### Performance
Code from [StandardLayerTests.java:119](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L119) executed in 0.00 seconds: 
```java
    getPerformanceTester().test(layer, permPrototype);
```
Logging: 
```
    100 batches
    Input Dimensions:
    	[2, 2, 3]
    Performance:
    	Evaluation performance: 0.000354s +- 0.000011s [0.000338s - 0.000368s]
    	Learning performance: 0.000194s +- 0.000040s [0.000159s - 0.000269s]
    
```

