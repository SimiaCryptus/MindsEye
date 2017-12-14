# ImgConcatLayer
## BandLimitTest
### Json Serialization
Code from [StandardLayerTests.java:68](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L68) executed in 0.00 seconds: 
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
      "class": "com.simiacryptus.mindseye.layers.cudnn.ImgConcatLayer",
      "id": "2d2ed9a5-33b5-4d68-b296-399228bfb334",
      "isFrozen": false,
      "name": "ImgConcatLayer/2d2ed9a5-33b5-4d68-b296-399228bfb334",
      "maxBands": 3
    }
```



### Example Input/Output Pair
Code from [StandardLayerTests.java:152](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L152) executed in 0.00 seconds: 
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
    	[ [ 1.784, 1.62 ], [ 0.42, -1.308 ] ],
    	[ [ -0.636, -1.588 ], [ 1.656, 0.78 ] ]
    ],
    [
    	[ [ 0.536, -1.12 ], [ 0.128, 1.916 ] ],
    	[ [ 0.032, -0.196 ], [ 1.388, -0.328 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 1.784, 1.62, 0.536 ], [ 0.42, -1.308, 0.128 ] ],
    	[ [ -0.636, -1.588, 0.032 ], [ 1.656, 0.78, 1.388 ] ]
    ]
    --------------------
    Derivative: 
    [
    	[ [ 1.0, 1.0 ], [ 1.0, 1.0 ] ],
    	[ [ 1.0, 1.0 ], [ 1.0, 1.0 ] ]
    ],
    [
    	[ [ 1.0, 0.0 ], [ 1.0, 0.0 ] ],
    	[ [ 1.0, 0.0 ], [ 1.0, 0.0 ] ]
    ]
```



[GPU Log](etc/cuda.log)

### Batch Execution
Code from [StandardLayerTests.java:101](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L101) executed in 0.01 seconds: 
```java
    return getBatchingTester().test(layer, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (280#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (240#)}
```



### Differential Validation
Code from [StandardLayerTests.java:109](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L109) executed in 0.03 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ -1.156, 0.472 ], [ 0.484, -0.124 ] ],
    	[ [ -1.224, -1.352 ], [ 0.504, 1.86 ] ]
    ],
    [
    	[ [ -1.36, -1.02 ], [ -0.468, 0.12 ] ],
    	[ [ 0.196, -0.288 ], [ 0.904, -1.324 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.1617664412995192, negative=4, min=1.86, max=1.86, mean=-0.06700000000000003, count=8.0, positive=4, stdDev=1.0495556202507803, zeros=0},
    {meanExponent=-0.28484110575642063, negative=5, min=-1.324, max=-1.324, mean=-0.405, count=8.0, positive=3, stdDev=0.7502312976675927, zeros=0}
    Output: [
    	[ [ -1.156, 0.472, -1.36 ], [ 0.484, -0.124, -0.468 ] ],
    	[ [ -1.224, -1.352, 0.196 ], [ 0.504, 1.86, 0.904 ] ]
    ]
    Outputs Statistics: {meanExponent=-0.18682685559333104, negative=6, min=0.904, max=0.904, mean=-0.10533333333333339, count=12.0, positive=6, stdDev=0.9848672781423675, zeros=0}
    Feedback for input 0
    Inputs Values: [
    	[ [ -1.156, 0.472 ], [ 0.484, -0.124 ] ],
    	[ [ -1.224, -1.352 ], [ 0.504, 1.86 ] ]
    ]
    Value Statistics: {meanExponent=-0.1617664412995192, negative=4, min=1.86, max=1.86, mean=-
```
...[skipping 3331 bytes](etc/71.txt)...
```
    0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ] ]
    Measured Statistics: {meanExponent=-4.7830642341045674E-14, negative=0, min=0.0, max=0.0, mean=0.04166666666666208, count=96.0, positive=4, stdDev=0.1998263134713413, zeros=92}
    Feedback Error: [ [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ] ]
    Error Statistics: {meanExponent=-12.958078098036825, negative=4, min=0.0, max=0.0, mean=-4.588921835117314E-15, count=96.0, positive=0, stdDev=2.2007695994873667E-14, zeros=92}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 6.4590e-15 +- 2.5641e-14 [0.0000e+00 - 1.1013e-13] (192#)
    relativeTol: 5.1672e-14 +- 1.1261e-14 [1.4322e-14 - 5.5067e-14] (12#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=6.4590e-15 +- 2.5641e-14 [0.0000e+00 - 1.1013e-13] (192#), relativeTol=5.1672e-14 +- 1.1261e-14 [1.4322e-14 - 5.5067e-14] (12#)}
```



### Performance
Code from [StandardLayerTests.java:119](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L119) executed in 0.39 seconds: 
```java
    getPerformanceTester().test(layer, permPrototype);
```
Logging: 
```
    100 batches
    Input Dimensions:
    	[100, 100, 1]
    	[100, 100, 1]
    Performance:
    	Evaluation performance: 0.013509s +- 0.001946s [0.011035s - 0.016117s]
    	Learning performance: 0.048993s +- 0.003436s [0.045756s - 0.055073s]
    
```

