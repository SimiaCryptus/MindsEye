# ImgConcatLayer
## BandLimitTest
### Json Serialization
Code from [StandardLayerTests.java:69](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L69) executed in 0.00 seconds: 
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
      "id": "969a4ef1-1523-490c-ab17-2a0e80593074",
      "isFrozen": false,
      "name": "ImgConcatLayer/969a4ef1-1523-490c-ab17-2a0e80593074",
      "maxBands": 3
    }
```



### Example Input/Output Pair
Code from [StandardLayerTests.java:153](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L153) executed in 0.00 seconds: 
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
    	[ [ 0.284, -1.228 ], [ -1.708, 1.948 ] ],
    	[ [ -0.704, 1.552 ], [ 1.088, -1.076 ] ]
    ],
    [
    	[ [ 0.176, 1.488 ], [ -0.456, 0.34 ] ],
    	[ [ -0.704, -0.628 ], [ -1.876, -0.104 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 0.284, -1.228, 0.176 ], [ -1.708, 1.948, -0.456 ] ],
    	[ [ -0.704, 1.552, -0.704 ], [ 1.088, -1.076, -1.876 ] ]
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
Code from [StandardLayerTests.java:102](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L102) executed in 0.01 seconds: 
```java
    return getBatchingTester().test(layer, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (280#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (240#)}
```



### Differential Validation
Code from [StandardLayerTests.java:110](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L110) executed in 0.03 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ 1.66, -0.732 ], [ -1.3, 0.5 ] ],
    	[ [ 0.936, -0.132 ], [ 0.184, 1.36 ] ]
    ],
    [
    	[ [ -1.648, 1.724 ], [ -0.648, 1.188 ] ],
    	[ [ -1.788, 1.272 ], [ 1.5, 1.9 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.20153262036687353, negative=3, min=1.36, max=1.36, mean=0.3095, count=8.0, positive=5, stdDev=0.951451391296476, zeros=0},
    {meanExponent=0.1439481751432954, negative=3, min=1.9, max=1.9, mean=0.4375, count=8.0, positive=5, stdDev=1.443217152752835, zeros=0}
    Output: [
    	[ [ 1.66, -0.732, -1.648 ], [ -1.3, 0.5, -0.648 ] ],
    	[ [ 0.936, -0.132, -1.788 ], [ 0.184, 1.36, 1.5 ] ]
    ]
    Outputs Statistics: {meanExponent=-0.09627249801564315, negative=6, min=1.5, max=1.5, mean=-0.009000000000000008, count=12.0, positive=6, stdDev=1.1758039802620164, zeros=0}
    Feedback for input 0
    Inputs Values: [
    	[ [ 1.66, -0.732 ], [ -1.3, 0.5 ] ],
    	[ [ 0.936, -0.132 ], [ 0.184, 1.36 ] ]
    ]
    Value Statistics: {meanExponent=-0.20153262036687353, negative=3, min=1.36, max=1.36, mean=0.3095, count=8.0, positive=5, stdDev=0.9514513
```
...[skipping 3261 bytes](etc/34.txt)...
```
    0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ] ]
    Measured Statistics: {meanExponent=-4.7830642341045674E-14, negative=0, min=0.0, max=0.0, mean=0.04166666666666208, count=96.0, positive=4, stdDev=0.1998263134713413, zeros=92}
    Feedback Error: [ [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ] ]
    Error Statistics: {meanExponent=-12.958078098036825, negative=4, min=0.0, max=0.0, mean=-4.588921835117314E-15, count=96.0, positive=0, stdDev=2.2007695994873667E-14, zeros=92}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 6.8834e-15 +- 2.6659e-14 [0.0000e+00 - 1.1013e-13] (192#)
    relativeTol: 5.5067e-14 +- 0.0000e+00 [5.5067e-14 - 5.5067e-14] (12#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=6.8834e-15 +- 2.6659e-14 [0.0000e+00 - 1.1013e-13] (192#), relativeTol=5.5067e-14 +- 0.0000e+00 [5.5067e-14 - 5.5067e-14] (12#)}
```



### Performance
Code from [StandardLayerTests.java:120](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L120) executed in 0.01 seconds: 
```java
    getPerformanceTester().test(layer, permPrototype);
```
Logging: 
```
    Evaluation performance: 0.000330s +- 0.000066s [0.000243s - 0.000402s]
    Learning performance: 0.000232s +- 0.000038s [0.000200s - 0.000305s]
    
```

