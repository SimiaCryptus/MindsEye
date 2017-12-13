# ImgConcatLayer
## Double
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
      "id": "f96b9627-02fe-4834-a389-da87b521bcf5",
      "isFrozen": false,
      "name": "ImgConcatLayer/f96b9627-02fe-4834-a389-da87b521bcf5",
      "maxBands": -1
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
    	[ [ -0.04 ], [ -0.516 ] ],
    	[ [ -1.356 ], [ 1.248 ] ]
    ],
    [
    	[ [ -1.5 ], [ -1.1 ] ],
    	[ [ 0.816 ], [ 0.5 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ -0.04, -1.5 ], [ -0.516, -1.1 ] ],
    	[ [ -1.356, 0.816 ], [ 1.248, 0.5 ] ]
    ]
    --------------------
    Derivative: 
    [
    	[ [ 1.0 ], [ 1.0 ] ],
    	[ [ 1.0 ], [ 1.0 ] ]
    ],
    [
    	[ [ 1.0 ], [ 1.0 ] ],
    	[ [ 1.0 ], [ 1.0 ] ]
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
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (160#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (160#)}
```



### Differential Validation
Code from [StandardLayerTests.java:110](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L110) executed in 0.02 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ -0.512 ], [ -1.344 ] ],
    	[ [ 1.736 ], [ -1.144 ] ]
    ],
    [
    	[ [ -1.776 ], [ -0.352 ] ],
    	[ [ 0.912 ], [ -1.816 ] ]
    ]
    Inputs Statistics: {meanExponent=0.03391124374777893, negative=3, min=-1.144, max=-1.144, mean=-0.316, count=4.0, positive=1, stdDev=1.2238758106932255, zeros=0},
    {meanExponent=0.0037740768585489415, negative=3, min=-1.816, max=-1.816, mean=-0.758, count=4.0, positive=1, stdDev=1.1302017519009604, zeros=0}
    Output: [
    	[ [ -0.512, -1.776 ], [ -1.344, -0.352 ] ],
    	[ [ 1.736, 0.912 ], [ -1.144, -1.816 ] ]
    ]
    Outputs Statistics: {meanExponent=0.018842660303163936, negative=6, min=-1.816, max=-1.816, mean=-0.537, count=8.0, positive=2, stdDev=1.1985220064729725, zeros=0}
    Feedback for input 0
    Inputs Values: [
    	[ [ -0.512 ], [ -1.344 ] ],
    	[ [ 1.736 ], [ -1.144 ] ]
    ]
    Value Statistics: {meanExponent=0.03391124374777893, negative=3, min=-1.144, max=-1.144, mean=-0.316, count=4.0, positive=1, stdDev=1.2238758106932255, zeros=0}
    Implemented Feedback: [ [ 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
```
...[skipping 1913 bytes](etc/35.txt)...
```
    998899, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9999999999998899, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9999999999998899 ] ]
    Measured Statistics: {meanExponent=-4.7830642341045674E-14, negative=0, min=0.9999999999998899, max=0.9999999999998899, mean=0.12499999999998623, count=32.0, positive=4, stdDev=0.3307189138830374, zeros=28}
    Feedback Error: [ [ 0.0, 0.0, 0.0, 0.0, -1.1013412404281553E-13, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, -1.1013412404281553E-13, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.1013412404281553E-13, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.1013412404281553E-13 ] ]
    Error Statistics: {meanExponent=-12.958078098036825, negative=4, min=-1.1013412404281553E-13, max=-1.1013412404281553E-13, mean=-1.3766765505351941E-14, count=32.0, positive=0, stdDev=3.6423437884903677E-14, zeros=28}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 1.3767e-14 +- 3.6423e-14 [0.0000e+00 - 1.1013e-13] (64#)
    relativeTol: 5.5067e-14 +- 0.0000e+00 [5.5067e-14 - 5.5067e-14] (8#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=1.3767e-14 +- 3.6423e-14 [0.0000e+00 - 1.1013e-13] (64#), relativeTol=5.5067e-14 +- 0.0000e+00 [5.5067e-14 - 5.5067e-14] (8#)}
```



### Performance
Code from [StandardLayerTests.java:120](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L120) executed in 0.01 seconds: 
```java
    getPerformanceTester().test(layer, permPrototype);
```
Logging: 
```
    Evaluation performance: 0.000296s +- 0.000040s [0.000260s - 0.000373s]
    Learning performance: 0.000218s +- 0.000017s [0.000202s - 0.000250s]
    
```

