# SubsampleLayer
## Float
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
      "class": "com.simiacryptus.mindseye.layers.cudnn.SubsampleLayer",
      "id": "657866d0-e37c-4abb-8174-574f027f71a1",
      "isFrozen": false,
      "name": "SubsampleLayer/657866d0-e37c-4abb-8174-574f027f71a1",
      "maxBands": -1
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
    	[ [ 1.764 ], [ -1.368 ] ],
    	[ [ -1.128 ], [ 0.644 ] ]
    ],
    [
    	[ [ 0.388 ], [ 1.364 ] ],
    	[ [ -0.524 ], [ -1.484 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 1.764, 0.388 ], [ -1.368, 1.364 ] ],
    	[ [ -1.128, -0.524 ], [ 0.644, -1.484 ] ]
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
Code from [StandardLayerTests.java:101](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L101) executed in 0.01 seconds: 
```java
    return getBatchingTester().test(layer, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (160#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (160#)}
```



### Differential Validation
Code from [StandardLayerTests.java:109](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L109) executed in 0.02 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ -0.652 ], [ 1.168 ] ],
    	[ [ 0.748 ], [ 0.336 ] ]
    ],
    [
    	[ [ -0.688 ], [ 1.404 ] ],
    	[ [ -0.84 ], [ -0.932 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.1795171715593484, negative=1, min=0.336, max=0.336, mean=0.39999999999999997, count=4.0, positive=3, stdDev=0.6748570218942677, zeros=0},
    {meanExponent=-0.030337313888709812, negative=3, min=-0.932, max=-0.932, mean=-0.264, count=4.0, positive=1, stdDev=0.9669539802906858, zeros=0}
    Output: [
    	[ [ -0.652, -0.688 ], [ 1.168, 1.404 ] ],
    	[ [ 0.748, -0.84 ], [ 0.336, -0.932 ] ]
    ]
    Outputs Statistics: {meanExponent=-0.1049272427240291, negative=4, min=-0.932, max=-0.932, mean=0.06799999999999999, count=8.0, positive=4, stdDev=0.8974630911630851, zeros=0}
    Feedback for input 0
    Inputs Values: [
    	[ [ -0.652 ], [ 1.168 ] ],
    	[ [ 0.748 ], [ 0.336 ] ]
    ]
    Value Statistics: {meanExponent=-0.1795171715593484, negative=1, min=0.336, max=0.336, mean=0.39999999999999997, count=4.0, positive=3, stdDev=0.6748570218942677, zeros=0}
    Implemented Feedback: [ [ 1.0, 0.0
```
...[skipping 1938 bytes](etc/92.txt)...
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
Code from [StandardLayerTests.java:119](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L119) executed in 0.01 seconds: 
```java
    getPerformanceTester().test(layer, permPrototype);
```
Logging: 
```
    100 batches
    Input Dimensions:
    	[2, 2, 1]
    	[2, 2, 1]
    Performance:
    	Evaluation performance: 0.000295s +- 0.000062s [0.000248s - 0.000416s]
    	Learning performance: 0.000227s +- 0.000016s [0.000202s - 0.000244s]
    
```

