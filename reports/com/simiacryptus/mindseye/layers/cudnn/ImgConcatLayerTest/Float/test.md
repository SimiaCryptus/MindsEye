# ImgConcatLayer
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
      "class": "com.simiacryptus.mindseye.layers.cudnn.ImgConcatLayer",
      "id": "b20f743a-43e9-4779-94b1-ab5820fa4b49",
      "isFrozen": false,
      "name": "ImgConcatLayer/b20f743a-43e9-4779-94b1-ab5820fa4b49",
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
    	[ [ -0.568 ], [ 0.268 ] ],
    	[ [ 0.952 ], [ -0.848 ] ]
    ],
    [
    	[ [ -1.556 ], [ 0.108 ] ],
    	[ [ -0.632 ], [ 1.392 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ -0.568, -1.556 ], [ 0.268, 0.108 ] ],
    	[ [ 0.952, -0.632 ], [ -0.848, 1.392 ] ]
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
    	[ [ 0.092 ], [ 0.492 ] ],
    	[ [ 1.86 ], [ -0.6 ] ]
    ],
    [
    	[ [ 0.552 ], [ 0.752 ] ],
    	[ [ -1.916 ], [ 0.804 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.3241457188213811, negative=1, min=-0.6, max=-0.6, mean=0.46099999999999997, count=4.0, positive=3, stdDev=0.8972240522857153, zeros=0},
    {meanExponent=-0.04854788204704548, negative=1, min=0.804, max=0.804, mean=0.04800000000000004, count=4.0, positive=3, stdDev=1.1378119352511644, zeros=0}
    Output: [
    	[ [ 0.092, 0.552 ], [ 0.492, 0.752 ] ],
    	[ [ 1.86, -1.916 ], [ -0.6, 0.804 ] ]
    ]
    Outputs Statistics: {meanExponent=-0.18634680043421326, negative=2, min=0.804, max=0.804, mean=0.2545, count=8.0, positive=6, stdDev=1.045206080158358, zeros=0}
    Feedback for input 0
    Inputs Values: [
    	[ [ 0.092 ], [ 0.492 ] ],
    	[ [ 1.86 ], [ -0.6 ] ]
    ]
    Value Statistics: {meanExponent=-0.3241457188213811, negative=1, min=-0.6, max=-0.6, mean=0.46099999999999997, count=4.0, positive=3, stdDev=0.8972240522857153, zeros=0}
    Implemented Feedback: [ [ 1.0, 0.0, 0.0, 0.0, 0.0, 0.0
```
...[skipping 1924 bytes](etc/73.txt)...
```
    998899, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9999999999998899, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9999999999998899 ] ]
    Measured Statistics: {meanExponent=-4.7830642341045674E-14, negative=0, min=0.9999999999998899, max=0.9999999999998899, mean=0.12499999999998623, count=32.0, positive=4, stdDev=0.3307189138830374, zeros=28}
    Feedback Error: [ [ 0.0, 0.0, 0.0, 0.0, -1.1013412404281553E-13, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, -1.1013412404281553E-13, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.1013412404281553E-13, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.1013412404281553E-13 ] ]
    Error Statistics: {meanExponent=-12.958078098036825, negative=4, min=-1.1013412404281553E-13, max=-1.1013412404281553E-13, mean=-1.3766765505351941E-14, count=32.0, positive=0, stdDev=3.6423437884903677E-14, zeros=28}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 1.2493e-14 +- 3.4401e-14 [0.0000e+00 - 1.1013e-13] (64#)
    relativeTol: 4.9974e-14 +- 1.3475e-14 [1.4322e-14 - 5.5067e-14] (8#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=1.2493e-14 +- 3.4401e-14 [0.0000e+00 - 1.1013e-13] (64#), relativeTol=4.9974e-14 +- 1.3475e-14 [1.4322e-14 - 5.5067e-14] (8#)}
```



### Performance
Code from [StandardLayerTests.java:119](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L119) executed in 0.56 seconds: 
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
    	Evaluation performance: 0.012760s +- 0.000562s [0.012050s - 0.013763s]
    	Learning performance: 0.085040s +- 0.066855s [0.046310s - 0.217821s]
    
```

