# ImgConcatLayer
## Double
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
      "id": "594577c7-f41f-4763-a1a8-45eba033113c",
      "isFrozen": false,
      "name": "ImgConcatLayer/594577c7-f41f-4763-a1a8-45eba033113c",
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
    	[ [ -0.18 ], [ 1.4 ] ],
    	[ [ 0.392 ], [ 1.492 ] ]
    ],
    [
    	[ [ 0.452 ], [ -0.408 ] ],
    	[ [ 1.008 ], [ 1.452 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ -0.18, 0.452 ], [ 1.4, -0.408 ] ],
    	[ [ 0.392, 1.008 ], [ 1.492, 1.452 ] ]
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
    	[ [ -0.104 ], [ -0.808 ] ],
    	[ [ -1.616 ], [ 1.788 ] ]
    ],
    [
    	[ [ 1.256 ], [ 0.82 ] ],
    	[ [ -1.216 ], [ 0.54 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.1536866072570418, negative=3, min=1.788, max=1.788, mean=-0.1850000000000001, count=4.0, positive=1, stdDev=1.2584891735728203, zeros=0},
    {meanExponent=-0.04246729336385534, negative=1, min=0.54, max=0.54, mean=0.35, count=4.0, positive=3, stdDev=0.9394402588775935, zeros=0}
    Output: [
    	[ [ -0.104, 1.256 ], [ -0.808, 0.82 ] ],
    	[ [ -1.616, -1.216 ], [ 1.788, 0.54 ] ]
    ]
    Outputs Statistics: {meanExponent=-0.09807695031044858, negative=4, min=0.54, max=0.54, mean=0.08249999999999995, count=8.0, positive=4, stdDev=1.1422467990762768, zeros=0}
    Feedback for input 0
    Inputs Values: [
    	[ [ -0.104 ], [ -0.808 ] ],
    	[ [ -1.616 ], [ 1.788 ] ]
    ]
    Value Statistics: {meanExponent=-0.1536866072570418, negative=3, min=1.788, max=1.788, mean=-0.1850000000000001, count=4.0, positive=1, stdDev=1.2584891735728203, zeros=0}
    Implemented Feedback: [ [ 1.0, 0.0, 0.0, 0.0
```
...[skipping 1915 bytes](etc/72.txt)...
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
    	Evaluation performance: 0.013216s +- 0.000535s [0.012505s - 0.013838s]
    	Learning performance: 0.049050s +- 0.003435s [0.045499s - 0.053732s]
    
```

