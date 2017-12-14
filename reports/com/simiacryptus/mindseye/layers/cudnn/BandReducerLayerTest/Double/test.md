# BandReducerLayer
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
      "class": "com.simiacryptus.mindseye.layers.cudnn.BandReducerLayer",
      "id": "f89945f4-cd30-4b75-abf5-0a8e15ab6780",
      "isFrozen": false,
      "name": "BandReducerLayer/f89945f4-cd30-4b75-abf5-0a8e15ab6780",
      "mode": 0
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
    	[ [ 0.96, 1.36 ], [ -0.132, 0.312 ], [ 0.816, -0.464 ] ],
    	[ [ -1.404, -0.392 ], [ -1.34, -1.824 ], [ 0.488, 0.948 ] ],
    	[ [ -1.3, -0.612 ], [ 1.976, -0.856 ], [ 0.876, -0.504 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 1.976, 1.36 ] ]
    ]
    --------------------
    Derivative: 
    [
    	[ [ 0.0, 1.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ] ],
    	[ [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ] ],
    	[ [ 0.0, 0.0 ], [ 1.0, 0.0 ], [ 0.0, 0.0 ] ]
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
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (200#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (40#)}
```



### Differential Validation
Code from [StandardLayerTests.java:109](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L109) executed in 0.01 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ -0.32, 1.736 ], [ -1.088, 1.928 ], [ 1.888, -1.016 ] ],
    	[ [ -0.468, -1.412 ], [ -0.82, 0.876 ], [ -1.296, 1.688 ] ],
    	[ [ -0.832, -0.004 ], [ 0.16, 1.844 ], [ -0.576, 0.74 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.16736523920490945, negative=10, min=0.74, max=0.74, mean=0.16822222222222225, count=18.0, positive=8, stdDev=1.1851746967128494, zeros=0}
    Output: [
    	[ [ 1.888, 1.928 ] ]
    ]
    Outputs Statistics: {meanExponent=0.2805545097644311, negative=0, min=1.928, max=1.928, mean=1.908, count=2.0, positive=2, stdDev=0.019999999999998897, zeros=0}
    Feedback for input 0
    Inputs Values: [
    	[ [ -0.32, 1.736 ], [ -1.088, 1.928 ], [ 1.888, -1.016 ] ],
    	[ [ -0.468, -1.412 ], [ -0.82, 0.876 ], [ -1.296, 1.688 ] ],
    	[ [ -0.832, -0.004 ], [ 0.16, 1.844 ], [ -0.576, 0.74 ] ]
    ]
    Value Statistics: {meanExponent=-0.16736523920490945, negative=10, min=0.74, max=0.74, mean=0.16822222222222225, count=18.0, positive=8, stdDev=1.1851746967128494, zeros=0}
    Implemented Feedback: [ [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 
```
...[skipping 71 bytes](etc/44.txt)...
```
    .0 ], ... ]
    Implemented Statistics: {meanExponent=0.0, negative=0, min=0.0, max=0.0, mean=0.05555555555555555, count=36.0, positive=2, stdDev=0.2290614236454256, zeros=34}
    Measured Feedback: [ [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.9999999999998899, 0.0 ], [ 0.0, 0.0 ], ... ]
    Measured Statistics: {meanExponent=-4.7830642341045674E-14, negative=0, min=0.0, max=0.0, mean=0.05555555555554944, count=36.0, positive=2, stdDev=0.22906142364540036, zeros=34}
    Feedback Error: [ [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ -1.1013412404281553E-13, 0.0 ], [ 0.0, 0.0 ], ... ]
    Error Statistics: {meanExponent=-12.958078098036825, negative=2, min=0.0, max=0.0, mean=-6.118562446823085E-15, count=36.0, positive=0, stdDev=2.5227479245189218E-14, zeros=34}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 6.1186e-15 +- 2.5227e-14 [0.0000e+00 - 1.1013e-13] (36#)
    relativeTol: 5.5067e-14 +- 0.0000e+00 [5.5067e-14 - 5.5067e-14] (2#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=6.1186e-15 +- 2.5227e-14 [0.0000e+00 - 1.1013e-13] (36#), relativeTol=5.5067e-14 +- 0.0000e+00 [5.5067e-14 - 5.5067e-14] (2#)}
```



### Performance
Code from [StandardLayerTests.java:119](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L119) executed in 0.18 seconds: 
```java
    getPerformanceTester().test(layer, permPrototype);
```
Logging: 
```
    100 batches
    Input Dimensions:
    	[100, 100, 3]
    Performance:
    	Evaluation performance: 0.019115s +- 0.001188s [0.017437s - 0.020837s]
    	Learning performance: 0.000534s +- 0.000046s [0.000477s - 0.000579s]
    
```

