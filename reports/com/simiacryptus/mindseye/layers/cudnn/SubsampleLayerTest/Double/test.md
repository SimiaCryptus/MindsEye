# SubsampleLayer
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
      "class": "com.simiacryptus.mindseye.layers.cudnn.SubsampleLayer",
      "id": "d3989644-f5ec-4344-8ab6-8e81a5eba6c7",
      "isFrozen": false,
      "name": "SubsampleLayer/d3989644-f5ec-4344-8ab6-8e81a5eba6c7",
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
    	[ [ 1.996 ], [ 0.04 ] ],
    	[ [ 0.724 ], [ 0.096 ] ]
    ],
    [
    	[ [ 1.352 ], [ -1.584 ] ],
    	[ [ 0.136 ], [ 1.132 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 1.996, 1.352 ], [ 0.04, -1.584 ] ],
    	[ [ 0.724, 0.136 ], [ 0.096, 1.132 ] ]
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
    	[ [ 0.524 ], [ 1.392 ] ],
    	[ [ -0.376 ], [ -0.94 ] ]
    ],
    [
    	[ [ -0.9 ], [ 1.532 ] ],
    	[ [ -1.168 ], [ -0.096 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.14717844480359257, negative=2, min=-0.94, max=-0.94, mean=0.15000000000000002, count=4.0, positive=2, stdDev=0.8870197292056136, zeros=0},
    {meanExponent=-0.20269616236203522, negative=3, min=-0.096, max=-0.096, mean=-0.158, count=4.0, positive=1, stdDev=1.05245047389414, zeros=0}
    Output: [
    	[ [ 0.524, -0.9 ], [ 1.392, 1.532 ] ],
    	[ [ -0.376, -1.168 ], [ -0.94, -0.096 ] ]
    ]
    Outputs Statistics: {meanExponent=-0.1749373035828139, negative=5, min=-0.096, max=-0.096, mean=-0.003999999999999993, count=8.0, positive=3, stdDev=0.9853649070268334, zeros=0}
    Feedback for input 0
    Inputs Values: [
    	[ [ 0.524 ], [ 1.392 ] ],
    	[ [ -0.376 ], [ -0.94 ] ]
    ]
    Value Statistics: {meanExponent=-0.14717844480359257, negative=2, min=-0.94, max=-0.94, mean=0.15000000000000002, count=4.0, positive=2, stdDev=0.8870197292056136, zeros=0}
    Implemented Feedback: [ [ 1.0, 0.0,
```
...[skipping 1926 bytes](etc/91.txt)...
```
    9999999998899, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9999999999998899, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0000000000000286 ] ]
    Measured Statistics: {meanExponent=-3.276302567614995E-14, negative=0, min=1.0000000000000286, max=1.0000000000000286, mean=0.12499999999999056, count=32.0, positive=4, stdDev=0.33071891388304886, zeros=28}
    Feedback Error: [ [ 0.0, 0.0, 0.0, 0.0, -1.1013412404281553E-13, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, -1.1013412404281553E-13, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.1013412404281553E-13, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.864375403532904E-14 ] ]
    Error Statistics: {meanExponent=-13.104301089584563, negative=3, min=2.864375403532904E-14, max=2.864375403532904E-14, mean=-9.429956815409923E-15, count=32.0, positive=1, stdDev=3.2769779210413227E-14, zeros=28}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 1.2493e-14 +- 3.4401e-14 [0.0000e+00 - 1.1013e-13] (64#)
    relativeTol: 4.9974e-14 +- 1.3475e-14 [1.4322e-14 - 5.5067e-14] (8#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=1.2493e-14 +- 3.4401e-14 [0.0000e+00 - 1.1013e-13] (64#), relativeTol=4.9974e-14 +- 1.3475e-14 [1.4322e-14 - 5.5067e-14] (8#)}
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
    	Evaluation performance: 0.000503s +- 0.000213s [0.000250s - 0.000848s]
    	Learning performance: 0.000275s +- 0.000137s [0.000190s - 0.000549s]
    
```

