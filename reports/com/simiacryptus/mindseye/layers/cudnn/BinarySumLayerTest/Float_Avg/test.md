# BinarySumLayer
## Float_Avg
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
      "class": "com.simiacryptus.mindseye.layers.cudnn.BinarySumLayer",
      "id": "373093e5-2fad-4d52-851e-2ab486116950",
      "isFrozen": false,
      "name": "BinarySumLayer/373093e5-2fad-4d52-851e-2ab486116950",
      "rightFactor": 0.5,
      "leftFactor": 0.5
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
    	[ [ -0.28 ], [ -1.92 ] ],
    	[ [ 0.028 ], [ -1.568 ] ]
    ],
    [
    	[ [ -1.044 ], [ -1.884 ] ],
    	[ [ -1.78 ], [ 1.312 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 0.0 ], [ 0.0 ] ],
    	[ [ 0.0 ], [ 0.0 ] ]
    ]
    --------------------
    Derivative: 
    [
    	[ [ 0.0 ], [ 0.0 ] ],
    	[ [ 0.0 ], [ 0.0 ] ]
    ],
    [
    	[ [ 0.0 ], [ 0.0 ] ],
    	[ [ 0.0 ], [ 0.0 ] ]
    ]
```



[GPU Log](etc/cuda.log)

### Batch Execution
Code from [StandardLayerTests.java:101](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L101) executed in 0.02 seconds: 
```java
    return getBatchingTester().test(layer, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (120#), relativeTol=0.0000e+00 +- 0.0000e+00 [Infinity - -Infinity] (0#)}
```



### Differential Validation
Code from [StandardLayerTests.java:109](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L109) executed in 0.03 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ -0.62 ], [ -1.104 ] ],
    	[ [ -1.008 ], [ 0.284 ] ]
    ],
    [
    	[ [ -1.248 ], [ -0.28 ] ],
    	[ [ -0.78 ], [ -0.168 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.17696509123800547, negative=3, min=0.284, max=0.284, mean=-0.6120000000000001, count=4.0, positive=1, stdDev=0.5481240735453972, zeros=0},
    {meanExponent=-0.3348058747237581, negative=4, min=-0.168, max=-0.168, mean=-0.619, count=4.0, positive=0, stdDev=0.43008255021565334, zeros=0}
    Output: [
    	[ [ 0.0 ], [ 0.0 ] ],
    	[ [ 0.0 ], [ 0.0 ] ]
    ]
    Outputs Statistics: {meanExponent=NaN, negative=0, min=0.0, max=0.0, mean=0.0, count=4.0, positive=0, stdDev=0.0, zeros=4}
    Feedback for input 0
    Inputs Values: [
    	[ [ -0.62 ], [ -1.104 ] ],
    	[ [ -1.008 ], [ 0.284 ] ]
    ]
    Value Statistics: {meanExponent=-0.17696509123800547, negative=3, min=0.284, max=0.284, mean=-0.6120000000000001, count=4.0, positive=1, stdDev=0.5481240735453972, zeros=0}
    Implemented Feedback: [ [ 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0 ] ]
    I
```
...[skipping 767 bytes](etc/48.txt)...
```
    gative=4, min=-0.168, max=-0.168, mean=-0.619, count=4.0, positive=0, stdDev=0.43008255021565334, zeros=0}
    Implemented Feedback: [ [ 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0 ] ]
    Implemented Statistics: {meanExponent=NaN, negative=0, min=0.0, max=0.0, mean=0.0, count=16.0, positive=0, stdDev=0.0, zeros=16}
    Measured Feedback: [ [ 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0 ] ]
    Measured Statistics: {meanExponent=NaN, negative=0, min=0.0, max=0.0, mean=0.0, count=16.0, positive=0, stdDev=0.0, zeros=16}
    Feedback Error: [ [ 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0 ] ]
    Error Statistics: {meanExponent=NaN, negative=0, min=0.0, max=0.0, mean=0.0, count=16.0, positive=0, stdDev=0.0, zeros=16}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (32#)
    relativeTol: 0.0000e+00 +- 0.0000e+00 [Infinity - -Infinity] (0#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (32#), relativeTol=0.0000e+00 +- 0.0000e+00 [Infinity - -Infinity] (0#)}
```



### Performance
Code from [StandardLayerTests.java:119](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L119) executed in 0.27 seconds: 
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
    	Evaluation performance: 0.014407s +- 0.000952s [0.013447s - 0.016173s]
    	Learning performance: 0.026253s +- 0.001365s [0.024049s - 0.027565s]
    
```

