# BinarySumLayer
## Double_Subtract
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
      "id": "60725891-7696-46d9-adfe-55db4172e708",
      "isFrozen": false,
      "name": "BinarySumLayer/60725891-7696-46d9-adfe-55db4172e708",
      "rightFactor": -1.0,
      "leftFactor": 1.0
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
    	[ [ 0.812 ], [ 1.176 ] ],
    	[ [ -1.424 ], [ -1.928 ] ]
    ],
    [
    	[ [ -1.908 ], [ -0.176 ] ],
    	[ [ -0.036 ], [ 1.532 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 2.7199999999999998 ], [ 1.3519999999999999 ] ],
    	[ [ -1.388 ], [ -3.46 ] ]
    ]
    --------------------
    Derivative: 
    [
    	[ [ 1.0 ], [ 1.0 ] ],
    	[ [ 1.0 ], [ 1.0 ] ]
    ],
    [
    	[ [ -1.0 ], [ -1.0 ] ],
    	[ [ -1.0 ], [ -1.0 ] ]
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
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (120#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (120#)}
```



### Differential Validation
Code from [StandardLayerTests.java:109](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L109) executed in 0.02 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ -0.708 ], [ 0.976 ] ],
    	[ [ -1.04 ], [ -1.0 ] ]
    ],
    [
    	[ [ -1.376 ], [ 1.576 ] ],
    	[ [ -0.5 ], [ 0.944 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.03587089633618971, negative=3, min=-1.0, max=-1.0, mean=-0.443, count=4.0, positive=1, stdDev=0.8292231304058033, zeros=0},
    {meanExponent=0.0025291614217791856, negative=2, min=0.944, max=0.944, mean=0.16100000000000003, count=4.0, positive=2, stdDev=1.1634650832749558, zeros=0}
    Output: [
    	[ [ 0.6679999999999999 ], [ -0.6000000000000001 ] ],
    	[ [ -0.54 ], [ -1.944 ] ]
    ]
    Outputs Statistics: {meanExponent=-0.0939955666818966, negative=3, min=-1.944, max=-1.944, mean=-0.6040000000000001, count=4.0, positive=1, stdDev=0.9243505828418134, zeros=0}
    Feedback for input 0
    Inputs Values: [
    	[ [ -0.708 ], [ 0.976 ] ],
    	[ [ -1.04 ], [ -1.0 ] ]
    ]
    Value Statistics: {meanExponent=-0.03587089633618971, negative=3, min=-1.0, max=-1.0, mean=-0.443, count=4.0, positive=1, stdDev=0.8292231304058033, zeros=0}
    Implemented Feedback: [ [ 1.0, 0.0, 0.0, 0.0 ], [ 0.0, 1.0
```
...[skipping 1516 bytes](etc/46.txt)...
```
    0, stdDev=0.4330127018922193, zeros=12}
    Measured Feedback: [ [ -0.9999999999998899, 0.0, 0.0, 0.0 ], [ 0.0, -0.9999999999998899, 0.0, 0.0 ], [ 0.0, 0.0, -0.9999999999998899, 0.0 ], [ 0.0, 0.0, 0.0, -0.9999999999998899 ] ]
    Measured Statistics: {meanExponent=-4.7830642341045674E-14, negative=4, min=-0.9999999999998899, max=-0.9999999999998899, mean=-0.24999999999997247, count=16.0, positive=0, stdDev=0.4330127018921716, zeros=12}
    Feedback Error: [ [ 1.1013412404281553E-13, 0.0, 0.0, 0.0 ], [ 0.0, 1.1013412404281553E-13, 0.0, 0.0 ], [ 0.0, 0.0, 1.1013412404281553E-13, 0.0 ], [ 0.0, 0.0, 0.0, 1.1013412404281553E-13 ] ]
    Error Statistics: {meanExponent=-12.958078098036825, negative=0, min=1.1013412404281553E-13, max=1.1013412404281553E-13, mean=2.7533531010703882E-14, count=16.0, positive=4, stdDev=4.7689474622312385E-14, zeros=12}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 2.7534e-14 +- 4.7689e-14 [0.0000e+00 - 1.1013e-13] (32#)
    relativeTol: 5.5067e-14 +- 0.0000e+00 [5.5067e-14 - 5.5067e-14] (8#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=2.7534e-14 +- 4.7689e-14 [0.0000e+00 - 1.1013e-13] (32#), relativeTol=5.5067e-14 +- 0.0000e+00 [5.5067e-14 - 5.5067e-14] (8#)}
```



### Performance
Code from [StandardLayerTests.java:119](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L119) executed in 0.25 seconds: 
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
    	Evaluation performance: 0.012946s +- 0.001824s [0.010708s - 0.015049s]
    	Learning performance: 0.024447s +- 0.000926s [0.023432s - 0.025868s]
    
```

