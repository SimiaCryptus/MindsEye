# ImgBandScaleLayer
## ImgBandScaleLayerTest
### Json Serialization
Code from [StandardLayerTests.java:68](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L68) executed in 0.00 seconds: 
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
      "class": "com.simiacryptus.mindseye.layers.java.ImgBandScaleLayer",
      "id": "b23bc951-a0a3-4b28-92bd-da39daeb75ff",
      "isFrozen": false,
      "name": "ImgBandScaleLayer/b23bc951-a0a3-4b28-92bd-da39daeb75ff",
      "bias": [
        0.0,
        0.0,
        0.0
      ]
    }
```



### Example Input/Output Pair
Code from [StandardLayerTests.java:152](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L152) executed in 0.00 seconds: 
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
    	[ [ 0.884, -1.956, 0.984 ], [ 0.012, 0.124, -0.8 ] ],
    	[ [ -0.872, 0.216, -0.248 ], [ -0.148, 0.012, -0.356 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 0.0, -0.0, 0.0 ], [ 0.0, 0.0, -0.0 ] ],
    	[ [ -0.0, 0.0, -0.0 ], [ -0.0, 0.0, -0.0 ] ]
    ]
    --------------------
    Derivative: 
    [
    	[ [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ] ],
    	[ [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ] ]
    ]
```



### Batch Execution
Code from [StandardLayerTests.java:101](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L101) executed in 0.00 seconds: 
```java
    return getBatchingTester().test(layer, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (240#), relativeTol=0.0000e+00 +- 0.0000e+00 [Infinity - -Infinity] (0#)}
```



### Differential Validation
Code from [StandardLayerTests.java:109](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L109) executed in 0.01 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ 1.992, -1.448, 1.348 ], [ 1.376, 0.172, 0.968 ] ],
    	[ [ -1.516, 0.756, -1.344 ], [ 1.224, -0.68, -0.472 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.02236478158929513, negative=5, min=-0.472, max=-0.472, mean=0.19799999999999993, count=12.0, positive=7, stdDev=1.1948115611537524, zeros=0}
    Output: [
    	[ [ 0.0, -0.0, 0.0 ], [ 0.0, 0.0, 0.0 ] ],
    	[ [ -0.0, 0.0, -0.0 ], [ 0.0, -0.0, -0.0 ] ]
    ]
    Outputs Statistics: {meanExponent=NaN, negative=0, min=-0.0, max=-0.0, mean=0.0, count=12.0, positive=0, stdDev=0.0, zeros=12}
    Feedback for input 0
    Inputs Values: [
    	[ [ 1.992, -1.448, 1.348 ], [ 1.376, 0.172, 0.968 ] ],
    	[ [ -1.516, 0.756, -1.344 ], [ 1.224, -0.68, -0.472 ] ]
    ]
    Value Statistics: {meanExponent=-0.02236478158929513, negative=5, min=-0.472, max=-0.472, mean=0.19799999999999993, count=12.0, positive=7, stdDev=1.1948115611537524, zeros=0}
    Implemented Feedback: [ [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
```
...[skipping 1965 bytes](etc/114.txt)...
```
    00002, -1.516, 1.376, 1.224, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, -1.448, 0.7560000000000001, 0.172, -0.6800000000000002, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ] ]
    Measured Statistics: {meanExponent=-0.022364781589295103, negative=5, min=-0.47200000000000003, max=-0.47200000000000003, mean=0.06600000000000002, count=36.0, positive=7, stdDev=0.6961107830095885, zeros=24}
    Gradient Error: [ [ 2.220446049250313E-16, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 1.1102230246251565E-16, 0.0, -1.1102230246251565E-16, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ] ]
    Error Statistics: {meanExponent=-15.904418104247007, negative=2, min=-5.551115123125783E-17, max=-5.551115123125783E-17, mean=1.3877787807814457E-17, count=36.0, positive=4, stdDev=6.049187461416408E-17, zeros=30}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 4.6259e-18 +- 2.7367e-17 [0.0000e+00 - 2.2204e-16] (180#)
    relativeTol: 3.4109e-17 +- 3.5037e-17 [0.0000e+00 - 8.2361e-17] (12#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=4.6259e-18 +- 2.7367e-17 [0.0000e+00 - 2.2204e-16] (180#), relativeTol=3.4109e-17 +- 3.5037e-17 [0.0000e+00 - 8.2361e-17] (12#)}
```



### Performance
Code from [StandardLayerTests.java:119](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L119) executed in 0.01 seconds: 
```java
    getPerformanceTester().test(layer, permPrototype);
```
Logging: 
```
    100 batches
    Input Dimensions:
    	[2, 2, 3]
    Performance:
    	Evaluation performance: 0.000411s +- 0.000066s [0.000345s - 0.000522s]
    	Learning performance: 0.000255s +- 0.000029s [0.000210s - 0.000302s]
    
```

