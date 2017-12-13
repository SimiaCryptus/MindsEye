# ImgBandBiasLayer
## ImgBandBiasLayerTest
### Json Serialization
Code from [StandardLayerTests.java:69](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L69) executed in 0.00 seconds: 
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
      "class": "com.simiacryptus.mindseye.layers.java.ImgBandBiasLayer",
      "id": "b2c9c838-ba3f-4566-b4df-da140c5abd15",
      "isFrozen": false,
      "name": "ImgBandBiasLayer/b2c9c838-ba3f-4566-b4df-da140c5abd15",
      "bias": [
        0.0,
        0.0,
        0.0
      ]
    }
```



### Example Input/Output Pair
Code from [StandardLayerTests.java:153](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L153) executed in 0.00 seconds: 
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
    	[ [ 0.424, 1.28, -0.032 ], [ -0.248, 0.092, -1.748 ] ],
    	[ [ 1.832, -1.48, 1.632 ], [ 1.448, 1.712, 0.144 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 0.424, 1.28, -0.032 ], [ -0.248, 0.092, -1.748 ] ],
    	[ [ 1.832, -1.48, 1.632 ], [ 1.448, 1.712, 0.144 ] ]
    ]
    --------------------
    Derivative: 
    [
    	[ [ 1.0, 1.0, 1.0 ], [ 1.0, 1.0, 1.0 ] ],
    	[ [ 1.0, 1.0, 1.0 ], [ 1.0, 1.0, 1.0 ] ]
    ]
```



### Batch Execution
Code from [StandardLayerTests.java:102](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L102) executed in 0.00 seconds: 
```java
    return getBatchingTester().test(layer, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (240#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (239#)}
```



### Differential Validation
Code from [StandardLayerTests.java:110](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L110) executed in 0.01 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ 1.472, -1.036, -1.572 ], [ 1.408, 0.476, 0.44 ] ],
    	[ [ -1.676, -0.004, -1.948 ], [ 0.584, 1.256, -1.312 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.1709465274295923, negative=6, min=-1.312, max=-1.312, mean=-0.15933333333333338, count=12.0, positive=6, stdDev=1.2265429285416616, zeros=0}
    Output: [
    	[ [ 1.472, -1.036, -1.572 ], [ 1.408, 0.476, 0.44 ] ],
    	[ [ -1.676, -0.004, -1.948 ], [ 0.584, 1.256, -1.312 ] ]
    ]
    Outputs Statistics: {meanExponent=-0.1709465274295923, negative=6, min=-1.312, max=-1.312, mean=-0.15933333333333338, count=12.0, positive=6, stdDev=1.2265429285416616, zeros=0}
    Feedback for input 0
    Inputs Values: [
    	[ [ 1.472, -1.036, -1.572 ], [ 1.408, 0.476, 0.44 ] ],
    	[ [ -1.676, -0.004, -1.948 ], [ 0.584, 1.256, -1.312 ] ]
    ]
    Value Statistics: {meanExponent=-0.1709465274295923, negative=6, min=-1.312, max=-1.312, mean=-0.15933333333333338, count=12.0, positive=6, stdDev=1.2265429285416616, zeros=0}
    Implemented Feedback: [ [ 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 
```
...[skipping 2655 bytes](etc/71.txt)...
```
    998899, 0.9999999999998899, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ] ]
    Measured Statistics: {meanExponent=-4.3905025945951446E-14, negative=0, min=0.9999999999998899, max=0.9999999999998899, mean=0.3333333333332996, count=36.0, positive=12, stdDev=0.47140452079098405, zeros=24}
    Gradient Error: [ [ -1.1013412404281553E-13, -1.1013412404281553E-13, -1.1013412404281553E-13, -1.1013412404281553E-13, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, -1.1013412404281553E-13, -1.6653345369377348E-15, -1.1013412404281553E-13, -1.1013412404281553E-13, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ] ]
    Error Statistics: {meanExponent=-13.109779799128367, negative=12, min=-1.1013412404281553E-13, max=-1.1013412404281553E-13, mean=-3.369835275021968E-14, count=36.0, positive=0, stdDev=5.0702484103554865E-14, zeros=24}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 1.3479e-14 +- 3.6067e-14 [0.0000e+00 - 1.1013e-13] (180#)
    relativeTol: 5.0548e-14 +- 1.4990e-14 [8.3267e-16 - 5.5067e-14] (24#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=1.3479e-14 +- 3.6067e-14 [0.0000e+00 - 1.1013e-13] (180#), relativeTol=5.0548e-14 +- 1.4990e-14 [8.3267e-16 - 5.5067e-14] (24#)}
```



### Performance
Code from [StandardLayerTests.java:120](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L120) executed in 0.01 seconds: 
```java
    getPerformanceTester().test(layer, permPrototype);
```
Logging: 
```
    Evaluation performance: 0.000756s +- 0.000055s [0.000700s - 0.000855s]
    Learning performance: 0.000184s +- 0.000010s [0.000168s - 0.000195s]
    
```

