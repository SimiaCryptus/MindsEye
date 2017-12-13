# ImgBandScaleLayer
## ImgBandScaleLayerTest
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
      "class": "com.simiacryptus.mindseye.layers.java.ImgBandScaleLayer",
      "id": "628d84b6-cfca-4abe-9d5e-23804213f70a",
      "isFrozen": false,
      "name": "ImgBandScaleLayer/628d84b6-cfca-4abe-9d5e-23804213f70a",
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
    	[ [ 1.852, 0.008, -0.276 ], [ -1.944, 1.088, -0.66 ] ],
    	[ [ -0.648, -0.416, -1.044 ], [ 1.596, 1.852, 0.252 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 0.0, 0.0, -0.0 ], [ -0.0, 0.0, -0.0 ] ],
    	[ [ -0.0, -0.0, -0.0 ], [ 0.0, 0.0, 0.0 ] ]
    ]
    --------------------
    Derivative: 
    [
    	[ [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ] ],
    	[ [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ] ]
    ]
```



### Batch Execution
Code from [StandardLayerTests.java:102](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L102) executed in 0.00 seconds: 
```java
    return getBatchingTester().test(layer, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (240#), relativeTol=0.0000e+00 +- 0.0000e+00 [Infinity - -Infinity] (0#)}
```



### Differential Validation
Code from [StandardLayerTests.java:110](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L110) executed in 0.00 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ -0.96, -0.504, -1.168 ], [ 0.248, 0.372, -0.64 ] ],
    	[ [ 0.992, -0.64, 0.136 ], [ 1.928, -1.188, 1.4 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.16953322772066803, negative=6, min=1.4, max=1.4, mean=-0.0020000000000000018, count=12.0, positive=6, stdDev=0.9853073970424999, zeros=0}
    Output: [
    	[ [ -0.0, -0.0, -0.0 ], [ 0.0, 0.0, -0.0 ] ],
    	[ [ 0.0, -0.0, 0.0 ], [ 0.0, -0.0, 0.0 ] ]
    ]
    Outputs Statistics: {meanExponent=NaN, negative=0, min=0.0, max=0.0, mean=0.0, count=12.0, positive=0, stdDev=0.0, zeros=12}
    Feedback for input 0
    Inputs Values: [
    	[ [ -0.96, -0.504, -1.168 ], [ 0.248, 0.372, -0.64 ] ],
    	[ [ 0.992, -0.64, 0.136 ], [ 1.928, -1.188, 1.4 ] ]
    ]
    Value Statistics: {meanExponent=-0.16953322772066803, negative=6, min=1.4, max=1.4, mean=-0.0020000000000000018, count=12.0, positive=6, stdDev=0.9853073970424999, zeros=0}
    Implemented Feedback: [ [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ],
```
...[skipping 1878 bytes](etc/72.txt)...
```
    ev=0.568868272195867, zeros=24}
    Measured Gradient: [ [ -0.9600000000000001, 0.992, 0.248, 1.928, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, -0.504, -0.6400000000000001, 0.37200000000000005, -1.188, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ] ]
    Measured Statistics: {meanExponent=-0.16953322772066803, negative=6, min=1.4, max=1.4, mean=-6.666666666666795E-4, count=36.0, positive=6, stdDev=0.568868272195867, zeros=24}
    Gradient Error: [ [ -1.1102230246251565E-16, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, -1.1102230246251565E-16, 5.551115123125783E-17, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ] ]
    Error Statistics: {meanExponent=-16.029847269106998, negative=3, min=0.0, max=0.0, mean=-7.709882115452476E-18, count=36.0, positive=1, stdDev=3.245484927078791E-17, zeros=32}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 2.1588e-18 +- 1.4761e-17 [0.0000e+00 - 1.1102e-16] (180#)
    relativeTol: 2.5492e-17 +- 3.6695e-17 [0.0000e+00 - 8.6736e-17] (12#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=2.1588e-18 +- 1.4761e-17 [0.0000e+00 - 1.1102e-16] (180#), relativeTol=2.5492e-17 +- 3.6695e-17 [0.0000e+00 - 8.6736e-17] (12#)}
```



### Performance
Code from [StandardLayerTests.java:120](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L120) executed in 0.00 seconds: 
```java
    getPerformanceTester().test(layer, permPrototype);
```
Logging: 
```
    Evaluation performance: 0.000360s +- 0.000049s [0.000308s - 0.000447s]
    Learning performance: 0.000241s +- 0.000017s [0.000225s - 0.000272s]
    
```

