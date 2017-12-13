# ProductLayer
## Double
### Json Serialization
Code from [StandardLayerTests.java:69](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L69) executed in 0.00 seconds: 
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
      "class": "com.simiacryptus.mindseye.layers.cudnn.ProductLayer",
      "id": "133aaaf5-b467-4369-84b7-537dd9b22501",
      "isFrozen": false,
      "name": "ProductLayer/133aaaf5-b467-4369-84b7-537dd9b22501"
    }
```



### Example Input/Output Pair
Code from [StandardLayerTests.java:153](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L153) executed in 0.00 seconds: 
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
    	[ [ -0.848 ], [ 1.172 ] ],
    	[ [ 0.808 ], [ -1.848 ] ]
    ],
    [
    	[ [ 1.62 ], [ 1.568 ] ],
    	[ [ 1.032 ], [ -1.416 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ -1.37376 ], [ 1.837696 ] ],
    	[ [ 0.833856 ], [ 2.616768 ] ]
    ]
    --------------------
    Derivative: 
    [
    	[ [ 1.62 ], [ 1.568 ] ],
    	[ [ 1.032 ], [ -1.416 ] ]
    ],
    [
    	[ [ -0.848 ], [ 1.172 ] ],
    	[ [ 0.808 ], [ -1.848 ] ]
    ]
```



[GPU Log](etc/cuda.log)

### Batch Execution
Code from [StandardLayerTests.java:102](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L102) executed in 0.02 seconds: 
```java
    return getBatchingTester().test(layer, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (120#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (120#)}
```



### Differential Validation
Code from [StandardLayerTests.java:110](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L110) executed in 0.02 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ 0.24 ], [ -0.648 ] ],
    	[ [ 0.884 ], [ 0.48 ] ]
    ],
    [
    	[ [ -0.488 ], [ -0.684 ] ],
    	[ [ -1.104 ], [ 0.348 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.2951300625072851, negative=1, min=0.48, max=0.48, mean=0.23900000000000002, count=4.0, positive=3, stdDev=0.5614436748241091, zeros=0},
    {meanExponent=-0.222993939734353, negative=3, min=0.348, max=0.348, mean=-0.4820000000000001, count=4.0, positive=1, stdDev=0.5283521552903896, zeros=0}
    Output: [
    	[ [ -0.11711999999999999 ], [ 0.44323200000000007 ] ],
    	[ [ -0.9759360000000001 ], [ 0.16704 ] ]
    ]
    Outputs Statistics: {meanExponent=-0.518124002241638, negative=2, min=0.16704, max=0.16704, mean=-0.120696, count=4.0, positive=2, stdDev=0.532037367168886, zeros=0}
    Feedback for input 0
    Inputs Values: [
    	[ [ 0.24 ], [ -0.648 ] ],
    	[ [ 0.884 ], [ 0.48 ] ]
    ]
    Value Statistics: {meanExponent=-0.2951300625072851, negative=1, min=0.48, max=0.48, mean=0.23900000000000002, count=4.0, positive=3, stdDev=0.5614436748241091, zeros=0}
    Implemented Feedback: [ [ -0.
```
...[skipping 1609 bytes](etc/41.txt)...
```
    sitive=3, stdDev=0.2991904702693587, zeros=12}
    Measured Feedback: [ [ 0.23999999999996247, 0.0, 0.0, 0.0 ], [ 0.0, 0.8839999999998849, 0.0, 0.0 ], [ 0.0, 0.0, -0.648000000000315, 0.0 ], [ 0.0, 0.0, 0.0, 0.47999999999992493 ] ]
    Measured Statistics: {meanExponent=-0.2951300625072804, negative=1, min=0.47999999999992493, max=0.47999999999992493, mean=0.05974999999996608, count=16.0, positive=3, stdDev=0.2991904702693774, zeros=12}
    Feedback Error: [ [ -3.752553823233029E-14, 0.0, 0.0, 0.0 ], [ 0.0, -1.1513012765362873E-13, 0.0, 0.0 ], [ 0.0, 0.0, -3.149702720861569E-13, 0.0 ], [ 0.0, 0.0, 0.0, -7.505107646466058E-14 ] ]
    Error Statistics: {meanExponent=-12.997714398089958, negative=4, min=-7.505107646466058E-14, max=-7.505107646466058E-14, mean=-3.391731340229853E-14, count=16.0, positive=0, stdDev=7.948895454512184E-14, zeros=12}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 4.7240e-14 +- 1.0863e-13 [0.0000e+00 - 4.5031e-13] (32#)
    relativeTol: 1.5471e-13 +- 1.1124e-13 [1.6608e-14 - 3.7725e-13] (8#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=4.7240e-14 +- 1.0863e-13 [0.0000e+00 - 4.5031e-13] (32#), relativeTol=1.5471e-13 +- 1.1124e-13 [1.6608e-14 - 3.7725e-13] (8#)}
```



### Performance
Code from [StandardLayerTests.java:120](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L120) executed in 0.01 seconds: 
```java
    getPerformanceTester().test(layer, permPrototype);
```
Logging: 
```
    Evaluation performance: 0.000434s +- 0.000102s [0.000319s - 0.000611s]
    Learning performance: 0.000248s +- 0.000052s [0.000176s - 0.000331s]
    
```

