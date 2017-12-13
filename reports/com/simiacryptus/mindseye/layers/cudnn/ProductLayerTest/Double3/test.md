# ProductLayer
## Double3
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
      "id": "49780997-72ba-4c8b-8dde-9c3bee208852",
      "isFrozen": false,
      "name": "ProductLayer/49780997-72ba-4c8b-8dde-9c3bee208852"
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
    	[ [ 1.032 ], [ 1.072 ] ],
    	[ [ -1.22 ], [ 0.308 ] ]
    ],
    [
    	[ [ 1.996 ], [ 0.84 ] ],
    	[ [ -0.444 ], [ -0.968 ] ]
    ],
    [
    	[ [ 1.68 ], [ 1.288 ] ],
    	[ [ -0.44 ], [ -0.692 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 3.46058496 ], [ 1.1598182400000001 ] ],
    	[ [ -0.23833920000000003 ], [ 0.20631564799999996 ] ]
    ]
    --------------------
    Derivative: 
    [
    	[ [ 3.35328 ], [ 1.08192 ] ],
    	[ [ 0.19536 ], [ 0.6698559999999999 ] ]
    ],
    [
    	[ [ 1.73376 ], [ 1.3807360000000002 ] ],
    	[ [ 0.5367999999999999 ], [ -0.213136 ] ]
    ],
    [
    	[ [ 2.059872 ], [ 0.9004800000000001 ] ],
    	[ [ 0.54168 ], [ -0.29814399999999996 ] ]
    ]
```



[GPU Log](etc/cuda.log)

### Batch Execution
Code from [StandardLayerTests.java:102](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L102) executed in 0.04 seconds: 
```java
    return getBatchingTester().test(layer, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (160#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (160#)}
```



### Differential Validation
Code from [StandardLayerTests.java:110](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L110) executed in 0.05 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ 0.392 ], [ -0.292 ] ],
    	[ [ 0.264 ], [ -1.832 ] ]
    ],
    [
    	[ [ 0.368 ], [ 0.552 ] ],
    	[ [ -0.268 ], [ 0.808 ] ]
    ],
    [
    	[ [ 0.116 ], [ 1.544 ] ],
    	[ [ 0.64 ], [ -0.86 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.3142004213323655, negative=2, min=-1.832, max=-1.832, mean=-0.367, count=4.0, positive=2, stdDev=0.8840378951153621, zeros=0},
    {meanExponent=-0.33916673719847706, negative=1, min=0.808, max=0.808, mean=0.365, count=4.0, positive=3, stdDev=0.39746572179245854, zeros=0},
    {meanExponent=-0.2515540723864773, negative=1, min=-0.86, max=-0.86, mean=0.36, count=4.0, positive=3, stdDev=0.8700850533137551, zeros=0}
    Output: [
    	[ [ 0.016733696 ], [ -0.24886809599999998 ] ],
    	[ [ -0.04528128000000001 ], [ 1.2730201600000002 ] ]
    ]
    Outputs Statistics: {meanExponent=-0.9049212309173198, negative=2, min=1.2730201600000002, max=1.2730201600000002, mean=0.24890112000000003, count=4.0, positive=2, stdDev=0.599382807086659, zeros=0}
    Feedback for input 0
    Inputs Values: [
    	[ [ 0.392 ], [ -0.292 ] ],
    	[ [ 0.264 ]
```
...[skipping 3289 bytes](etc/42.txt)...
```
    6.0, positive=1, stdDev=0.3613604753373009, zeros=12}
    Measured Feedback: [ [ 0.14425600000003008, 0.0, 0.0, 0.0 ], [ 0.0, -0.0707519999999795, 0.0, 0.0 ], [ 0.0, 0.0, -0.16118400000014743, 0.0 ], [ 0.0, 0.0, 0.0, -1.480255999999347 ] ]
    Measured Statistics: {meanExponent=-0.6533671585308, negative=3, min=-1.480255999999347, max=-1.480255999999347, mean=-0.09799599999996524, count=16.0, positive=1, stdDev=0.3613604753371477, zeros=12}
    Feedback Error: [ [ 3.008704396734174E-14, 0.0, 0.0, 0.0 ], [ 0.0, 2.0511370379949767E-14, 0.0, 0.0 ], [ 0.0, 0.0, -1.474376176702208E-13, 0.0 ], [ 0.0, 0.0, 0.0, 6.532552276894421E-13 ] ]
    Error Statistics: {meanExponent=-13.056483650933814, negative=1, min=6.532552276894421E-13, max=6.532552276894421E-13, mean=3.477600152290705E-14, count=16.0, positive=3, stdDev=1.640229148676092E-13, zeros=12}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 6.9525e-14 +- 2.0484e-13 [0.0000e+00 - 1.0578e-12] (48#)
    relativeTol: 2.5712e-13 +- 1.1731e-13 [1.0428e-13 - 4.5736e-13] (12#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=6.9525e-14 +- 2.0484e-13 [0.0000e+00 - 1.0578e-12] (48#), relativeTol=2.5712e-13 +- 1.1731e-13 [1.0428e-13 - 4.5736e-13] (12#)}
```



### Performance
Code from [StandardLayerTests.java:120](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L120) executed in 0.01 seconds: 
```java
    getPerformanceTester().test(layer, permPrototype);
```
Logging: 
```
    Evaluation performance: 0.000694s +- 0.000047s [0.000626s - 0.000755s]
    Learning performance: 0.000209s +- 0.000009s [0.000195s - 0.000221s]
    
```

