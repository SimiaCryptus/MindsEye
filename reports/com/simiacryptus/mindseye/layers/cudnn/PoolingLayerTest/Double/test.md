# PoolingLayer
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
      "class": "com.simiacryptus.mindseye.layers.cudnn.PoolingLayer",
      "id": "b2df03d5-28f1-4dab-8318-30279f3e7c05",
      "isFrozen": false,
      "name": "PoolingLayer/b2df03d5-28f1-4dab-8318-30279f3e7c05",
      "mode": 0,
      "windowX": 2,
      "windowY": 2,
      "paddingX": 0,
      "paddingY": 0,
      "strideX": 2,
      "strideY": 2
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
    	[ [ 0.544, -1.364 ], [ 1.248, 1.804 ], [ 0.132, 1.172 ], [ 1.908, -1.2 ] ],
    	[ [ -0.104, 1.056 ], [ 1.476, -0.464 ], [ -1.58, -0.036 ], [ 0.772, -1.012 ] ],
    	[ [ 1.48, -0.092 ], [ -1.24, -0.332 ], [ 1.336, -1.46 ], [ -0.032, -0.224 ] ],
    	[ [ -0.452, 1.964 ], [ -0.28, 1.136 ], [ 1.944, 1.696 ], [ -0.304, 0.432 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 1.476, 1.804 ], [ 1.908, 1.172 ] ],
    	[ [ 1.48, 1.964 ], [ 1.944, 1.696 ] ]
    ]
    --------------------
    Derivative: 
    [
    	[ [ 0.0, 0.0 ], [ 0.0, 1.0 ], [ 0.0, 1.0 ], [ 1.0, 0.0 ] ],
    	[ [ 0.0, 0.0 ], [ 1.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ] ],
    	[ [ 1.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ] ],
    	[ [ 0.0, 1.0 ], [ 0.0, 0.0 ], [ 1.0, 1.0 ], [ 0.0, 0.0 ] ]
    ]
```



[GPU Log](etc/cuda.log)

### Batch Execution
Code from [StandardLayerTests.java:102](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L102) executed in 0.01 seconds: 
```java
    return getBatchingTester().test(layer, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (400#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (160#)}
```



### Differential Validation
Code from [StandardLayerTests.java:110](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L110) executed in 0.02 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ 0.592, -0.7 ], [ -1.292, -1.996 ], [ -0.14, -1.236 ], [ 1.872, 1.276 ] ],
    	[ [ 1.448, 1.96 ], [ -1.304, 1.3 ], [ -0.028, 0.036 ], [ 0.724, -1.416 ] ],
    	[ [ 1.224, -0.972 ], [ -1.908, -0.84 ], [ 0.304, -0.328 ], [ 0.58, -1.156 ] ],
    	[ [ 0.56, -0.684 ], [ 1.536, 1.096 ], [ -0.292, -1.144 ], [ 1.556, 0.836 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.1282949065672334, negative=16, min=0.836, max=0.836, mean=0.045750000000000055, count=32.0, positive=16, stdDev=1.1522317204017605, zeros=0}
    Output: [
    	[ [ 1.448, 1.96 ], [ 1.872, 1.276 ] ],
    	[ [ 1.536, 1.096 ], [ 1.556, 0.836 ] ]
    ]
    Outputs Statistics: {meanExponent=0.14644984899267052, negative=0, min=0.836, max=0.836, mean=1.4475, count=8.0, positive=8, stdDev=0.35166710110557736, zeros=0}
    Feedback for input 0
    Inputs Values: [
    	[ [ 0.592, -0.7 ], [ -1.292, -1.996 ], [ -0.14, -1.236 ], [ 1.872, 1.276 ] ],
    	[ [ 1.448, 1.96 ], [ -1.304, 1.3 ], [ -0.028, 0.036 ], [ 0.724, -1.416 ] ],
    	[ [ 1.224, -0.972 ], [ -1.908, -0.84 ], [ 0.304, -0.328 ], [ 0.58, 
```
...[skipping 1222 bytes](etc/39.txt)...
```
    d Statistics: {meanExponent=-4.7830642341045674E-14, negative=0, min=0.9999999999998899, max=0.9999999999998899, mean=0.03124999999999656, count=256.0, positive=8, stdDev=0.17399263633841902, zeros=248}
    Feedback Error: [ [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ -1.1013412404281553E-13, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, -1.1013412404281553E-13, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], ... ]
    Error Statistics: {meanExponent=-12.958078098036824, negative=8, min=-1.1013412404281553E-13, max=-1.1013412404281553E-13, mean=-3.4416913763379853E-15, count=256.0, positive=0, stdDev=1.9162526593034043E-14, zeros=248}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 3.4417e-15 +- 1.9163e-14 [0.0000e+00 - 1.1013e-13] (256#)
    relativeTol: 5.5067e-14 +- 0.0000e+00 [5.5067e-14 - 5.5067e-14] (8#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=3.4417e-15 +- 1.9163e-14 [0.0000e+00 - 1.1013e-13] (256#), relativeTol=5.5067e-14 +- 0.0000e+00 [5.5067e-14 - 5.5067e-14] (8#)}
```



### Performance
Code from [StandardLayerTests.java:120](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L120) executed in 0.00 seconds: 
```java
    getPerformanceTester().test(layer, permPrototype);
```
Logging: 
```
    Evaluation performance: 0.000229s +- 0.000040s [0.000199s - 0.000301s]
    Learning performance: 0.000283s +- 0.000008s [0.000275s - 0.000296s]
    
```

