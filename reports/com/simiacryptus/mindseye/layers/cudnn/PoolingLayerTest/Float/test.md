# PoolingLayer
## Float
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
      "id": "903882c5-ac2f-4da1-adb0-0e7a803c8533",
      "isFrozen": false,
      "name": "PoolingLayer/903882c5-ac2f-4da1-adb0-0e7a803c8533",
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
    	[ [ 0.388, 0.96 ], [ 1.348, 0.396 ], [ -1.008, 0.144 ], [ -0.74, -0.192 ] ],
    	[ [ 1.56, -0.696 ], [ 0.884, 0.196 ], [ 0.608, -1.908 ], [ -1.884, -1.72 ] ],
    	[ [ -0.608, -1.416 ], [ -0.016, -0.704 ], [ 0.292, -0.96 ], [ -0.452, -1.076 ] ],
    	[ [ 1.804, 0.288 ], [ 0.46, -0.448 ], [ -1.428, 1.192 ], [ 1.36, -0.476 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 0.0, 0.0 ], [ 0.0, 0.0 ] ],
    	[ [ 0.0, 0.0 ], [ 0.0, 0.0 ] ]
    ]
    --------------------
    Derivative: 
    [
    	[ [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ] ],
    	[ [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ] ],
    	[ [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ] ],
    	[ [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ] ]
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
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (400#), relativeTol=0.0000e+00 +- 0.0000e+00 [Infinity - -Infinity] (0#)}
```



### Differential Validation
Code from [StandardLayerTests.java:110](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L110) executed in 0.06 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ -0.78, 0.184 ], [ -1.996, -1.42 ], [ 0.2, 0.696 ], [ -1.576, -1.064 ] ],
    	[ [ 1.184, 0.12 ], [ 0.9, 0.532 ], [ 1.788, 0.248 ], [ 1.948, -0.344 ] ],
    	[ [ 1.748, 1.2 ], [ -0.964, 1.44 ], [ -0.184, -0.828 ], [ 0.572, -1.132 ] ],
    	[ [ -1.78, -1.184 ], [ 0.564, -0.188 ], [ -0.42, -0.868 ], [ -0.416, 0.892 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.14931747692829586, negative=16, min=0.892, max=0.892, mean=-0.02899999999999998, count=32.0, positive=16, stdDev=1.0765323961683642, zeros=0}
    Output: [
    	[ [ 0.0, 0.0 ], [ 0.0, 0.0 ] ],
    	[ [ 0.0, 0.0 ], [ 0.0, 0.0 ] ]
    ]
    Outputs Statistics: {meanExponent=NaN, negative=0, min=0.0, max=0.0, mean=0.0, count=8.0, positive=0, stdDev=0.0, zeros=8}
    Feedback for input 0
    Inputs Values: [
    	[ [ -0.78, 0.184 ], [ -1.996, -1.42 ], [ 0.2, 0.696 ], [ -1.576, -1.064 ] ],
    	[ [ 1.184, 0.12 ], [ 0.9, 0.532 ], [ 1.788, 0.248 ], [ 1.948, -0.344 ] ],
    	[ [ 1.748, 1.2 ], [ -0.964, 1.44 ], [ -0.184, -0.828 ], [ 0.572, -1.132 ] ],
    	[ [ -1.78, -1.184 ], [ 0.564, -0.188 ], [ -0.42,
```
...[skipping 894 bytes](etc/40.txt)...
```
    0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], ... ]
    Measured Statistics: {meanExponent=NaN, negative=0, min=0.0, max=0.0, mean=0.0, count=256.0, positive=0, stdDev=0.0, zeros=256}
    Feedback Error: [ [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], ... ]
    Error Statistics: {meanExponent=NaN, negative=0, min=0.0, max=0.0, mean=0.0, count=256.0, positive=0, stdDev=0.0, zeros=256}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (256#)
    relativeTol: 0.0000e+00 +- 0.0000e+00 [Infinity - -Infinity] (0#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (256#), relativeTol=0.0000e+00 +- 0.0000e+00 [Infinity - -Infinity] (0#)}
```



### Performance
Code from [StandardLayerTests.java:120](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L120) executed in 0.01 seconds: 
```java
    getPerformanceTester().test(layer, permPrototype);
```
Logging: 
```
    Evaluation performance: 0.000214s +- 0.000019s [0.000189s - 0.000249s]
    Learning performance: 0.001452s +- 0.001125s [0.000409s - 0.003337s]
    
```

