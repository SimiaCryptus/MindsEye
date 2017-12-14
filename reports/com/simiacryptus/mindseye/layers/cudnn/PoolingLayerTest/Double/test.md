# PoolingLayer
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
      "class": "com.simiacryptus.mindseye.layers.cudnn.PoolingLayer",
      "id": "4b1e161b-343d-407a-af28-5533e822ad69",
      "isFrozen": false,
      "name": "PoolingLayer/4b1e161b-343d-407a-af28-5533e822ad69",
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
    	[ [ -1.128, -1.724 ], [ -1.488, 0.8 ], [ 1.828, 1.452 ], [ 0.248, 1.044 ] ],
    	[ [ 0.476, 0.312 ], [ -1.152, 0.128 ], [ 0.776, -0.792 ], [ 0.8, -0.96 ] ],
    	[ [ -1.008, 1.18 ], [ -1.684, -0.016 ], [ 0.9, 0.684 ], [ 0.028, 1.38 ] ],
    	[ [ -0.152, 0.576 ], [ 0.468, -1.708 ], [ -1.516, -1.132 ], [ -1.836, -0.804 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 0.476, 0.8 ], [ 1.828, 1.452 ] ],
    	[ [ 0.468, 1.18 ], [ 0.9, 1.38 ] ]
    ]
    --------------------
    Derivative: 
    [
    	[ [ 0.0, 0.0 ], [ 0.0, 1.0 ], [ 1.0, 1.0 ], [ 0.0, 0.0 ] ],
    	[ [ 1.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ] ],
    	[ [ 0.0, 1.0 ], [ 0.0, 0.0 ], [ 1.0, 0.0 ], [ 0.0, 1.0 ] ],
    	[ [ 0.0, 0.0 ], [ 1.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ] ]
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
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (400#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (160#)}
```



### Differential Validation
Code from [StandardLayerTests.java:109](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L109) executed in 0.02 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ -0.38, -0.136 ], [ 1.076, 1.62 ], [ 0.26, 0.364 ], [ 0.264, -1.496 ] ],
    	[ [ -0.704, -1.832 ], [ -1.288, -1.712 ], [ 1.272, 0.72 ], [ -0.248, -1.212 ] ],
    	[ [ -1.552, -0.048 ], [ -0.344, -0.288 ], [ -1.232, -0.152 ], [ 1.02, 1.304 ] ],
    	[ [ 0.816, -0.808 ], [ 1.956, 0.628 ], [ -0.792, 0.688 ], [ -1.672, 1.508 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.16463284703575956, negative=18, min=1.508, max=1.508, mean=-0.07499999999999998, count=32.0, positive=14, stdDev=1.0750693000918592, zeros=0}
    Output: [
    	[ [ 1.076, 1.62 ], [ 1.272, 0.72 ] ],
    	[ [ 1.956, 0.628 ], [ 1.02, 1.508 ] ]
    ]
    Outputs Statistics: {meanExponent=0.05993461263763956, negative=0, min=1.508, max=1.508, mean=1.225, count=8.0, positive=8, stdDev=0.425491480525756, zeros=0}
    Feedback for input 0
    Inputs Values: [
    	[ [ -0.38, -0.136 ], [ 1.076, 1.62 ], [ 0.26, 0.364 ], [ 0.264, -1.496 ] ],
    	[ [ -0.704, -1.832 ], [ -1.288, -1.712 ], [ 1.272, 0.72 ], [ -0.248, -1.212 ] ],
    	[ [ -1.552, -0.048 ], [ -0.344, -0.288 ], [ -1.232, -0.152 ], 
```
...[skipping 1231 bytes](etc/77.txt)...
```
    d Statistics: {meanExponent=-4.7830642341045674E-14, negative=0, min=0.9999999999998899, max=0.9999999999998899, mean=0.03124999999999656, count=256.0, positive=8, stdDev=0.17399263633841902, zeros=248}
    Feedback Error: [ [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ -1.1013412404281553E-13, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, -1.1013412404281553E-13, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], ... ]
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
Code from [StandardLayerTests.java:119](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L119) executed in 0.19 seconds: 
```java
    getPerformanceTester().test(layer, permPrototype);
```
Logging: 
```
    100 batches
    Input Dimensions:
    	[100, 100, 2]
    Performance:
    	Evaluation performance: 0.012856s +- 0.001272s [0.011465s - 0.014833s]
    	Learning performance: 0.011681s +- 0.000887s [0.010657s - 0.012944s]
    
```

