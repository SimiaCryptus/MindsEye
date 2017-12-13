# BandReducerLayer
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
      "class": "com.simiacryptus.mindseye.layers.cudnn.BandReducerLayer",
      "id": "249bb078-df53-439f-a08f-3593c5670e18",
      "isFrozen": false,
      "name": "BandReducerLayer/249bb078-df53-439f-a08f-3593c5670e18",
      "mode": 0
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
    	[ [ 1.944, -1.928 ], [ -0.152, -0.796 ], [ -0.468, -1.68 ] ],
    	[ [ -0.256, -0.108 ], [ 1.716, 1.376 ], [ -1.52, 0.848 ] ],
    	[ [ 1.952, -0.068 ], [ 1.472, -1.7 ], [ -1.1, 0.208 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 0.0, 0.0 ] ]
    ]
    --------------------
    Derivative: 
    [
    	[ [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ] ],
    	[ [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ] ],
    	[ [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ] ]
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
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (200#), relativeTol=0.0000e+00 +- 0.0000e+00 [Infinity - -Infinity] (0#)}
```



### Differential Validation
Code from [StandardLayerTests.java:110](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L110) executed in 0.01 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ -1.884, -0.576 ], [ 0.9, 1.908 ], [ -1.38, -1.28 ] ],
    	[ [ 1.948, -0.38 ], [ -0.044, 1.256 ], [ -0.288, -0.388 ] ],
    	[ [ -1.132, 0.588 ], [ 1.74, 0.36 ], [ 0.516, -0.752 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.14520025331478237, negative=10, min=-0.752, max=-0.752, mean=0.06177777777777782, count=18.0, positive=8, stdDev=1.1346558732132328, zeros=0}
    Output: [
    	[ [ 0.0, 0.0 ] ]
    ]
    Outputs Statistics: {meanExponent=NaN, negative=0, min=0.0, max=0.0, mean=0.0, count=2.0, positive=0, stdDev=0.0, zeros=2}
    Feedback for input 0
    Inputs Values: [
    	[ [ -1.884, -0.576 ], [ 0.9, 1.908 ], [ -1.38, -1.28 ] ],
    	[ [ 1.948, -0.38 ], [ -0.044, 1.256 ], [ -0.288, -0.388 ] ],
    	[ [ -1.132, 0.588 ], [ 1.74, 0.36 ], [ 0.516, -0.752 ] ]
    ]
    Value Statistics: {meanExponent=-0.14520025331478237, negative=10, min=-0.752, max=-0.752, mean=0.06177777777777782, count=18.0, positive=8, stdDev=1.1346558732132328, zeros=0}
    Implemented Feedback: [ [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], ... ]
    Implemented Statistics: {meanExponent=NaN, negative=0, min=0.0, max=0.0, mean=0.0, count=36.0, positive=0, stdDev=0.0, zeros=36}
    Measured Feedback: [ [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], ... ]
    Measured Statistics: {meanExponent=NaN, negative=0, min=0.0, max=0.0, mean=0.0, count=36.0, positive=0, stdDev=0.0, zeros=36}
    Feedback Error: [ [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], ... ]
    Error Statistics: {meanExponent=NaN, negative=0, min=0.0, max=0.0, mean=0.0, count=36.0, positive=0, stdDev=0.0, zeros=36}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (36#)
    relativeTol: 0.0000e+00 +- 0.0000e+00 [Infinity - -Infinity] (0#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (36#), relativeTol=0.0000e+00 +- 0.0000e+00 [Infinity - -Infinity] (0#)}
```



### Performance
Code from [StandardLayerTests.java:120](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L120) executed in 0.01 seconds: 
```java
    getPerformanceTester().test(layer, permPrototype);
```
Logging: 
```
    Evaluation performance: 0.000267s +- 0.000040s [0.000232s - 0.000344s]
    Learning performance: 0.000321s +- 0.000068s [0.000253s - 0.000453s]
    
```

