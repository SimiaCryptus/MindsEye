# BinarySumLayer
## Float_Add
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
      "class": "com.simiacryptus.mindseye.layers.cudnn.BinarySumLayer",
      "id": "2a8a74f6-fbb5-43e1-96e9-b5b0eff1b662",
      "isFrozen": false,
      "name": "BinarySumLayer/2a8a74f6-fbb5-43e1-96e9-b5b0eff1b662",
      "rightFactor": 1.0,
      "leftFactor": 1.0
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
    	[ [ 0.432 ], [ -1.972 ] ],
    	[ [ 0.124 ], [ 1.82 ] ]
    ],
    [
    	[ [ 0.316 ], [ 0.256 ] ],
    	[ [ 1.632 ], [ -0.64 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 0.0 ], [ 0.0 ] ],
    	[ [ 0.0 ], [ 0.0 ] ]
    ]
    --------------------
    Derivative: 
    [
    	[ [ 0.0 ], [ 0.0 ] ],
    	[ [ 0.0 ], [ 0.0 ] ]
    ],
    [
    	[ [ 0.0 ], [ 0.0 ] ],
    	[ [ 0.0 ], [ 0.0 ] ]
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
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (120#), relativeTol=0.0000e+00 +- 0.0000e+00 [Infinity - -Infinity] (0#)}
```



### Differential Validation
Code from [StandardLayerTests.java:110](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L110) executed in 0.01 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ -0.28 ], [ 1.916 ] ],
    	[ [ -1.624 ], [ -1.72 ] ]
    ],
    [
    	[ [ -0.656 ], [ -1.484 ] ],
    	[ [ 0.552 ], [ 1.816 ] ]
    ]
    Inputs Statistics: {meanExponent=0.04391700197436257, negative=3, min=-1.72, max=-1.72, mean=-0.42700000000000005, count=4.0, positive=1, stdDev=1.4676440304106442, zeros=0},
    {meanExponent=-0.0026518344417665457, negative=2, min=1.816, max=1.816, mean=0.056999999999999995, count=4.0, positive=2, stdDev=1.2472124919194805, zeros=0}
    Output: [
    	[ [ 0.0 ], [ 0.0 ] ],
    	[ [ 0.0 ], [ 0.0 ] ]
    ]
    Outputs Statistics: {meanExponent=NaN, negative=0, min=0.0, max=0.0, mean=0.0, count=4.0, positive=0, stdDev=0.0, zeros=4}
    Feedback for input 0
    Inputs Values: [
    	[ [ -0.28 ], [ 1.916 ] ],
    	[ [ -1.624 ], [ -1.72 ] ]
    ]
    Value Statistics: {meanExponent=0.04391700197436257, negative=3, min=-1.72, max=-1.72, mean=-0.42700000000000005, count=4.0, positive=1, stdDev=1.4676440304106442, zeros=0}
    Implemented Feedback: [ [ 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0
```
...[skipping 793 bytes](etc/10.txt)...
```
    in=1.816, max=1.816, mean=0.056999999999999995, count=4.0, positive=2, stdDev=1.2472124919194805, zeros=0}
    Implemented Feedback: [ [ 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0 ] ]
    Implemented Statistics: {meanExponent=NaN, negative=0, min=0.0, max=0.0, mean=0.0, count=16.0, positive=0, stdDev=0.0, zeros=16}
    Measured Feedback: [ [ 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0 ] ]
    Measured Statistics: {meanExponent=NaN, negative=0, min=0.0, max=0.0, mean=0.0, count=16.0, positive=0, stdDev=0.0, zeros=16}
    Feedback Error: [ [ 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0 ] ]
    Error Statistics: {meanExponent=NaN, negative=0, min=0.0, max=0.0, mean=0.0, count=16.0, positive=0, stdDev=0.0, zeros=16}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (32#)
    relativeTol: 0.0000e+00 +- 0.0000e+00 [Infinity - -Infinity] (0#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (32#), relativeTol=0.0000e+00 +- 0.0000e+00 [Infinity - -Infinity] (0#)}
```



### Performance
Code from [StandardLayerTests.java:120](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L120) executed in 0.01 seconds: 
```java
    getPerformanceTester().test(layer, permPrototype);
```
Logging: 
```
    Evaluation performance: 0.000310s +- 0.000030s [0.000266s - 0.000351s]
    Learning performance: 0.000214s +- 0.000016s [0.000192s - 0.000243s]
    
```

