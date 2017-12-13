# MaxSubsampleLayer
## MaxSubsampleLayerTest
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
      "class": "com.simiacryptus.mindseye.layers.java.MaxSubsampleLayer",
      "id": "67b57b59-b985-4fcb-9309-516236923a5a",
      "isFrozen": false,
      "name": "MaxSubsampleLayer/67b57b59-b985-4fcb-9309-516236923a5a",
      "inner": [
        2,
        2,
        1
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
    	[ [ -1.016, 0.588, 0.192 ], [ -1.728, 0.408, 0.18 ] ],
    	[ [ 0.516, -1.116, -0.516 ], [ -1.12, 0.02, 1.056 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 0.516, 0.588, 1.056 ] ]
    ]
    --------------------
    Derivative: 
    [
    	[ [ 0.0, 1.0, 0.0 ], [ 0.0, 0.0, 0.0 ] ],
    	[ [ 1.0, 0.0, 0.0 ], [ 0.0, 0.0, 1.0 ] ]
    ]
```



### Batch Execution
Code from [StandardLayerTests.java:102](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L102) executed in 0.00 seconds: 
```java
    return getBatchingTester().test(layer, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (150#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (60#)}
```



### Differential Validation
Code from [StandardLayerTests.java:110](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L110) executed in 0.00 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ 0.288, -1.352, 0.012 ], [ 0.856, 0.356, 0.788 ] ],
    	[ [ -0.084, -0.812, 0.916 ], [ -0.688, -1.512, -0.124 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.4203089075761753, negative=6, min=-0.124, max=-0.124, mean=-0.11299999999999999, count=12.0, positive=6, stdDev=0.7915438501224469, zeros=0}
    Output: [
    	[ [ 0.856, 0.356, 0.916 ] ]
    ]
    Outputs Statistics: {meanExponent=-0.18472692122737375, negative=0, min=0.916, max=0.916, mean=0.7093333333333334, count=3.0, positive=3, stdDev=0.251042271783503, zeros=0}
    Feedback for input 0
    Inputs Values: [
    	[ [ 0.288, -1.352, 0.012 ], [ 0.856, 0.356, 0.788 ] ],
    	[ [ -0.084, -0.812, 0.916 ], [ -0.688, -1.512, -0.124 ] ]
    ]
    Value Statistics: {meanExponent=-0.4203089075761753, negative=6, min=-0.124, max=-0.124, mean=-0.11299999999999999, count=12.0, positive=6, stdDev=0.7915438501224469, zeros=0}
    Implemented Feedback: [ [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 1.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 1.0, 0.0 ], [ 0.0, 0.0,
```
...[skipping 116 bytes](etc/84.txt)...
```
    unt=36.0, positive=3, stdDev=0.2763853991962833, zeros=33}
    Measured Feedback: [ [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.9999999999998899, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.9999999999998899, 0.0 ], [ 0.0, 0.0, 0.0 ], ... ]
    Measured Statistics: {meanExponent=-4.7830642341045674E-14, negative=0, min=0.0, max=0.0, mean=0.08333333333332416, count=36.0, positive=3, stdDev=0.2763853991962529, zeros=33}
    Feedback Error: [ [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ -1.1013412404281553E-13, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, -1.1013412404281553E-13, 0.0 ], [ 0.0, 0.0, 0.0 ], ... ]
    Error Statistics: {meanExponent=-12.958078098036827, negative=3, min=0.0, max=0.0, mean=-9.177843670234628E-15, count=36.0, positive=0, stdDev=3.0439463838706555E-14, zeros=33}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 9.1778e-15 +- 3.0439e-14 [0.0000e+00 - 1.1013e-13] (36#)
    relativeTol: 5.5067e-14 +- 0.0000e+00 [5.5067e-14 - 5.5067e-14] (3#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=9.1778e-15 +- 3.0439e-14 [0.0000e+00 - 1.1013e-13] (36#), relativeTol=5.5067e-14 +- 0.0000e+00 [5.5067e-14 - 5.5067e-14] (3#)}
```



### Performance
Code from [StandardLayerTests.java:120](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L120) executed in 0.01 seconds: 
```java
    getPerformanceTester().test(layer, permPrototype);
```
Logging: 
```
    Evaluation performance: 0.001431s +- 0.000401s [0.000925s - 0.002078s]
    Learning performance: 0.000037s +- 0.000007s [0.000031s - 0.000048s]
    
```

