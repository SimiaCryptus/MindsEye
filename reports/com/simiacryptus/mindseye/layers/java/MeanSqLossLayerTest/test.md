# MeanSqLossLayer
## MeanSqLossLayerTest
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
      "class": "com.simiacryptus.mindseye.layers.java.MeanSqLossLayer",
      "id": "efc3c2c1-c43d-4a9c-8377-0da55cdd6ba5",
      "isFrozen": false,
      "name": "MeanSqLossLayer/efc3c2c1-c43d-4a9c-8377-0da55cdd6ba5"
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
    	[ [ 0.552 ], [ 0.816 ], [ -0.14 ] ],
    	[ [ -0.736 ], [ 1.04 ], [ -1.632 ] ]
    ],
    [
    	[ [ 0.04 ], [ -1.976 ], [ 1.5 ] ],
    	[ [ 1.292 ], [ 1.372 ], [ -1.168 ] ]
    ]]
    --------------------
    Output: 
    [ 2.5308853333333334 ]
    --------------------
    Derivative: 
    [
    	[ [ 0.17066666666666666 ], [ 0.9306666666666665 ], [ -0.5466666666666666 ] ],
    	[ [ -0.6759999999999999 ], [ -0.11066666666666669 ], [ -0.15466666666666665 ] ]
    ],
    [
    	[ [ -0.17066666666666666 ], [ -0.9306666666666665 ], [ 0.5466666666666666 ] ],
    	[ [ 0.6759999999999999 ], [ 0.11066666666666669 ], [ 0.15466666666666665 ] ]
    ]
```



### Batch Execution
Code from [StandardLayerTests.java:102](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L102) executed in 0.00 seconds: 
```java
    return getBatchingTester().test(layer, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (130#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (130#)}
```



### Differential Validation
Code from [StandardLayerTests.java:110](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L110) executed in 0.00 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ -0.44 ], [ -1.94 ], [ 1.368 ] ],
    	[ [ -0.276 ], [ -1.076 ], [ 0.984 ] ]
    ],
    [
    	[ [ -0.816 ], [ 0.272 ], [ -1.792 ] ],
    	[ [ -1.86 ], [ -1.996 ], [ -0.868 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.07782384072875992, negative=4, min=0.984, max=0.984, mean=-0.22999999999999998, count=6.0, positive=2, stdDev=1.1338306751892013, zeros=0},
    {meanExponent=0.01796504574332114, negative=5, min=-0.868, max=-0.868, mean=-1.1766666666666667, count=6.0, positive=1, stdDev=0.8000913836695295, zeros=0}
    Output: [ 3.6342133333333337 ]
    Outputs Statistics: {meanExponent=0.5604104174051727, negative=0, min=3.6342133333333337, max=3.6342133333333337, mean=3.6342133333333337, count=1.0, positive=1, stdDev=0.0, zeros=0}
    Feedback for input 0
    Inputs Values: [
    	[ [ -0.44 ], [ -1.94 ], [ 1.368 ] ],
    	[ [ -0.276 ], [ -1.076 ], [ 0.984 ] ]
    ]
    Value Statistics: {meanExponent=-0.07782384072875992, negative=4, min=0.984, max=0.984, mean=-0.22999999999999998, count=6.0, positive=2, stdDev=1.1338306751892013, zeros=0}
    Impleme
```
...[skipping 1710 bytes](etc/85.txt)...
```
    55555555555553, count=6.0, positive=1, stdDev=0.5515670157283755, zeros=0}
    Measured Feedback: [ [ -0.125316666670372 ], [ -0.5279833333382555 ], [ 0.7373499999996369 ], [ -0.3066500000059591 ], [ -1.053316666670412 ], [ -0.6173166666645358 ] ]
    Measured Statistics: {meanExponent=-0.3353312233076764, negative=5, min=-0.6173166666645358, max=-0.6173166666645358, mean=-0.31553888889164955, count=6.0, positive=1, stdDev=0.5515670157289878, zeros=0}
    Feedback Error: [ [ 1.6666662961295486E-5 ], [ 1.666666174449105E-5 ], [ 1.6666666303732924E-5 ], [ 1.6666660707542746E-5 ], [ 1.6666662921327458E-5 ], [ 1.6666668797515882E-5 ] ]
    Error Statistics: {meanExponent=-4.778151322320604, negative=0, min=1.6666668797515882E-5, max=1.6666668797515882E-5, mean=1.6666663905984258E-5, count=6.0, positive=6, stdDev=2.7847474288855415E-12, zeros=0}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 1.6667e-05 +- 2.4908e-12 [1.6667e-05 - 1.6667e-05] (12#)
    relativeTol: 2.3693e-05 +- 2.0053e-05 [7.9113e-06 - 6.6494e-05] (12#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=1.6667e-05 +- 2.4908e-12 [1.6667e-05 - 1.6667e-05] (12#), relativeTol=2.3693e-05 +- 2.0053e-05 [7.9113e-06 - 6.6494e-05] (12#)}
```



### Performance
Code from [StandardLayerTests.java:120](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L120) executed in 0.00 seconds: 
```java
    getPerformanceTester().test(layer, permPrototype);
```
Logging: 
```
    Evaluation performance: 0.000247s +- 0.000011s [0.000226s - 0.000260s]
    Learning performance: 0.000065s +- 0.000015s [0.000053s - 0.000093s]
    
```

