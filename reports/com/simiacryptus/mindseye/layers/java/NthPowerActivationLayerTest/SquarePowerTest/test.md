# NthPowerActivationLayer
## SquarePowerTest
### Json Serialization
Code from [LayerTestBase.java:121](../../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L121) executed in 0.00 seconds: 
```java
    JsonObject json = layer.getJson();
    NNLayer echo = NNLayer.fromJson(json);
    assert (echo != null) : "Failed to deserialize";
    assert (layer != echo) : "Serialization did not copy";
    Assert.assertEquals("Serialization not equal", layer, echo);
    return new GsonBuilder().setPrettyPrinting().create().toJson(json);
```

Returns: 

```
    {
      "class": "com.simiacryptus.mindseye.layers.java.NthPowerActivationLayer",
      "id": "e2a3bda5-e7e7-4c05-aeb3-4ede00001f17",
      "isFrozen": false,
      "name": "NthPowerActivationLayer/e2a3bda5-e7e7-4c05-aeb3-4ede00001f17",
      "power": 2.0
    }
```



### Example Input/Output Pair
Code from [LayerTestBase.java:159](../../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L159) executed in 0.00 seconds: 
```java
    SimpleEval eval = SimpleEval.run(layer, inputPrototype);
    return String.format("--------------------\nInput: \n[%s]\n--------------------\nOutput: \n%s",
      Arrays.stream(inputPrototype).map(t -> t.prettyPrint()).reduce((a, b) -> a + ",\n" + b).get(),
      eval.getOutput().prettyPrint());
```

Returns: 

```
    --------------------
    Input: 
    [[
    	[ [ 0.464 ], [ 0.94 ], [ -0.292 ] ],
    	[ [ -0.636 ], [ -0.568 ], [ -1.016 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 0.21529600000000002 ], [ 0.8835999999999999 ], [ 0.08526399999999999 ] ],
    	[ [ 0.404496 ], [ 0.32262399999999997 ], [ 1.032256 ] ]
    ]
```



### Batch Execution
Code from [LayerTestBase.java:178](../../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L178) executed in 0.00 seconds: 
```java
    BatchingTester batchingTester = getBatchingTester();
    return batchingTester == null ? null : batchingTester.test(layer, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (120#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (120#)}
```



### Differential Validation
Code from [LayerTestBase.java:186](../../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L186) executed in 0.00 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ 0.464 ], [ 0.94 ], [ -0.292 ] ],
    	[ [ -0.636 ], [ -0.568 ], [ -1.016 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.22171202584827818, negative=4, min=-1.016, max=-1.016, mean=-0.18466666666666665, count=6.0, positive=2, stdDev=0.6756386279332729, zeros=0}
    Output: [
    	[ [ 0.21529600000000002 ], [ 0.8835999999999999 ], [ 0.08526399999999999 ] ],
    	[ [ 0.404496 ], [ 0.32262399999999997 ], [ 1.032256 ] ]
    ]
    Outputs Statistics: {meanExponent=-0.44342405169655635, negative=0, min=1.032256, max=1.032256, mean=0.4905893333333333, count=6.0, positive=6, stdDev=0.34726541779003695, zeros=0}
    Feedback for input 0
    Inputs Values: [
    	[ [ 0.464 ], [ 0.94 ], [ -0.292 ] ],
    	[ [ -0.636 ], [ -0.568 ], [ -1.016 ] ]
    ]
    Value Statistics: {meanExponent=-0.22171202584827818, negative=4, min=-1.016, max=-1.016, mean=-0.18466666666666665, count=6.0, positive=2, stdDev=0.6756386279332729, zeros=0}
    Implemented Feedback: [ [ 0.928, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, -1.272, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 1.88, 0.0, 0.0, 0.
```
...[skipping 521 bytes](etc/77.txt)...
```
     0.0, 0.0, 0.0, 0.0, -0.583899999999915, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, -2.031899999999087 ] ]
    Measured Statistics: {meanExponent=0.07930159908913477, negative=4, min=-2.031899999999087, max=-2.031899999999087, mean=-0.06153888888888675, count=36.0, positive=2, stdDev=0.5685597692989052, zeros=30}
    Feedback Error: [ [ 9.999999998722142E-5, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 1.000000000075385E-4, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 9.999999905208057E-5, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 1.000000000324075E-4, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 1.0000000008492105E-4, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 1.0000000091281436E-4 ] ]
    Error Statistics: {meanExponent=-3.999999999944278, negative=0, min=1.0000000091281436E-4, max=1.0000000091281436E-4, mean=1.6666666668805094E-5, count=36.0, positive=6, stdDev=3.7267799629778164E-5, zeros=30}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 1.6667e-05 +- 3.7268e-05 [0.0000e+00 - 1.0000e-04] (36#)
    relativeTol: 4.5671e-05 +- 2.0471e-05 [2.4607e-05 - 8.5624e-05] (6#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=1.6667e-05 +- 3.7268e-05 [0.0000e+00 - 1.0000e-04] (36#), relativeTol=4.5671e-05 +- 2.0471e-05 [2.4607e-05 - 8.5624e-05] (6#)}
```



### Performance
Code from [LayerTestBase.java:192](../../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L192) executed in 0.00 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 0.1396 +- 0.0521 [0.0912 - 0.5871]
    Learning performance: 0.0010 +- 0.0017 [0.0000 - 0.0114]
    
```

### Function Plots
Code from [ActivationLayerTestBase.java:103](../../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/java/ActivationLayerTestBase.java#L103) executed in 0.00 seconds: 
```java
    return plot("Value Plot", plotData, x -> new double[]{x[0], x[1]});
```

Returns: 

![Result](etc/test.40.png)



Code from [ActivationLayerTestBase.java:107](../../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/java/ActivationLayerTestBase.java#L107) executed in 0.00 seconds: 
```java
    return plot("Derivative Plot", plotData, x -> new double[]{x[0], x[2]});
```

Returns: 

![Result](etc/test.41.png)



