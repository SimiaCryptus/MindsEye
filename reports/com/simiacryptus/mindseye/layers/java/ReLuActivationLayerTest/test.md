# ReLuActivationLayer
## ReLuActivationLayerTest
### Json Serialization
Code from [LayerTestBase.java:121](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L121) executed in 0.00 seconds: 
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
      "class": "com.simiacryptus.mindseye.layers.java.ReLuActivationLayer",
      "id": "e2a3bda5-e7e7-4c05-aeb3-4ede0000360d",
      "isFrozen": true,
      "name": "ReLuActivationLayer/e2a3bda5-e7e7-4c05-aeb3-4ede0000360d",
      "weights": [
        1.0
      ]
    }
```



### Example Input/Output Pair
Code from [LayerTestBase.java:159](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L159) executed in 0.00 seconds: 
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
    	[ [ 0.328 ], [ 0.256 ], [ 0.244 ] ],
    	[ [ 1.592 ], [ -1.432 ], [ -0.056 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 0.328 ], [ 0.256 ], [ 0.244 ] ],
    	[ [ 1.592 ], [ 0.0 ], [ 0.0 ] ]
    ]
```



### Batch Execution
Code from [LayerTestBase.java:178](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L178) executed in 0.00 seconds: 
```java
    BatchingTester batchingTester = getBatchingTester();
    return batchingTester == null ? null : batchingTester.test(layer, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (120#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (74#)}
```



### Differential Validation
Code from [LayerTestBase.java:186](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L186) executed in 0.00 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ 0.328 ], [ 0.256 ], [ 0.244 ] ],
    	[ [ 1.592 ], [ -1.432 ], [ -0.056 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.43040370937634237, negative=2, min=-0.056, max=-0.056, mean=0.15533333333333335, count=6.0, positive=4, stdDev=0.8828089009267835, zeros=0}
    Output: [
    	[ [ 0.328 ], [ 0.256 ], [ 0.244 ] ],
    	[ [ 1.592 ], [ 0.0 ], [ 0.0 ] ]
    ]
    Outputs Statistics: {meanExponent=-0.3716383253090229, negative=0, min=0.0, max=0.0, mean=0.4033333333333333, count=6.0, positive=4, stdDev=0.5463596698960209, zeros=2}
    Feedback for input 0
    Inputs Values: [
    	[ [ 0.328 ], [ 0.256 ], [ 0.244 ] ],
    	[ [ 1.592 ], [ -1.432 ], [ -0.056 ] ]
    ]
    Value Statistics: {meanExponent=-0.43040370937634237, negative=2, min=-0.056, max=-0.056, mean=0.15533333333333335, count=6.0, positive=4, stdDev=0.8828089009267835, zeros=0}
    Implemented Feedback: [ [ 1.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 1.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 1.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 1.0, 0.0 ], [ 0.0, 0.0, 0.
```
...[skipping 1178 bytes](etc/82.txt)...
```
    .0 ]
    Implemented Gradient: [ [ 0.328, 1.592, 0.256, 0.0, 0.244, 0.0 ] ]
    Implemented Statistics: {meanExponent=-0.3716383253090229, negative=0, min=0.0, max=0.0, mean=0.4033333333333333, count=6.0, positive=4, stdDev=0.5463596698960209, zeros=2}
    Measured Gradient: [ [ 0.32799999999999496, 1.5919999999991497, 0.256000000000145, 0.0, 0.2440000000000775, 0.0 ] ]
    Measured Statistics: {meanExponent=-0.37163832530898666, negative=0, min=0.0, max=0.0, mean=0.40333333333322785, count=6.0, positive=4, stdDev=0.5463596698957024, zeros=2}
    Gradient Error: [ [ -5.051514762044462E-15, -8.504308368628699E-13, 1.4499512701604544E-13, 0.0, 7.749356711883593E-14, 0.0 ] ]
    Error Statistics: {meanExponent=-13.079080078728019, negative=2, min=0.0, max=0.0, mean=-1.054989429150055E-13, count=6.0, positive=2, stdDev=3.3751711170318744E-13, zeros=2}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 3.6155e-14 +- 1.3309e-13 [0.0000e+00 - 8.5043e-13] (42#)
    relativeTol: 1.1713e-13 +- 9.9489e-14 [7.7005e-15 - 2.8319e-13] (8#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=3.6155e-14 +- 1.3309e-13 [0.0000e+00 - 8.5043e-13] (42#), relativeTol=1.1713e-13 +- 9.9489e-14 [7.7005e-15 - 2.8319e-13] (8#)}
```



### Performance
Code from [LayerTestBase.java:192](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L192) executed in 0.00 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 0.1992 +- 0.0566 [0.1624 - 0.6612]
    Learning performance: 0.0397 +- 0.0152 [0.0313 - 0.1539]
    
```

### Function Plots
Code from [ActivationLayerTestBase.java:103](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/java/ActivationLayerTestBase.java#L103) executed in 0.00 seconds: 
```java
    return plot("Value Plot", plotData, x -> new double[]{x[0], x[1]});
```

Returns: 

![Result](etc/test.44.png)



Code from [ActivationLayerTestBase.java:107](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/java/ActivationLayerTestBase.java#L107) executed in 0.00 seconds: 
```java
    return plot("Derivative Plot", plotData, x -> new double[]{x[0], x[2]});
```

Returns: 

![Result](etc/test.45.png)



