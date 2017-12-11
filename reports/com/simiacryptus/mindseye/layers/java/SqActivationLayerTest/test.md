# SqActivationLayer
## SqActivationLayerTest
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
      "class": "com.simiacryptus.mindseye.layers.java.SqActivationLayer",
      "id": "e2a3bda5-e7e7-4c05-aeb3-4ede00003669",
      "isFrozen": true,
      "name": "SqActivationLayer/e2a3bda5-e7e7-4c05-aeb3-4ede00003669"
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
    	[ [ 1.436 ], [ 0.332 ], [ -0.816 ] ],
    	[ [ -0.8 ], [ -0.796 ], [ -0.036 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 2.062096 ], [ 0.11022400000000002 ], [ 0.6658559999999999 ] ],
    	[ [ 0.6400000000000001 ], [ 0.6336160000000001 ], [ 0.0012959999999999998 ] ]
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
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (120#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (120#)}
```



### Differential Validation
Code from [LayerTestBase.java:186](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L186) executed in 0.00 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ 1.436 ], [ 0.332 ], [ -0.816 ] ],
    	[ [ -0.8 ], [ -0.796 ], [ -0.036 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.34161862702315354, negative=4, min=-0.036, max=-0.036, mean=-0.11333333333333334, count=6.0, positive=2, stdDev=0.8201647530967313, zeros=0}
    Output: [
    	[ [ 2.062096 ], [ 0.11022400000000002 ], [ 0.6658559999999999 ] ],
    	[ [ 0.6400000000000001 ], [ 0.6336160000000001 ], [ 0.0012959999999999998 ] ]
    ]
    Outputs Statistics: {meanExponent=-0.6832372540463071, negative=0, min=0.0012959999999999998, max=0.0012959999999999998, mean=0.6855146666666667, count=6.0, positive=6, stdDev=0.6707262509038658, zeros=0}
    Feedback for input 0
    Inputs Values: [
    	[ [ 1.436 ], [ 0.332 ], [ -0.816 ] ],
    	[ [ -0.8 ], [ -0.796 ], [ -0.036 ] ]
    ]
    Value Statistics: {meanExponent=-0.34161862702315354, negative=4, min=-0.036, max=-0.036, mean=-0.11333333333333334, count=6.0, positive=2, stdDev=0.8201647530967313, zeros=0}
    Implemented Feedback: [ [ 2.872, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, -1.6, 0.0, 0.0, 0.0, 0.0 ],
```
...[skipping 562 bytes](etc/90.txt)...
```
    0.0, 0.0, 0.0, -1.6318999999997974, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, -0.07190000000000235 ] ]
    Measured Statistics: {meanExponent=-0.04068931813889759, negative=4, min=-0.07190000000000235, max=-0.07190000000000235, mean=-0.037761111111168144, count=36.0, positive=2, stdDev=0.6749639415377024, zeros=30}
    Feedback Error: [ [ 9.999999760168521E-5, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 1.0000000001264553E-4, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 9.999999988974384E-5, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 1.0000000024268374E-4, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 1.0000000020249367E-4, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 9.999999999764364E-5 ] ]
    Error Statistics: {meanExponent=-4.000000001486087, negative=0, min=9.999999999764364E-5, max=9.999999999764364E-5, mean=1.666666660963599E-5, count=36.0, positive=6, stdDev=3.726779949747203E-5, zeros=30}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 1.6667e-05 +- 3.7268e-05 [0.0000e+00 - 1.0000e-04] (36#)
    relativeTol: 1.4682e-04 +- 2.4578e-04 [1.7409e-05 - 6.9493e-04] (6#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=1.6667e-05 +- 3.7268e-05 [0.0000e+00 - 1.0000e-04] (36#), relativeTol=1.4682e-04 +- 2.4578e-04 [1.7409e-05 - 6.9493e-04] (6#)}
```



### Performance
Code from [LayerTestBase.java:192](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L192) executed in 0.00 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 0.1340 +- 0.0466 [0.0855 - 0.4560]
    Learning performance: 0.0011 +- 0.0025 [0.0000 - 0.0171]
    
```

### Function Plots
Code from [ActivationLayerTestBase.java:103](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/java/ActivationLayerTestBase.java#L103) executed in 0.00 seconds: 
```java
    return plot("Value Plot", plotData, x -> new double[]{x[0], x[1]});
```

Returns: 

![Result](etc/test.51.png)



Code from [ActivationLayerTestBase.java:107](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/java/ActivationLayerTestBase.java#L107) executed in 0.00 seconds: 
```java
    return plot("Derivative Plot", plotData, x -> new double[]{x[0], x[2]});
```

Returns: 

![Result](etc/test.52.png)



