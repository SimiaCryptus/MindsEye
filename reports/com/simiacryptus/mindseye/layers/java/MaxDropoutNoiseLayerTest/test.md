# MaxDropoutNoiseLayer
## MaxDropoutNoiseLayerTest
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
      "class": "com.simiacryptus.mindseye.layers.java.MaxDropoutNoiseLayer",
      "id": "e2a3bda5-e7e7-4c05-aeb3-4ede00001ed2",
      "isFrozen": false,
      "name": "MaxDropoutNoiseLayer/e2a3bda5-e7e7-4c05-aeb3-4ede00001ed2",
      "kernelSize": [
        2,
        2,
        1
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
    	[ [ 0.7 ], [ 0.608 ], [ -0.844 ], [ -0.584 ] ],
    	[ [ -1.932 ], [ 0.128 ], [ -1.636 ], [ -0.744 ] ],
    	[ [ -0.06 ], [ -0.076 ], [ 1.28 ], [ 0.428 ] ],
    	[ [ 0.696 ], [ -0.716 ], [ 0.404 ], [ 0.176 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 0.7 ], [ 0.0 ], [ -0.0 ], [ -0.584 ] ],
    	[ [ -0.0 ], [ 0.0 ], [ -0.0 ], [ -0.0 ] ],
    	[ [ -0.0 ], [ -0.0 ], [ 1.28 ], [ 0.0 ] ],
    	[ [ 0.696 ], [ -0.0 ], [ 0.0 ], [ 0.0 ] ]
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
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (320#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (80#)}
```



### Differential Validation
Code from [LayerTestBase.java:186](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L186) executed in 0.00 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ 0.7 ], [ 0.608 ], [ -0.844 ], [ -0.584 ] ],
    	[ [ -1.932 ], [ 0.128 ], [ -1.636 ], [ -0.744 ] ],
    	[ [ -0.06 ], [ -0.076 ], [ 1.28 ], [ 0.428 ] ],
    	[ [ 0.696 ], [ -0.716 ], [ 0.404 ], [ 0.176 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.32828968026236127, negative=8, min=0.176, max=0.176, mean=-0.13574999999999998, count=16.0, positive=8, stdDev=0.8521836289791069, zeros=0}
    Output: [
    	[ [ 0.7 ], [ 0.0 ], [ -0.0 ], [ -0.584 ] ],
    	[ [ -0.0 ], [ 0.0 ], [ -0.0 ], [ -0.0 ] ],
    	[ [ -0.0 ], [ -0.0 ], [ 1.28 ], [ 0.0 ] ],
    	[ [ 0.696 ], [ -0.0 ], [ 0.0 ], [ 0.0 ] ]
    ]
    Outputs Statistics: {meanExponent=-0.1096674759037283, negative=1, min=0.0, max=0.0, mean=0.13075, count=16.0, positive=3, stdDev=0.40929382783032536, zeros=12}
    Feedback for input 0
    Inputs Values: [
    	[ [ 0.7 ], [ 0.608 ], [ -0.844 ], [ -0.584 ] ],
    	[ [ -1.932 ], [ 0.128 ], [ -1.636 ], [ -0.744 ] ],
    	[ [ -0.06 ], [ -0.076 ], [ 1.28 ], [ 0.428 ] ],
    	[ [ 0.696 ], [ -0.716 ], [ 0.404 ], [ 0.176 ] ]
    ]
    Value Statistics: {meanExponent=-0.32828968
```
...[skipping 1135 bytes](etc/69.txt)...
```
    0.0, 0.0, ... ], ... ]
    Measured Statistics: {meanExponent=-4.7830642341045674E-14, negative=0, min=0.0, max=0.0, mean=0.01562499999999828, count=256.0, positive=4, stdDev=0.12401959270613902, zeros=252}
    Feedback Error: [ [ -1.1013412404281553E-13, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, -1.1013412404281553E-13, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], ... ]
    Error Statistics: {meanExponent=-12.958078098036825, negative=4, min=0.0, max=0.0, mean=-1.7208456881689926E-15, count=256.0, positive=0, stdDev=1.365878920683888E-14, zeros=252}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 1.7208e-15 +- 1.3659e-14 [0.0000e+00 - 1.1013e-13] (256#)
    relativeTol: 5.5067e-14 +- 0.0000e+00 [5.5067e-14 - 5.5067e-14] (4#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=1.7208e-15 +- 1.3659e-14 [0.0000e+00 - 1.1013e-13] (256#), relativeTol=5.5067e-14 +- 0.0000e+00 [5.5067e-14 - 5.5067e-14] (4#)}
```



### Performance
Code from [LayerTestBase.java:192](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L192) executed in 0.01 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 0.2514 +- 0.1143 [0.1653 - 1.0801]
    Learning performance: 0.0019 +- 0.0033 [0.0000 - 0.0285]
    
```

### Function Plots
Code from [ActivationLayerTestBase.java:103](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/java/ActivationLayerTestBase.java#L103) executed in 0.00 seconds: 
```java
    return plot("Value Plot", plotData, x -> new double[]{x[0], x[1]});
```

Returns: 

![Result](etc/test.30.png)



Code from [ActivationLayerTestBase.java:107](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/java/ActivationLayerTestBase.java#L107) executed in 0.00 seconds: 
```java
    return plot("Derivative Plot", plotData, x -> new double[]{x[0], x[2]});
```

Returns: 

![Result](etc/test.31.png)



