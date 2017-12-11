# AbsActivationLayer
## AbsActivationLayerTest
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
      "class": "com.simiacryptus.mindseye.layers.java.AbsActivationLayer",
      "id": "e2a3bda5-e7e7-4c05-aeb3-4ede00001dfa",
      "isFrozen": true,
      "name": "AbsActivationLayer/e2a3bda5-e7e7-4c05-aeb3-4ede00001dfa"
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
    	[ [ -1.544 ], [ 0.012 ], [ 0.856 ] ],
    	[ [ -1.388 ], [ -1.1 ], [ -1.36 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 1.544 ], [ 0.012 ], [ 0.856 ] ],
    	[ [ 1.388 ], [ 1.1 ], [ 1.36 ] ]
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
    	[ [ -1.544 ], [ 0.012 ], [ 0.856 ] ],
    	[ [ -1.388 ], [ -1.1 ], [ -1.36 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.24706277227137097, negative=4, min=-1.36, max=-1.36, mean=-0.754, count=6.0, positive=2, stdDev=0.8842948226317585, zeros=0}
    Output: [
    	[ [ 1.544 ], [ 0.012 ], [ 0.856 ] ],
    	[ [ 1.388 ], [ 1.1 ], [ 1.36 ] ]
    ]
    Outputs Statistics: {meanExponent=-0.24706277227137097, negative=0, min=1.36, max=1.36, mean=1.0433333333333334, count=6.0, positive=6, stdDev=0.5118094263384454, zeros=0}
    Feedback for input 0
    Inputs Values: [
    	[ [ -1.544 ], [ 0.012 ], [ 0.856 ] ],
    	[ [ -1.388 ], [ -1.1 ], [ -1.36 ] ]
    ]
    Value Statistics: {meanExponent=-0.24706277227137097, negative=4, min=-1.36, max=-1.36, mean=-0.754, count=6.0, positive=2, stdDev=0.8842948226317585, zeros=0}
    Implemented Feedback: [ [ -1.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, -1.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 1.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, -1.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 1.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, -1.0 ] ]
    Im
```
...[skipping 404 bytes](etc/42.txt)...
```
    .9999999999998899, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, -0.9999999999998899 ] ]
    Measured Statistics: {meanExponent=-4.029281597748704E-14, negative=4, min=-0.9999999999998899, max=-0.9999999999998899, mean=-0.055555555555546546, count=36.0, positive=2, stdDev=0.40445054940443625, zeros=30}
    Feedback Error: [ [ 1.1013412404281553E-13, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 1.1013412404281553E-13, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, -5.995204332975845E-15, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 1.1013412404281553E-13, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, -1.1013412404281553E-13, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 1.1013412404281553E-13 ] ]
    Error Statistics: {meanExponent=-13.168764416758693, negative=2, min=1.1013412404281553E-13, max=1.1013412404281553E-13, mean=9.011310216540854E-15, count=36.0, positive=4, stdDev=4.005559859707196E-14, zeros=30}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 1.5463e-14 +- 3.8034e-14 [0.0000e+00 - 1.1013e-13] (36#)
    relativeTol: 4.6389e-14 +- 1.9405e-14 [2.9976e-15 - 5.5067e-14] (6#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=1.5463e-14 +- 3.8034e-14 [0.0000e+00 - 1.1013e-13] (36#), relativeTol=4.6389e-14 +- 1.9405e-14 [2.9976e-15 - 5.5067e-14] (6#)}
```



### Performance
Code from [LayerTestBase.java:192](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L192) executed in 0.00 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 0.2340 +- 0.1057 [0.1482 - 1.1228]
    Learning performance: 0.0034 +- 0.0103 [0.0000 - 0.1054]
    
```

### Function Plots
Code from [ActivationLayerTestBase.java:103](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/java/ActivationLayerTestBase.java#L103) executed in 0.01 seconds: 
```java
    return plot("Value Plot", plotData, x -> new double[]{x[0], x[1]});
```

Returns: 

![Result](etc/test.12.png)



Code from [ActivationLayerTestBase.java:107](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/java/ActivationLayerTestBase.java#L107) executed in 0.01 seconds: 
```java
    return plot("Derivative Plot", plotData, x -> new double[]{x[0], x[2]});
```

Returns: 

![Result](etc/test.13.png)



