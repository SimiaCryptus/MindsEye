# GaussianNoiseLayer
## GaussianNoiseLayerTest
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
      "class": "com.simiacryptus.mindseye.layers.java.GaussianNoiseLayer",
      "id": "e2a3bda5-e7e7-4c05-aeb3-4ede00001e5c",
      "isFrozen": false,
      "name": "GaussianNoiseLayer/e2a3bda5-e7e7-4c05-aeb3-4ede00001e5c",
      "value": 1.0
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
    	[ [ 1.56 ], [ -0.5 ], [ 0.376 ] ],
    	[ [ 1.364 ], [ 0.34 ], [ -0.104 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 1.5962047815389786 ], [ -2.1222266857164005 ], [ 0.6541620186794119 ] ],
    	[ [ 2.967903000725274 ], [ 1.4200658950631417 ], [ 0.44020697761200256 ] ]
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
    	[ [ 1.56 ], [ -0.5 ], [ 0.376 ] ],
    	[ [ 1.364 ], [ 0.34 ], [ -0.104 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.3082318209533938, negative=2, min=-0.104, max=-0.104, mean=0.506, count=6.0, positive=4, stdDev=0.7388811361330951, zeros=0}
    Output: [
    	[ [ 1.5962047815389786 ], [ -2.1222266857164005 ], [ 0.6541620186794119 ] ],
    	[ [ 2.967903000725274 ], [ 1.4200658950631417 ], [ 0.44020697761200256 ] ]
    ]
    Outputs Statistics: {meanExponent=0.10233013766541958, negative=1, min=0.44020697761200256, max=0.44020697761200256, mean=0.8260526646504015, count=6.0, positive=5, stdDev=1.5494237250977712, zeros=0}
    Feedback for input 0
    Inputs Values: [
    	[ [ 1.56 ], [ -0.5 ], [ 0.376 ] ],
    	[ [ 1.364 ], [ 0.34 ], [ -0.104 ] ]
    ]
    Value Statistics: {meanExponent=-0.3082318209533938, negative=2, min=-0.104, max=-0.104, mean=0.506, count=6.0, positive=4, stdDev=0.7388811361330951, zeros=0}
    Implemented Feedback: [ [ 1.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 1.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 1.0, 0.0, 0.0, 0.0 ], [ 0.0
```
...[skipping 505 bytes](etc/57.txt)...
```
    9999999998899, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.9999999999998899 ] ]
    Measured Statistics: {meanExponent=-3.692731311925336E-13, negative=0, min=0.9999999999998899, max=0.9999999999998899, mean=0.16666666666652497, count=36.0, positive=6, stdDev=0.37267799624964804, zeros=30}
    Feedback Error: [ [ -1.1013412404281553E-13, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, -2.3305801732931286E-12, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, -2.3305801732931286E-12, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, -1.1013412404281553E-13, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, -1.1013412404281553E-13, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, -1.1013412404281553E-13 ] ]
    Error Statistics: {meanExponent=-12.516230716189694, negative=6, min=-1.1013412404281553E-13, max=-1.1013412404281553E-13, mean=-1.4171380118770886E-13, count=36.0, positive=0, stdDev=5.319968968506581E-13, zeros=30}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 1.4171e-13 +- 5.3200e-13 [0.0000e+00 - 2.3306e-12] (36#)
    relativeTol: 4.2514e-13 +- 5.2336e-13 [5.5067e-14 - 1.1653e-12] (6#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=1.4171e-13 +- 5.3200e-13 [0.0000e+00 - 2.3306e-12] (36#), relativeTol=4.2514e-13 +- 5.2336e-13 [5.5067e-14 - 1.1653e-12] (6#)}
```



### Performance
Code from [LayerTestBase.java:192](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L192) executed in 0.00 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 0.2609 +- 0.6796 [0.1368 - 6.9592]
    Learning performance: 0.0027 +- 0.0030 [0.0000 - 0.0257]
    
```

### Function Plots
Code from [ActivationLayerTestBase.java:103](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/java/ActivationLayerTestBase.java#L103) executed in 0.00 seconds: 
```java
    return plot("Value Plot", plotData, x -> new double[]{x[0], x[1]});
```

Returns: 

![Result](etc/test.20.png)



Code from [ActivationLayerTestBase.java:107](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/java/ActivationLayerTestBase.java#L107) executed in 0.00 seconds: 
```java
    return plot("Derivative Plot", plotData, x -> new double[]{x[0], x[2]});
```

Returns: 

![Result](etc/test.21.png)



