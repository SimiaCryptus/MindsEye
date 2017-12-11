# NthPowerActivationLayer
## NthPowerTest
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
      "id": "e2a3bda5-e7e7-4c05-aeb3-4ede00001f09",
      "isFrozen": false,
      "name": "NthPowerActivationLayer/e2a3bda5-e7e7-4c05-aeb3-4ede00001f09",
      "power": 2.5
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
    	[ [ -0.724 ], [ 0.228 ], [ -0.896 ] ],
    	[ [ 1.268 ], [ 0.452 ], [ 0.692 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 0.0 ], [ 0.02482201978824447 ], [ 0.0 ] ],
    	[ [ 1.8104983521090452 ], [ 0.13735551039558624 ], [ 0.3983503803226902 ] ]
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
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (120#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (56#)}
```



### Differential Validation
Code from [LayerTestBase.java:186](../../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L186) executed in 0.00 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ -0.724 ], [ 0.228 ], [ -0.896 ] ],
    	[ [ 1.268 ], [ 0.452 ], [ 0.692 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.20527579905440338, negative=2, min=0.692, max=0.692, mean=0.17, count=6.0, positive=4, stdDev=0.7633880620846342, zeros=0}
    Output: [
    	[ [ 0.0 ], [ 0.02482201978824447 ], [ 0.0 ] ],
    	[ [ 1.8104983521090452 ], [ 0.13735551039558624 ], [ 0.3983503803226902 ] ]
    ]
    Outputs Statistics: {meanExponent=-0.6523133563660578, negative=0, min=0.3983503803226902, max=0.3983503803226902, mean=0.39517104376926104, count=6.0, positive=4, stdDev=0.6479594964698896, zeros=2}
    Feedback for input 0
    Inputs Values: [
    	[ [ -0.724 ], [ 0.228 ], [ -0.896 ] ],
    	[ [ 1.268 ], [ 0.452 ], [ 0.692 ] ]
    ]
    Value Statistics: {meanExponent=-0.20527579905440338, negative=2, min=0.692, max=0.692, mean=0.17, count=6.0, positive=4, stdDev=0.7633880620846342, zeros=0}
    Implemented Feedback: [ [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 3.5695945428017453, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.27217126960794374, 0.0, 0.0, 0.0 ], 
```
...[skipping 493 bytes](etc/75.txt)...
```
    0.0, 0.0, 0.0, 0.7598357440619097, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 1.4392830750553642 ] ]
    Measured Statistics: {meanExponent=0.006623910153671801, negative=0, min=1.4392830750553642, max=1.4392830750553642, mean=0.16781070294980918, count=36.0, positive=4, stdDev=0.6336128576395895, zeros=32}
    Feedback Error: [ [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 2.1113809907191694E-4, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 8.953656709603797E-5, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 1.2606267039272367E-4, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 1.5597851385340533E-4 ] ]
    Error Statistics: {meanExponent=-3.8574454182427096, negative=0, min=1.5597851385340533E-4, max=1.5597851385340533E-4, mean=1.618655140039122E-5, count=36.0, positive=4, stdDev=4.8127442885460404E-5, zeros=32}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 1.6187e-05 +- 4.8127e-05 [0.0000e+00 - 2.1114e-04] (36#)
    relativeTol: 8.2796e-05 +- 5.0793e-05 [2.9574e-05 - 1.6446e-04] (4#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=1.6187e-05 +- 4.8127e-05 [0.0000e+00 - 2.1114e-04] (36#), relativeTol=8.2796e-05 +- 5.0793e-05 [2.9574e-05 - 1.6446e-04] (4#)}
```



### Performance
Code from [LayerTestBase.java:192](../../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L192) executed in 0.00 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 0.1557 +- 0.0837 [0.1197 - 0.7780]
    Learning performance: 0.0014 +- 0.0023 [0.0000 - 0.0171]
    
```

### Function Plots
Code from [ActivationLayerTestBase.java:103](../../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/java/ActivationLayerTestBase.java#L103) executed in 0.00 seconds: 
```java
    return plot("Value Plot", plotData, x -> new double[]{x[0], x[1]});
```

Returns: 

![Result](etc/test.36.png)



Code from [ActivationLayerTestBase.java:107](../../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/java/ActivationLayerTestBase.java#L107) executed in 0.00 seconds: 
```java
    return plot("Derivative Plot", plotData, x -> new double[]{x[0], x[2]});
```

Returns: 

![Result](etc/test.37.png)



