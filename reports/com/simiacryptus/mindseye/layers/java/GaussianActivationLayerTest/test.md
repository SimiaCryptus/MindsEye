# GaussianActivationLayer
## GaussianActivationLayerTest
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
      "class": "com.simiacryptus.mindseye.layers.java.GaussianActivationLayer",
      "id": "e2a3bda5-e7e7-4c05-aeb3-4ede00001e55",
      "isFrozen": true,
      "name": "GaussianActivationLayer/e2a3bda5-e7e7-4c05-aeb3-4ede00001e55",
      "mean": 0.0,
      "stddev": 1.0
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
    	[ [ -0.26 ], [ 1.896 ], [ -0.092 ] ],
    	[ [ -1.376 ], [ 0.96 ], [ 0.704 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 0.3856833691918161 ], [ 0.06611586583241554 ], [ 0.3972575241295185 ] ],
    	[ [ 0.15479919266156578 ], [ 0.2516443410981171 ], [ 0.31137835421032156 ] ]
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
    	[ [ -0.26 ], [ 1.896 ], [ -0.092 ] ],
    	[ [ -1.376 ], [ 0.96 ], [ 0.704 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.22915636093340105, negative=3, min=0.704, max=0.704, mean=0.3053333333333333, count=6.0, positive=3, stdDev=1.0345773157295794, zeros=0}
    Output: [
    	[ [ 0.3856833691918161 ], [ 0.06611586583241554 ], [ 0.3972575241295185 ] ],
    	[ [ 0.15479919266156578 ], [ 0.2516443410981171 ], [ 0.31137835421032156 ] ]
    ]
    Outputs Statistics: {meanExponent=-0.6517578312758957, negative=0, min=0.31137835421032156, max=0.31137835421032156, mean=0.2611464411872924, count=6.0, positive=6, stdDev=0.11971902586251303, zeros=0}
    Feedback for input 0
    Inputs Values: [
    	[ [ -0.26 ], [ 1.896 ], [ -0.092 ] ],
    	[ [ -1.376 ], [ 0.96 ], [ 0.704 ] ]
    ]
    Value Statistics: {meanExponent=-0.22915636093340105, negative=3, min=0.704, max=0.704, mean=0.3053333333333333, count=6.0, positive=3, stdDev=1.0345773157295794, zeros=0}
    Implemented Feedback: [ [ 0.10027767598987221, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.213003689102314
```
...[skipping 672 bytes](etc/56.txt)...
```
    0.0, 0.0, 0.0, 0.0365279972808219, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, -0.2192182131621756 ] ]
    Measured Statistics: {meanExponent=-0.8809659043051038, negative=3, min=-0.2192182131621756, max=-0.2192182131621756, mean=-0.006565182622692474, count=36.0, positive=3, stdDev=0.07018573803162322, zeros=30}
    Feedback Error: [ [ -1.7981048665086097E-5, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 6.914301137306289E-6, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 8.577801057119139E-6, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, -9.856088391269058E-7, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, -1.9694939093803376E-5, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, -7.85179810924852E-6 ] ]
    Error Statistics: {meanExponent=-5.131505394614993, negative=4, min=-7.85179810924852E-6, max=-7.85179810924852E-6, mean=-8.617025698010964E-7, count=36.0, positive=2, stdDev=4.9116810854131384E-6, zeros=30}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 1.7224e-06 +- 4.6798e-06 [0.0000e+00 - 1.9695e-05] (36#)
    relativeTol: 7.1595e-05 +- 9.2809e-05 [2.0399e-06 - 2.6951e-04] (6#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=1.7224e-06 +- 4.6798e-06 [0.0000e+00 - 1.9695e-05] (36#), relativeTol=7.1595e-05 +- 9.2809e-05 [2.0399e-06 - 2.6951e-04] (6#)}
```



### Performance
Code from [LayerTestBase.java:192](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L192) executed in 0.00 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 0.1904 +- 0.0627 [0.1254 - 0.5956]
    Learning performance: 0.0027 +- 0.0041 [0.0000 - 0.0370]
    
```

### Function Plots
Code from [ActivationLayerTestBase.java:103](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/java/ActivationLayerTestBase.java#L103) executed in 0.00 seconds: 
```java
    return plot("Value Plot", plotData, x -> new double[]{x[0], x[1]});
```

Returns: 

![Result](etc/test.18.png)



Code from [ActivationLayerTestBase.java:107](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/java/ActivationLayerTestBase.java#L107) executed in 0.01 seconds: 
```java
    return plot("Derivative Plot", plotData, x -> new double[]{x[0], x[2]});
```

Returns: 

![Result](etc/test.19.png)



