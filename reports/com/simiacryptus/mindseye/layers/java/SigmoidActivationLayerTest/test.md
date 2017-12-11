# SigmoidActivationLayer
## SigmoidActivationLayerTest
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
      "class": "com.simiacryptus.mindseye.layers.java.SigmoidActivationLayer",
      "id": "e2a3bda5-e7e7-4c05-aeb3-4ede0000361c",
      "isFrozen": true,
      "name": "SigmoidActivationLayer/e2a3bda5-e7e7-4c05-aeb3-4ede0000361c",
      "balanced": true
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
    	[ [ 1.356 ], [ 0.468 ], [ -1.068 ] ],
    	[ [ 0.328 ], [ -0.112 ], [ -1.6 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 0.5902176475395715 ], [ 0.22982054821431697 ], [ -0.4884327702070025 ] ],
    	[ [ 0.162545333218451 ], [ -0.05594153467114671 ], [ -0.664036770267849 ] ]
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
    	[ [ 1.356 ], [ 0.468 ], [ -1.068 ] ],
    	[ [ 0.328 ], [ -0.112 ], [ -1.6 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.23328522594408466, negative=3, min=-1.6, max=-1.6, mean=-0.10466666666666669, count=6.0, positive=3, stdDev=0.9842748035426331, zeros=0}
    Output: [
    	[ [ 0.5902176475395715 ], [ 0.22982054821431697 ], [ -0.4884327702070025 ] ],
    	[ [ 0.162545333218451 ], [ -0.05594153467114671 ], [ -0.664036770267849 ] ]
    ]
    Outputs Statistics: {meanExponent=-0.5663155247697542, negative=3, min=-0.664036770267849, max=-0.664036770267849, mean=-0.037637924362276455, count=6.0, positive=3, stdDev=0.42851225569388623, zeros=0}
    Feedback for input 0
    Inputs Values: [
    	[ [ 1.356 ], [ 0.468 ], [ -1.068 ] ],
    	[ [ 0.328 ], [ -0.112 ], [ -1.6 ] ]
    ]
    Value Statistics: {meanExponent=-0.23328522594408466, negative=3, min=-1.6, max=-1.6, mean=-0.10466666666666669, count=6.0, positive=3, stdDev=0.9842748035426331, zeros=0}
    Implemented Feedback: [ [ 0.32582156426642706, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.48678950732445
```
...[skipping 651 bytes](etc/84.txt)...
```
    [ 0.0, 0.0, 0.0, 0.0, 0.38072601213023916, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.2795368647701135 ] ]
    Measured Statistics: {meanExponent=-0.3999391675380715, negative=0, min=0.2795368647701135, max=0.2795368647701135, mean=0.06791341272503902, count=36.0, positive=6, stdDev=0.15571421914802006, zeros=30}
    Feedback Error: [ [ -9.615270702534318E-6, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, -3.956642628366591E-6, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, -5.442383343379209E-6, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 1.3937482939985202E-6, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 9.29763628237934E-6, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 9.280903991604195E-6 ] ]
    Error Statistics: {meanExponent=-5.2672958913662855, negative=3, min=9.280903991604195E-6, max=9.280903991604195E-6, mean=2.661088593616492E-8, count=36.0, positive=3, stdDev=2.944987496024757E-6, zeros=30}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 1.0830e-06 +- 2.7388e-06 [0.0000e+00 - 9.6153e-06] (36#)
    relativeTol: 9.1292e-06 +- 5.6840e-06 [1.3981e-06 - 1.6601e-05] (6#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=1.0830e-06 +- 2.7388e-06 [0.0000e+00 - 9.6153e-06] (36#), relativeTol=9.1292e-06 +- 5.6840e-06 [1.3981e-06 - 1.6601e-05] (6#)}
```



### Performance
Code from [LayerTestBase.java:192](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L192) executed in 0.00 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 0.1370 +- 0.0348 [0.0855 - 0.3049]
    Learning performance: 0.0012 +- 0.0026 [0.0000 - 0.0228]
    
```

### Function Plots
Code from [ActivationLayerTestBase.java:103](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/java/ActivationLayerTestBase.java#L103) executed in 0.00 seconds: 
```java
    return plot("Value Plot", plotData, x -> new double[]{x[0], x[1]});
```

Returns: 

![Result](etc/test.46.png)



Code from [ActivationLayerTestBase.java:107](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/java/ActivationLayerTestBase.java#L107) executed in 0.00 seconds: 
```java
    return plot("Derivative Plot", plotData, x -> new double[]{x[0], x[2]});
```

Returns: 

![Result](etc/test.47.png)



