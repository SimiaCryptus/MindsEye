# NthPowerActivationLayer
## InvPowerTest
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
      "id": "e2a3bda5-e7e7-4c05-aeb3-4ede00001efb",
      "isFrozen": false,
      "name": "NthPowerActivationLayer/e2a3bda5-e7e7-4c05-aeb3-4ede00001efb",
      "power": -1.0
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
    	[ [ 1.924 ], [ 0.608 ], [ 1.944 ] ],
    	[ [ 1.036 ], [ -0.988 ], [ -1.568 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 0.5197505197505198 ], [ 1.6447368421052633 ], [ 0.51440329218107 ] ],
    	[ [ 0.9652509652509652 ], [ -1.0121457489878543 ], [ -0.6377551020408163 ] ]
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
    	[ [ 1.924 ], [ 0.608 ], [ 1.944 ] ],
    	[ [ 1.036 ], [ -0.988 ], [ -1.568 ] ]
    ]
    Inputs Statistics: {meanExponent=0.09371127765167447, negative=2, min=-1.568, max=-1.568, mean=0.49266666666666664, count=6.0, positive=4, stdDev=1.347981041739419, zeros=0}
    Output: [
    	[ [ 0.5197505197505198 ], [ 1.6447368421052633 ], [ 0.51440329218107 ] ],
    	[ [ 0.9652509652509652 ], [ -1.0121457489878543 ], [ -0.6377551020408163 ] ]
    ]
    Outputs Statistics: {meanExponent=-0.09371127765167447, negative=2, min=-0.6377551020408163, max=-0.6377551020408163, mean=0.3323734613765247, count=6.0, positive=4, stdDev=0.9073733094205472, zeros=0}
    Feedback for input 0
    Inputs Values: [
    	[ [ 1.924 ], [ 0.608 ], [ 1.944 ] ],
    	[ [ 1.036 ], [ -0.988 ], [ -1.568 ] ]
    ]
    Value Statistics: {meanExponent=0.09371127765167447, negative=2, min=-1.568, max=-1.568, mean=0.49266666666666664, count=6.0, positive=4, stdDev=1.347981041739419, zeros=0}
    Implemented Feedback: [ [ -0.27014060278093543, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, -0.931709425917
```
...[skipping 666 bytes](etc/73.txt)...
```
    .0, 0.0, 0.0, -0.26459713604265467, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, -0.406757511346445 ] ]
    Measured Statistics: {meanExponent=-0.18743698807180076, negative=6, min=-0.406757511346445, max=-0.406757511346445, mean=-0.15562105146850932, count=36.0, positive=0, stdDev=0.49073837939886766, zeros=30}
    Feedback Error: [ [ 1.4039841925050922E-5, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 8.992466249935216E-5, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 4.4485434620700204E-4, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, -1.0369865643378162E-4, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 1.3610964068599962E-5, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, -2.594116735299412E-5 ] ]
    Error Statistics: {meanExponent=-4.281148265828208, negative=2, min=-2.594116735299412E-5, max=-2.594116735299412E-5, mean=1.2021944192034148E-5, count=36.0, positive=4, stdDev=7.684536929611318E-5, zeros=30}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 1.9224e-05 +- 7.5367e-05 [0.0000e+00 - 4.4485e-04] (36#)
    relativeTol: 4.4116e-05 +- 1.9700e-05 [2.5720e-05 - 8.2230e-05] (6#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=1.9224e-05 +- 7.5367e-05 [0.0000e+00 - 4.4485e-04] (36#), relativeTol=4.4116e-05 +- 1.9700e-05 [2.5720e-05 - 8.2230e-05] (6#)}
```



### Performance
Code from [LayerTestBase.java:192](../../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L192) executed in 0.00 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 0.1951 +- 0.0408 [0.1311 - 0.4075]
    Learning performance: 0.0015 +- 0.0016 [0.0000 - 0.0086]
    
```

### Function Plots
Code from [ActivationLayerTestBase.java:103](../../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/java/ActivationLayerTestBase.java#L103) executed in 0.00 seconds: 
```java
    return plot("Value Plot", plotData, x -> new double[]{x[0], x[1]});
```

Returns: 

![Result](etc/test.32.png)



Code from [ActivationLayerTestBase.java:107](../../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/java/ActivationLayerTestBase.java#L107) executed in 0.00 seconds: 
```java
    return plot("Derivative Plot", plotData, x -> new double[]{x[0], x[2]});
```

Returns: 

![Result](etc/test.33.png)



