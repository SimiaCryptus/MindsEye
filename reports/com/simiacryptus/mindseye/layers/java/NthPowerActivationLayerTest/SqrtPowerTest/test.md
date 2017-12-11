# NthPowerActivationLayer
## SqrtPowerTest
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
      "id": "e2a3bda5-e7e7-4c05-aeb3-4ede00001f10",
      "isFrozen": false,
      "name": "NthPowerActivationLayer/e2a3bda5-e7e7-4c05-aeb3-4ede00001f10",
      "power": 0.5
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
    	[ [ 1.152 ], [ -1.86 ], [ -0.028 ] ],
    	[ [ -1.792 ], [ 1.684 ], [ -1.304 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 1.0733126291998991 ], [ 0.0 ], [ 0.0 ] ],
    	[ [ 0.0 ], [ 1.2976902558006667 ], [ 0.0 ] ]
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
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (120#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (62#)}
```



### Differential Validation
Code from [LayerTestBase.java:186](../../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L186) executed in 0.00 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ 1.152 ], [ -1.86 ], [ -0.028 ] ],
    	[ [ -1.792 ], [ 1.684 ], [ -1.304 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.10448647691117213, negative=4, min=-1.304, max=-1.304, mean=-0.35800000000000004, count=6.0, positive=2, stdDev=1.4003594776580286, zeros=0}
    Output: [
    	[ [ 1.0733126291998991 ], [ 0.0 ], [ 0.0 ] ],
    	[ [ 0.0 ], [ 1.2976902558006667 ], [ 0.0 ] ]
    ]
    Outputs Statistics: {meanExponent=0.07194864156270597, negative=0, min=0.0, max=0.0, mean=0.39516714750009435, count=6.0, positive=2, stdDev=0.5625918522368639, zeros=4}
    Feedback for input 0
    Inputs Values: [
    	[ [ 1.152 ], [ -1.86 ], [ -0.028 ] ],
    	[ [ -1.792 ], [ 1.684 ], [ -1.304 ] ]
    ]
    Value Statistics: {meanExponent=-0.10448647691117213, negative=4, min=-1.304, max=-1.304, mean=-0.35800000000000004, count=6.0, positive=2, stdDev=1.4003594776580286, zeros=0}
    Implemented Feedback: [ [ 0.4658474953124562, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.38529995718547116, 0.
```
...[skipping 286 bytes](etc/76.txt)...
```
    658373862120868, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.3852942373461232, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ] ]
    Measured Statistics: {meanExponent=-0.3729865730800733, negative=0, min=0.0, max=0.0, mean=0.023642545098839167, count=36.0, positive=2, stdDev=0.09794176301507165, zeros=34}
    Feedback Error: [ [ -1.0109100369359858E-5, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, -5.719839347939626E-6, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ] ]
    Error Statistics: {meanExponent=-5.118951830237771, negative=2, min=0.0, max=0.0, mean=-4.396927699249857E-7, count=36.0, positive=0, stdDev=1.8852541981178484E-6, zeros=34}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 4.3969e-07 +- 1.8853e-06 [0.0000e+00 - 1.0109e-05] (36#)
    relativeTol: 9.1365e-06 +- 1.7139e-06 [7.4226e-06 - 1.0850e-05] (2#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=4.3969e-07 +- 1.8853e-06 [0.0000e+00 - 1.0109e-05] (36#), relativeTol=9.1365e-06 +- 1.7139e-06 [7.4226e-06 - 1.0850e-05] (2#)}
```



### Performance
Code from [LayerTestBase.java:192](../../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L192) executed in 0.00 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 0.1661 +- 0.0395 [0.1140 - 0.3847]
    Learning performance: 0.0028 +- 0.0156 [0.0000 - 0.1567]
    
```

### Function Plots
Code from [ActivationLayerTestBase.java:103](../../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/java/ActivationLayerTestBase.java#L103) executed in 0.00 seconds: 
```java
    return plot("Value Plot", plotData, x -> new double[]{x[0], x[1]});
```

Returns: 

![Result](etc/test.38.png)



Code from [ActivationLayerTestBase.java:107](../../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/java/ActivationLayerTestBase.java#L107) executed in 0.07 seconds: 
```java
    return plot("Derivative Plot", plotData, x -> new double[]{x[0], x[2]});
```

Returns: 

![Result](etc/test.39.png)



