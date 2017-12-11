# NthPowerActivationLayer
## InvSqrtPowerTest
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
      "id": "e2a3bda5-e7e7-4c05-aeb3-4ede00001f02",
      "isFrozen": false,
      "name": "NthPowerActivationLayer/e2a3bda5-e7e7-4c05-aeb3-4ede00001f02",
      "power": -0.5
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
    	[ [ 0.784 ], [ 1.848 ], [ 0.3 ] ],
    	[ [ 1.028 ], [ 1.364 ], [ 0.332 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 1.1293848786315641 ], [ 0.7356123579206246 ], [ 1.8257418583505538 ] ],
    	[ [ 0.9862873039405895 ], [ 0.8562346815634271 ], [ 1.7355253362515581 ] ]
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
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (120#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (64#)}
```



### Differential Validation
Code from [LayerTestBase.java:186](../../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L186) executed in 0.00 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ 0.784 ], [ 1.848 ], [ 0.3 ] ],
    	[ [ 1.028 ], [ 1.364 ], [ 0.332 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.1156525245046763, negative=0, min=0.332, max=0.332, mean=0.9426666666666667, count=6.0, positive=6, stdDev=0.5500820140871925, zeros=0}
    Output: [
    	[ [ 1.1293848786315641 ], [ 0.7356123579206246 ], [ 1.8257418583505538 ] ],
    	[ [ 0.9862873039405895 ], [ 0.8562346815634271 ], [ 1.7355253362515581 ] ]
    ]
    Outputs Statistics: {meanExponent=0.05782626225233815, negative=0, min=1.7355253362515581, max=1.7355253362515581, mean=1.2114644027763861, count=6.0, positive=6, stdDev=0.42072194934658397, zeros=0}
    Feedback for input 0
    Inputs Values: [
    	[ [ 0.784 ], [ 1.848 ], [ 0.3 ] ],
    	[ [ 1.028 ], [ 1.364 ], [ 0.332 ] ]
    ]
    Value Statistics: {meanExponent=-0.1156525245046763, negative=0, min=0.332, max=0.332, mean=0.9426666666666667, count=6.0, positive=6, stdDev=0.5500820140871925, zeros=0}
    Implemented Feedback: [ [ -0.720270968515028, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, -0.47971172370651244, 0.0, 0.0,
```
...[skipping 642 bytes](etc/74.txt)...
```
    , [ 0.0, 0.0, 0.0, 0.0, -3.042142582727614, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, -2.613152670463137 ] ]
    Measured Statistics: {meanExponent=-0.12760477281684798, negative=6, min=-2.613152670463137, max=-2.613152670463137, mean=-0.20466797375557405, count=36.0, positive=0, stdDev=0.6553643660437771, zeros=30}
    Feedback Error: [ [ 6.889614880078998E-5, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 3.4995587518460436E-5, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 8.07712357175916E-6, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 1.7257139362458762E-5, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 7.605145233093857E-4, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 5.903058193301725E-4 ] ]
    Error Statistics: {meanExponent=-4.13689694245175, negative=0, min=5.903058193301725E-4, max=5.903058193301725E-4, mean=4.1112398385917404E-5, count=36.0, positive=6, stdDev=1.5566429913541817E-4, zeros=30}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 4.1112e-05 +- 1.5566e-04 [0.0000e+00 - 7.6051e-04] (36#)
    relativeTol: 6.1668e-05 +- 4.1519e-05 [2.0292e-05 - 1.2498e-04] (6#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=4.1112e-05 +- 1.5566e-04 [0.0000e+00 - 7.6051e-04] (36#), relativeTol=6.1668e-05 +- 4.1519e-05 [2.0292e-05 - 1.2498e-04] (6#)}
```



### Performance
Code from [LayerTestBase.java:192](../../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L192) executed in 0.00 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 0.1448 +- 0.0334 [0.0997 - 0.3477]
    Learning performance: 0.0017 +- 0.0021 [0.0000 - 0.0142]
    
```

### Function Plots
Code from [ActivationLayerTestBase.java:103](../../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/java/ActivationLayerTestBase.java#L103) executed in 0.00 seconds: 
```java
    return plot("Value Plot", plotData, x -> new double[]{x[0], x[1]});
```

Returns: 

![Result](etc/test.34.png)



Code from [ActivationLayerTestBase.java:107](../../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/java/ActivationLayerTestBase.java#L107) executed in 0.00 seconds: 
```java
    return plot("Derivative Plot", plotData, x -> new double[]{x[0], x[2]});
```

Returns: 

![Result](etc/test.35.png)



