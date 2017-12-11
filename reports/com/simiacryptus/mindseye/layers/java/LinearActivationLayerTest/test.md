# LinearActivationLayer
## LinearActivationLayerTest
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
      "class": "com.simiacryptus.mindseye.layers.java.LinearActivationLayer",
      "id": "e2a3bda5-e7e7-4c05-aeb3-4ede00001eb4",
      "isFrozen": false,
      "name": "LinearActivationLayer/e2a3bda5-e7e7-4c05-aeb3-4ede00001eb4",
      "weights": [
        1.0,
        0.0
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
    	[ [ -1.148 ], [ -1.94 ], [ -1.772 ] ],
    	[ [ -1.208 ], [ 0.832 ], [ 1.056 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ -1.148 ], [ -1.94 ], [ -1.772 ] ],
    	[ [ -1.208 ], [ 0.832 ], [ 1.056 ] ]
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
    	[ [ -1.148 ], [ -1.94 ], [ -1.772 ] ],
    	[ [ -1.208 ], [ 0.832 ], [ 1.056 ] ]
    ]
    Inputs Statistics: {meanExponent=0.10367691905280717, negative=4, min=1.056, max=1.056, mean=-0.6966666666666667, count=6.0, positive=2, stdDev=1.1955504543468205, zeros=0}
    Output: [
    	[ [ -1.148 ], [ -1.94 ], [ -1.772 ] ],
    	[ [ -1.208 ], [ 0.832 ], [ 1.056 ] ]
    ]
    Outputs Statistics: {meanExponent=0.10367691905280717, negative=4, min=1.056, max=1.056, mean=-0.6966666666666667, count=6.0, positive=2, stdDev=1.1955504543468205, zeros=0}
    Feedback for input 0
    Inputs Values: [
    	[ [ -1.148 ], [ -1.94 ], [ -1.772 ] ],
    	[ [ -1.208 ], [ 0.832 ], [ 1.056 ] ]
    ]
    Value Statistics: {meanExponent=0.10367691905280717, negative=4, min=1.056, max=1.056, mean=-0.6966666666666667, count=6.0, positive=2, stdDev=1.1955504543468205, zeros=0}
    Implemented Feedback: [ [ 1.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 1.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 1.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 1.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 1.0, 0.0 ], [ 0.0,
```
...[skipping 1793 bytes](etc/65.txt)...
```
    0.9999999999998899, 0.9999999999998899, 0.9999999999998899, 0.9999999999998899, 0.9999999999998899 ] ]
    Measured Statistics: {meanExponent=0.051838459526303025, negative=4, min=0.9999999999998899, max=0.9999999999998899, mean=0.15166666666658704, count=12.0, positive=8, stdDev=1.19763929832335, zeros=0}
    Gradient Error: [ [ 2.950972799453666E-13, 7.902567489281864E-13, -2.7489122089718876E-13, -5.00155472593633E-13, 1.1524114995609125E-13, -7.203126983768016E-13 ], [ -1.1013412404281553E-13, -1.1013412404281553E-13, -1.1013412404281553E-13, -1.1013412404281553E-13, -1.1013412404281553E-13, -1.1013412404281553E-13 ] ]
    Error Statistics: {meanExponent=-12.693611718226144, negative=9, min=-1.1013412404281553E-13, max=-1.1013412404281553E-13, mean=-7.963074644123935E-14, count=12.0, positive=3, stdDev=3.612621626974518E-13, zeros=0}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 8.3699e-14 +- 1.6948e-13 [0.0000e+00 - 7.9026e-13] (48#)
    relativeTol: 1.0341e-13 +- 1.0004e-13 [3.2517e-14 - 3.4106e-13] (18#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=8.3699e-14 +- 1.6948e-13 [0.0000e+00 - 7.9026e-13] (48#), relativeTol=1.0341e-13 +- 1.0004e-13 [3.2517e-14 - 3.4106e-13] (18#)}
```



### Performance
Code from [LayerTestBase.java:192](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L192) executed in 0.00 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 0.1724 +- 0.2045 [0.1054 - 2.0376]
    Learning performance: 0.0427 +- 0.0635 [0.0257 - 0.5985]
    
```

### Function Plots
Code from [ActivationLayerTestBase.java:103](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/java/ActivationLayerTestBase.java#L103) executed in 0.00 seconds: 
```java
    return plot("Value Plot", plotData, x -> new double[]{x[0], x[1]});
```

Returns: 

![Result](etc/test.24.png)



Code from [ActivationLayerTestBase.java:107](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/java/ActivationLayerTestBase.java#L107) executed in 0.00 seconds: 
```java
    return plot("Derivative Plot", plotData, x -> new double[]{x[0], x[2]});
```

Returns: 

![Result](etc/test.25.png)



