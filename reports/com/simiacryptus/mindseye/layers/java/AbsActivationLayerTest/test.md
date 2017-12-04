# AbsActivationLayer
## AbsActivationLayerTest
### Json Serialization
Code from [LayerTestBase.java:83](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L83) executed in 0.00 seconds: 
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
      "id": "a864e734-2f23-44db-97c1-504000002b74",
      "isFrozen": true,
      "name": "AbsActivationLayer/a864e734-2f23-44db-97c1-504000002b74"
    }
```



### Example Input/Output Pair
Code from [LayerTestBase.java:120](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L120) executed in 0.00 seconds: 
```java
    SimpleEval eval = SimpleEval.run(layer, inputPrototype);
    return String.format("--------------------\nInput: \n[%s]\n--------------------\nOutput: \n%s",
      Arrays.stream(inputPrototype).map(t->t.prettyPrint()).reduce((a,b)->a+",\n"+b).get(),
      eval.getOutput().prettyPrint());
```

Returns: 

```
    --------------------
    Input: 
    [[
    	[ [ 0.124 ], [ 0.44 ], [ -0.356 ] ],
    	[ [ -0.316 ], [ -1.42 ], [ 1.028 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 0.124 ], [ 0.44 ], [ 0.356 ] ],
    	[ [ 0.316 ], [ 1.42 ], [ 1.028 ] ]
    ]
```



### Batch Execution
Code from [LayerTestBase.java:138](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L138) executed in 0.00 seconds: 
```java
    BatchingTester batchingTester = getBatchingTester();
    return batchingTester==null?null:batchingTester.test(layer, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (120#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (120#)}
```



### Differential Validation
Code from [LayerTestBase.java:144](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.00 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ 0.124 ], [ 0.44 ], [ -0.356 ] ],
    	[ [ -0.316 ], [ -1.42 ], [ 1.028 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.34128451645299746, negative=3, min=1.028, max=1.028, mean=-0.08333333333333333, count=6.0, positive=3, stdDev=0.7601628480500449, zeros=0}
    Output: [
    	[ [ 0.124 ], [ 0.44 ], [ 0.356 ] ],
    	[ [ 0.316 ], [ 1.42 ], [ 1.028 ] ]
    ]
    Outputs Statistics: {meanExponent=-0.34128451645299746, negative=0, min=1.028, max=1.028, mean=0.614, count=6.0, positive=6, stdDev=0.45584646538061474, zeros=0}
    Feedback for input 0
    Inputs Values: [
    	[ [ 0.124 ], [ 0.44 ], [ -0.356 ] ],
    	[ [ -0.316 ], [ -1.42 ], [ 1.028 ] ]
    ]
    Value Statistics: {meanExponent=-0.34128451645299746, negative=3, min=1.028, max=1.028, mean=-0.08333333333333333, count=6.0, positive=3, stdDev=0.7601628480500449, zeros=0}
    Implemented Feedback: [ [ 1.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, -1.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 1.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, -1.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, -1.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 1.0 ] ]
    Implemented Statistics: {meanExponent=0.0, negative=3, min=1.0, max=1.0, mean=0.0, count=36.0, positive=3, stdDev=0.408248290463863, zeros=30}
    Measured Feedback: [ [ 1.0000000000000286, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, -0.9999999999998899, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.9999999999998899, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, -0.9999999999998899, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, -0.9999999999998899, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.9999999999998899 ] ]
    Measured Statistics: {meanExponent=-3.7785564564448525E-14, negative=3, min=0.9999999999998899, max=0.9999999999998899, mean=3.854941057726238E-15, count=36.0, positive=3, stdDev=0.4082482904638275, zeros=30}
    Feedback Error: [ [ 2.864375403532904E-14, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 1.1013412404281553E-13, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, -1.1013412404281553E-13, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 1.1013412404281553E-13, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 1.1013412404281553E-13, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, -1.1013412404281553E-13 ] ]
    Error Statistics: {meanExponent=-13.055560092401983, negative=2, min=-1.1013412404281553E-13, max=-1.1013412404281553E-13, mean=3.854941057726238E-15, count=36.0, positive=4, stdDev=4.114105495410483E-14, zeros=30}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 1.6092e-14 +- 3.8059e-14 [0.0000e+00 - 1.1013e-13] (36#)
    relativeTol: 4.8276e-14 +- 1.5185e-14 [1.4322e-14 - 5.5067e-14] (6#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=1.6092e-14 +- 3.8059e-14 [0.0000e+00 - 1.1013e-13] (36#), relativeTol=4.8276e-14 +- 1.5185e-14 [1.4322e-14 - 5.5067e-14] (6#)}
```



### Performance
Code from [LayerTestBase.java:149](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L149) executed in 0.00 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 0.1926 +- 0.0486 [0.1510 - 0.4417]
    Learning performance: 0.0025 +- 0.0019 [0.0000 - 0.0142]
    
```

### Function Plots
Code from [ActivationLayerTestBase.java:74](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/java/ActivationLayerTestBase.java#L74) executed in 0.09 seconds: 
```java
    return plot("Value Plot", plotData, x -> new double[]{x[0], x[1]});
```

Returns: 

![Result](etc/test.1.png)



Code from [ActivationLayerTestBase.java:78](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/java/ActivationLayerTestBase.java#L78) executed in 0.01 seconds: 
```java
    return plot("Derivative Plot", plotData, x -> new double[]{x[0], x[2]});
```

Returns: 

![Result](etc/test.2.png)



