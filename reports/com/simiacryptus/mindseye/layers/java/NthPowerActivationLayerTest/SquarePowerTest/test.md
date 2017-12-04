# NthPowerActivationLayer
## SquarePowerTest
### Json Serialization
Code from [LayerTestBase.java:83](../../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L83) executed in 0.00 seconds: 
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
      "id": "a864e734-2f23-44db-97c1-504000002c60",
      "isFrozen": false,
      "name": "NthPowerActivationLayer/a864e734-2f23-44db-97c1-504000002c60",
      "power": 2.0
    }
```



### Example Input/Output Pair
Code from [LayerTestBase.java:120](../../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L120) executed in 0.00 seconds: 
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
    	[ [ -1.104 ], [ -0.976 ], [ 0.94 ] ],
    	[ [ 1.688 ], [ 1.248 ], [ 1.308 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 1.2188160000000001 ], [ 0.952576 ], [ 0.8835999999999999 ] ],
    	[ [ 2.849344 ], [ 1.557504 ], [ 1.7108640000000002 ] ]
    ]
```



### Batch Execution
Code from [LayerTestBase.java:138](../../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L138) executed in 0.00 seconds: 
```java
    BatchingTester batchingTester = getBatchingTester();
    return batchingTester==null?null:batchingTester.test(layer, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (120#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (120#)}
```



### Differential Validation
Code from [LayerTestBase.java:144](../../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.00 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ -1.104 ], [ -0.976 ], [ 0.94 ] ],
    	[ [ 1.688 ], [ 1.248 ], [ 1.308 ] ]
    ]
    Inputs Statistics: {meanExponent=0.07429025271397675, negative=2, min=1.308, max=1.308, mean=0.5173333333333333, count=6.0, positive=4, stdDev=1.123009448857053, zeros=0}
    Output: [
    	[ [ 1.2188160000000001 ], [ 0.952576 ], [ 0.8835999999999999 ] ],
    	[ [ 2.849344 ], [ 1.557504 ], [ 1.7108640000000002 ] ]
    ]
    Outputs Statistics: {meanExponent=0.1485805054279535, negative=0, min=1.7108640000000002, max=1.7108640000000002, mean=1.528784, count=6.0, positive=6, stdDev=0.6610868440863122, zeros=0}
    Feedback for input 0
    Inputs Values: [
    	[ [ -1.104 ], [ -0.976 ], [ 0.94 ] ],
    	[ [ 1.688 ], [ 1.248 ], [ 1.308 ] ]
    ]
    Value Statistics: {meanExponent=0.07429025271397675, negative=2, min=1.308, max=1.308, mean=0.5173333333333333, count=6.0, positive=4, stdDev=1.123009448857053, zeros=0}
    Implemented Feedback: [ [ -2.208, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 3.376, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, -1.952, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 2.496, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 1.88, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 2.616 ] ]
    Implemented Statistics: {meanExponent=0.3753202483779579, negative=2, min=2.616, max=2.616, mean=0.17244444444444446, count=36.0, positive=4, stdDev=0.9947121427395869, zeros=30}
    Measured Feedback: [ [ -2.2078999999997073, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 3.3760999999987718, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, -1.9519000000001174, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 2.496100000000112, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 1.880099999999052, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 2.616099999999122 ] ]
    Measured Statistics: {meanExponent=0.3753249226645563, negative=2, min=2.616099999999122, max=2.616099999999122, mean=0.17246111111103424, count=36.0, positive=4, stdDev=0.9947265900953747, zeros=30}
    Feedback Error: [ [ 1.0000000029286582E-4, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 9.999999877186028E-5, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 9.999999988252739E-5, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 1.0000000011212151E-4, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 9.999999905208057E-5, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 9.999999912180257E-5 ] ]
    Error Statistics: {meanExponent=-4.0000000020026345, negative=0, min=9.999999912180257E-5, max=9.999999912180257E-5, mean=1.6666666589812725E-5, count=36.0, positive=6, stdDev=3.726779945314586E-5, zeros=30}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 1.6667e-05 +- 3.7268e-05 [0.0000e+00 - 1.0000e-04] (36#)
    relativeTol: 2.1468e-05 +- 4.0168e-06 [1.4810e-05 - 2.6595e-05] (6#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=1.6667e-05 +- 3.7268e-05 [0.0000e+00 - 1.0000e-04] (36#), relativeTol=2.1468e-05 +- 4.0168e-06 [1.4810e-05 - 2.6595e-05] (6#)}
```



### Performance
Code from [LayerTestBase.java:149](../../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L149) executed in 0.00 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 0.1308 +- 0.0234 [0.1026 - 0.2337]
    Learning performance: 0.0011 +- 0.0024 [0.0000 - 0.0200]
    
```

### Function Plots
Code from [ActivationLayerTestBase.java:74](../../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/java/ActivationLayerTestBase.java#L74) executed in 0.00 seconds: 
```java
    return plot("Value Plot", plotData, x -> new double[]{x[0], x[1]});
```

Returns: 

![Result](etc/test.1.png)



Code from [ActivationLayerTestBase.java:78](../../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/java/ActivationLayerTestBase.java#L78) executed in 0.00 seconds: 
```java
    return plot("Derivative Plot", plotData, x -> new double[]{x[0], x[2]});
```

Returns: 

![Result](etc/test.2.png)



