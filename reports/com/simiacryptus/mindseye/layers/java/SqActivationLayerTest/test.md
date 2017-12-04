# SqActivationLayer
## SqActivationLayerTest
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
      "class": "com.simiacryptus.mindseye.layers.java.SqActivationLayer",
      "id": "a864e734-2f23-44db-97c1-504000002c9a",
      "isFrozen": true,
      "name": "SqActivationLayer/a864e734-2f23-44db-97c1-504000002c9a"
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
    	[ [ -0.788 ], [ -1.84 ], [ 1.888 ] ],
    	[ [ -1.348 ], [ -1.3 ], [ -1.092 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 0.620944 ], [ 3.3856 ], [ 3.5645439999999997 ] ],
    	[ [ 1.8171040000000003 ], [ 1.6900000000000002 ], [ 1.1924640000000002 ] ]
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
    	[ [ -0.788 ], [ -1.84 ], [ 1.888 ] ],
    	[ [ -1.348 ], [ -1.3 ], [ -1.092 ] ]
    ]
    Inputs Statistics: {meanExponent=0.11986698555599971, negative=5, min=-1.092, max=-1.092, mean=-0.7466666666666667, count=6.0, positive=1, stdDev=1.2196713582855927, zeros=0}
    Output: [
    	[ [ 0.620944 ], [ 3.3856 ], [ 3.5645439999999997 ] ],
    	[ [ 1.8171040000000003 ], [ 1.6900000000000002 ], [ 1.1924640000000002 ] ]
    ]
    Outputs Statistics: {meanExponent=0.23973397111199943, negative=0, min=1.1924640000000002, max=1.1924640000000002, mean=2.045109333333334, count=6.0, positive=6, stdDev=1.0831233359642016, zeros=0}
    Feedback for input 0
    Inputs Values: [
    	[ [ -0.788 ], [ -1.84 ], [ 1.888 ] ],
    	[ [ -1.348 ], [ -1.3 ], [ -1.092 ] ]
    ]
    Value Statistics: {meanExponent=0.11986698555599971, negative=5, min=-1.092, max=-1.092, mean=-0.7466666666666667, count=6.0, positive=1, stdDev=1.2196713582855927, zeros=0}
    Implemented Feedback: [ [ -1.576, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, -2.696, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, -3.68, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, -2.6, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 3.776, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, -2.184 ] ]
    Implemented Statistics: {meanExponent=0.4208969812199809, negative=5, min=-2.184, max=-2.184, mean=-0.2488888888888889, count=36.0, positive=1, stdDev=1.1408157358705555, zeros=30}
    Measured Feedback: [ [ -1.5758999999992973, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, -2.6959000000004174, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, -3.679899999999847, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, -2.5998999999998773, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 3.7760999999969513, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, -2.183899999999017 ] ]
    Measured Statistics: {meanExponent=0.42088355508411784, negative=5, min=-2.183899999999017, max=-2.183899999999017, mean=-0.24887222222226402, count=36.0, positive=1, stdDev=1.1407975557111159, zeros=30}
    Feedback Error: [ [ 1.0000000070276016E-4, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 9.999999958276717E-5, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 1.0000000015297772E-4, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 1.0000000012277965E-4, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 9.99999969515386E-5, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 1.0000000098298045E-4 ] ]
    Error Statistics: {meanExponent=-4.0000000010887735, negative=0, min=1.0000000098298045E-4, max=1.0000000098298045E-4, mean=1.6666666624883437E-5, count=36.0, positive=6, stdDev=3.7267799531566354E-5, zeros=30}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 1.6667e-05 +- 3.7268e-05 [0.0000e+00 - 1.0000e-04] (36#)
    relativeTol: 1.9871e-05 +- 6.2624e-06 [1.3241e-05 - 3.1727e-05] (6#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=1.6667e-05 +- 3.7268e-05 [0.0000e+00 - 1.0000e-04] (36#), relativeTol=1.9871e-05 +- 6.2624e-06 [1.3241e-05 - 3.1727e-05] (6#)}
```



### Performance
Code from [LayerTestBase.java:149](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L149) executed in 0.00 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 0.1342 +- 0.0268 [0.1083 - 0.3021]
    Learning performance: 0.0009 +- 0.0019 [0.0000 - 0.0142]
    
```

### Function Plots
Code from [ActivationLayerTestBase.java:74](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/java/ActivationLayerTestBase.java#L74) executed in 0.00 seconds: 
```java
    return plot("Value Plot", plotData, x -> new double[]{x[0], x[1]});
```

Returns: 

![Result](etc/test.1.png)



Code from [ActivationLayerTestBase.java:78](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/java/ActivationLayerTestBase.java#L78) executed in 0.00 seconds: 
```java
    return plot("Derivative Plot", plotData, x -> new double[]{x[0], x[2]});
```

Returns: 

![Result](etc/test.2.png)



