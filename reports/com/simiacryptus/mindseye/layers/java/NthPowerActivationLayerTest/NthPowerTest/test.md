# NthPowerActivationLayer
## NthPowerTest
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
      "id": "a864e734-2f23-44db-97c1-504000002c52",
      "isFrozen": false,
      "name": "NthPowerActivationLayer/a864e734-2f23-44db-97c1-504000002c52",
      "power": 2.5
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
    	[ [ -0.732 ], [ 0.824 ], [ -1.688 ] ],
    	[ [ -1.528 ], [ -1.72 ], [ 0.884 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 0.0 ], [ 0.6163367007299045 ], [ 0.0 ] ],
    	[ [ 0.0 ], [ 0.0 ], [ 0.7347348884212754 ] ]
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
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (120#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (62#)}
```



### Differential Validation
Code from [LayerTestBase.java:144](../../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.00 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ -0.732 ], [ 0.824 ], [ -1.688 ] ],
    	[ [ -1.528 ], [ -1.72 ], [ 0.884 ] ]
    ]
    Inputs Statistics: {meanExponent=0.062319133534239506, negative=4, min=0.884, max=0.884, mean=-0.6599999999999999, count=6.0, positive=2, stdDev=1.1199095201547906, zeros=0}
    Output: [
    	[ [ 0.0 ], [ 0.6163367007299045 ], [ 0.0 ] ],
    	[ [ 0.0 ], [ 0.0 ], [ 0.7347348884212754 ] ]
    ]
    Outputs Statistics: {meanExponent=-0.1720256541122639, negative=0, min=0.7347348884212754, max=0.7347348884212754, mean=0.22517859819186334, count=6.0, positive=2, stdDev=0.32027953374828994, zeros=4}
    Feedback for input 0
    Inputs Values: [
    	[ [ -0.732 ], [ 0.824 ], [ -1.688 ] ],
    	[ [ -1.528 ], [ -1.72 ], [ 0.884 ] ]
    ]
    Value Statistics: {meanExponent=0.062319133534239506, negative=4, min=0.884, max=0.884, mean=-0.6599999999999999, count=6.0, positive=2, stdDev=1.1199095201547906, zeros=0}
    Implemented Feedback: [ [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 1.8699535823116036, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 2.0778701595624307 ] ]
    Implemented Statistics: {meanExponent=0.29472461620467927, negative=0, min=2.0778701595624307, max=2.0778701595624307, mean=0.10966177060761206, count=36.0, positive=2, stdDev=0.45281052753329537, zeros=34}
    Measured Feedback: [ [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 1.8701237878393506, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 2.078046452774762 ] ]
    Measured Statistics: {meanExponent=0.2947628030117213, negative=0, min=2.078046452774762, max=2.078046452774762, mean=0.10967139557261424, count=36.0, positive=2, stdDev=0.45285019295954687, zeros=34}
    Feedback Error: [ [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 1.7020552774704711E-4, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 1.7629321233147976E-4 ] ]
    Error Statistics: {meanExponent=-3.7613953740690977, negative=0, min=1.7629321233147976E-4, max=1.7629321233147976E-4, mean=9.624965002181302E-6, count=36.0, positive=2, stdDev=3.9691231939409244E-5, zeros=34}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 9.6250e-06 +- 3.9691e-05 [0.0000e+00 - 1.7629e-04] (36#)
    relativeTol: 4.3964e-05 +- 1.5444e-06 [4.2420e-05 - 4.5509e-05] (2#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=9.6250e-06 +- 3.9691e-05 [0.0000e+00 - 1.7629e-04] (36#), relativeTol=4.3964e-05 +- 1.5444e-06 [4.2420e-05 - 4.5509e-05] (2#)}
```



### Performance
Code from [LayerTestBase.java:149](../../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L149) executed in 0.00 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 0.1338 +- 0.0372 [0.0997 - 0.3762]
    Learning performance: 0.0020 +- 0.0053 [0.0000 - 0.0513]
    
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



