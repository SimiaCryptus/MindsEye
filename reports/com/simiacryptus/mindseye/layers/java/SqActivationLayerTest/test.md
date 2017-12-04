# SqActivationLayer
## SqActivationLayerTest
### Json Serialization
Code from [LayerTestBase.java:83](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L83) executed in 0.00 seconds: 
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
      "id": "370a9587-74a1-4959-b406-fa4500002c9a",
      "isFrozen": true,
      "name": "SqActivationLayer/370a9587-74a1-4959-b406-fa4500002c9a"
    }
```



### Example Input/Output Pair
Code from [LayerTestBase.java:120](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L120) executed in 0.00 seconds: 
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
    	[ [ 0.056 ], [ -0.1 ], [ 0.212 ] ],
    	[ [ -0.372 ], [ 1.132 ], [ -1.532 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 0.0031360000000000003 ], [ 0.010000000000000002 ], [ 0.044944 ] ],
    	[ [ 0.138384 ], [ 1.2814239999999997 ], [ 2.347024 ] ]
    ]
```



### Batch Execution
Code from [LayerTestBase.java:138](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L138) executed in 0.00 seconds: 
```java
    BatchingTester batchingTester = getBatchingTester();
    return batchingTester==null?null:batchingTester.test(layer, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (120#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (120#)}
```



### Differential Validation
Code from [LayerTestBase.java:144](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.00 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ 0.056 ], [ -0.1 ], [ 0.212 ] ],
    	[ [ -0.372 ], [ 1.132 ], [ -1.532 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.5193046633390521, negative=3, min=-1.532, max=-1.532, mean=-0.1006666666666667, count=6.0, positive=3, stdDev=0.792055273043211, zeros=0}
    Output: [
    	[ [ 0.0031360000000000003 ], [ 0.010000000000000002 ], [ 0.044944 ] ],
    	[ [ 0.138384 ], [ 1.2814239999999997 ], [ 2.347024 ] ]
    ]
    Outputs Statistics: {meanExponent=-1.0386093266781042, negative=0, min=2.347024, max=2.347024, mean=0.6374853333333333, count=6.0, positive=6, stdDev=0.8882120385123262, zeros=0}
    Feedback for input 0
    Inputs Values: [
    	[ [ 0.056 ], [ -0.1 ], [ 0.212 ] ],
    	[ [ -0.372 ], [ 1.132 ], [ -1.532 ] ]
    ]
    Value Statistics: {meanExponent=-0.5193046633390521, negative=3, min=-1.532, max=-1.532, mean=-0.1006666666666667, count=6.0, positive=3, stdDev=0.792055273043211, zeros=0}
    Implemented Feedback: [ [ 0.112, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, -0.744, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, -0.2, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 2.264, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.424, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, -3.064 ] ]
    Implemented Statistics: {meanExponent=-0.21827466767507095, negative=3, min=-3.064, max=-3.064, mean=-0.03355555555555557, count=36.0, positive=3, stdDev=0.6510485749263109, zeros=30}
    Measured Feedback: [ [ 0.11210000000000213, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, -0.7439000000000751, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, -0.1999000000000098, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 2.264100000000102, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.4240999999999273, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, -3.0638999999998973 ] ]
    Measured Statistics: {meanExponent=-0.21823809513006384, negative=3, min=-3.0638999999998973, max=-3.0638999999998973, mean=-0.033538888888887515, count=36.0, positive=3, stdDev=0.6510442809132674, zeros=30}
    Feedback Error: [ [ 1.0000000000212617E-4, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 9.999999992493791E-5, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 9.999999999021902E-5, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 1.0000000010235155E-4, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 9.999999992732489E-5, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 1.0000000010279564E-4 ] ]
    Error Statistics: {meanExponent=-3.999999999963986, negative=0, min=1.0000000010279564E-4, max=1.0000000010279564E-4, mean=1.6666666668048754E-5, count=36.0, positive=6, stdDev=3.726779962808694E-5, zeros=30}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 1.6667e-05 +- 3.7268e-05 [0.0000e+00 - 1.0000e-04] (36#)
    relativeTol: 1.5330e-04 +- 1.5265e-04 [1.6319e-05 - 4.4623e-04] (6#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=1.6667e-05 +- 3.7268e-05 [0.0000e+00 - 1.0000e-04] (36#), relativeTol=1.5330e-04 +- 1.5265e-04 [1.6319e-05 - 4.4623e-04] (6#)}
```



### Performance
Code from [LayerTestBase.java:149](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L149) executed in 0.00 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 0.1252 +- 0.0344 [0.0883 - 0.3049]
    Learning performance: 0.0013 +- 0.0018 [0.0000 - 0.0086]
    
```

### Function Plots
Code from [ActivationLayerTestBase.java:74](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/java/ActivationLayerTestBase.java#L74) executed in 0.00 seconds: 
```java
    return plot("Value Plot", plotData, x -> new double[]{x[0], x[1]});
```

Returns: 

![Result](etc/test.1.png)



Code from [ActivationLayerTestBase.java:78](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/java/ActivationLayerTestBase.java#L78) executed in 0.00 seconds: 
```java
    return plot("Derivative Plot", plotData, x -> new double[]{x[0], x[2]});
```

Returns: 

![Result](etc/test.2.png)



