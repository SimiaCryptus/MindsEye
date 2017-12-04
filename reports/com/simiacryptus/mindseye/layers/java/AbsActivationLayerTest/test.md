# AbsActivationLayer
## AbsActivationLayerTest
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
      "class": "com.simiacryptus.mindseye.layers.java.AbsActivationLayer",
      "id": "370a9587-74a1-4959-b406-fa4500002b74",
      "isFrozen": true,
      "name": "AbsActivationLayer/370a9587-74a1-4959-b406-fa4500002b74"
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
    	[ [ 0.028 ], [ -1.152 ], [ 1.768 ] ],
    	[ [ -0.504 ], [ 1.584 ], [ -1.256 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 0.028 ], [ 1.152 ], [ 1.768 ] ],
    	[ [ 0.504 ], [ 1.584 ], [ 1.256 ] ]
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
    	[ [ 0.028 ], [ -1.152 ], [ 1.768 ] ],
    	[ [ -0.504 ], [ 1.584 ], [ -1.256 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.20712197929889264, negative=3, min=-1.256, max=-1.256, mean=0.07800000000000003, count=6.0, positive=3, stdDev=1.208153963698336, zeros=0}
    Output: [
    	[ [ 0.028 ], [ 1.152 ], [ 1.768 ] ],
    	[ [ 0.504 ], [ 1.584 ], [ 1.256 ] ]
    ]
    Outputs Statistics: {meanExponent=-0.20712197929889264, negative=0, min=1.256, max=1.256, mean=1.0486666666666666, count=6.0, positive=6, stdDev=0.604994398504831, zeros=0}
    Feedback for input 0
    Inputs Values: [
    	[ [ 0.028 ], [ -1.152 ], [ 1.768 ] ],
    	[ [ -0.504 ], [ 1.584 ], [ -1.256 ] ]
    ]
    Value Statistics: {meanExponent=-0.20712197929889264, negative=3, min=-1.256, max=-1.256, mean=0.07800000000000003, count=6.0, positive=3, stdDev=1.208153963698336, zeros=0}
    Implemented Feedback: [ [ 1.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, -1.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, -1.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 1.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 1.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, -1.0 ] ]
    Implemented Statistics: {meanExponent=0.0, negative=3, min=-1.0, max=-1.0, mean=0.0, count=36.0, positive=3, stdDev=0.408248290463863, zeros=30}
    Measured Feedback: [ [ 0.999999999999994, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, -0.9999999999998899, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, -0.9999999999998899, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.9999999999998899, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.9999999999998899, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, -0.9999999999998899 ] ]
    Measured Statistics: {meanExponent=-4.029281597748704E-14, negative=3, min=-0.9999999999998899, max=-0.9999999999998899, mean=2.892747769717769E-15, count=36.0, positive=3, stdDev=0.40824829046382516, zeros=30}
    Feedback Error: [ [ -5.995204332975845E-15, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 1.1013412404281553E-13, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 1.1013412404281553E-13, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, -1.1013412404281553E-13, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, -1.1013412404281553E-13, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 1.1013412404281553E-13 ] ]
    Error Statistics: {meanExponent=-13.168764416758691, negative=3, min=1.1013412404281553E-13, max=1.1013412404281553E-13, mean=2.892747769717769E-15, count=36.0, positive=3, stdDev=4.095469083181168E-14, zeros=30}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 1.5463e-14 +- 3.8034e-14 [0.0000e+00 - 1.1013e-13] (36#)
    relativeTol: 4.6389e-14 +- 1.9405e-14 [2.9976e-15 - 5.5067e-14] (6#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=1.5463e-14 +- 3.8034e-14 [0.0000e+00 - 1.1013e-13] (36#), relativeTol=4.6389e-14 +- 1.9405e-14 [2.9976e-15 - 5.5067e-14] (6#)}
```



### Performance
Code from [LayerTestBase.java:149](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L149) executed in 0.00 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 0.1909 +- 0.0735 [0.1282 - 0.8008]
    Learning performance: 0.0030 +- 0.0021 [0.0000 - 0.0142]
    
```

### Function Plots
Code from [ActivationLayerTestBase.java:74](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/java/ActivationLayerTestBase.java#L74) executed in 0.10 seconds: 
```java
    return plot("Value Plot", plotData, x -> new double[]{x[0], x[1]});
```

Returns: 

![Result](etc/test.1.png)



Code from [ActivationLayerTestBase.java:78](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/java/ActivationLayerTestBase.java#L78) executed in 0.01 seconds: 
```java
    return plot("Derivative Plot", plotData, x -> new double[]{x[0], x[2]});
```

Returns: 

![Result](etc/test.2.png)



