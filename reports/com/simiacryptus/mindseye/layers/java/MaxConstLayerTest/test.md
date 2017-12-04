# MaxConstLayer
## MaxConstLayerTest
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
      "class": "com.simiacryptus.mindseye.layers.java.MaxConstLayer",
      "id": "370a9587-74a1-4959-b406-fa4500002c1b",
      "isFrozen": true,
      "name": "MaxConstLayer/370a9587-74a1-4959-b406-fa4500002c1b",
      "value": 0.0
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
    	[ [ 1.616 ], [ 1.884 ], [ -0.752 ] ],
    	[ [ -0.396 ], [ 0.028 ], [ 1.44 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 1.616 ], [ 1.884 ], [ 0.0 ] ],
    	[ [ 0.0 ], [ 0.028 ], [ 1.44 ] ]
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
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (120#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (66#)}
```



### Differential Validation
Code from [LayerTestBase.java:144](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.00 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ 1.616 ], [ 1.884 ], [ -0.752 ] ],
    	[ [ -0.396 ], [ 0.028 ], [ 1.44 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.2395073658583251, negative=2, min=1.44, max=1.44, mean=0.6366666666666666, count=6.0, positive=4, stdDev=1.042876577335763, zeros=0}
    Output: [
    	[ [ 1.616 ], [ 1.884 ], [ 0.0 ] ],
    	[ [ 0.0 ], [ 0.028 ], [ 1.44 ] ]
    ]
    Outputs Statistics: {meanExponent=-0.2277393054167763, negative=0, min=1.44, max=1.44, mean=0.828, count=6.0, positive=4, stdDev=0.8288337187478158, zeros=2}
    Feedback for input 0
    Inputs Values: [
    	[ [ 1.616 ], [ 1.884 ], [ -0.752 ] ],
    	[ [ -0.396 ], [ 0.028 ], [ 1.44 ] ]
    ]
    Value Statistics: {meanExponent=-0.2395073658583251, negative=2, min=1.44, max=1.44, mean=0.6366666666666666, count=6.0, positive=4, stdDev=1.042876577335763, zeros=0}
    Implemented Feedback: [ [ 1.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 1.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 1.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 1.0 ] ]
    Implemented Statistics: {meanExponent=0.0, negative=0, min=1.0, max=1.0, mean=0.1111111111111111, count=36.0, positive=4, stdDev=0.31426968052735443, zeros=32}
    Measured Feedback: [ [ 0.9999999999998899, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.9999999999998899, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.999999999999994, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.9999999999998899 ] ]
    Measured Statistics: {meanExponent=-3.652390279570773E-14, negative=0, min=0.9999999999998899, max=0.9999999999998899, mean=0.11111111111110177, count=36.0, positive=4, stdDev=0.31426968052732807, zeros=32}
    Feedback Error: [ [ -1.1013412404281553E-13, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, -1.1013412404281553E-13, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, -5.995204332975845E-15, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, -1.1013412404281553E-13 ] ]
    Error Statistics: {meanExponent=-13.27410757611963, negative=4, min=-1.1013412404281553E-13, max=-1.1013412404281553E-13, mean=-9.344377123928401E-15, count=36.0, positive=0, stdDev=3.040517705049855E-14, zeros=32}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 9.3444e-15 +- 3.0405e-14 [0.0000e+00 - 1.1013e-13] (36#)
    relativeTol: 4.2050e-14 +- 2.2547e-14 [2.9976e-15 - 5.5067e-14] (4#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=9.3444e-15 +- 3.0405e-14 [0.0000e+00 - 1.1013e-13] (36#), relativeTol=4.2050e-14 +- 2.2547e-14 [2.9976e-15 - 5.5067e-14] (4#)}
```



### Performance
Code from [LayerTestBase.java:149](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L149) executed in 0.00 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 0.1443 +- 0.0438 [0.0940 - 0.3163]
    Learning performance: 0.0029 +- 0.0040 [0.0000 - 0.0342]
    
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



