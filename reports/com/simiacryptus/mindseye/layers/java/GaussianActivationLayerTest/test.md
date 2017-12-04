# GaussianActivationLayer
## GaussianActivationLayerTest
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
      "class": "com.simiacryptus.mindseye.layers.java.GaussianActivationLayer",
      "id": "370a9587-74a1-4959-b406-fa4500002bb4",
      "isFrozen": true,
      "name": "GaussianActivationLayer/370a9587-74a1-4959-b406-fa4500002bb4",
      "mean": 0.0,
      "stddev": 1.0
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
    	[ [ -1.492 ], [ -1.336 ], [ 1.3 ] ],
    	[ [ 0.996 ], [ 0.36 ], [ -0.1 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 0.1310769749636993 ], [ 0.16342738214570318 ], [ 0.17136859204780736 ] ],
    	[ [ 0.2429386022500282 ], [ 0.3739106053731284 ], [ 0.3969525474770118 ] ]
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
    	[ [ -1.492 ], [ -1.336 ], [ 1.3 ] ],
    	[ [ 0.996 ], [ 0.36 ], [ -0.1 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.17198658787100005, negative=3, min=-0.1, max=-0.1, mean=-0.045333333333333316, count=6.0, positive=3, stdDev=1.0661648819744325, zeros=0}
    Output: [
    	[ [ 0.1310769749636993 ], [ 0.16342738214570318 ], [ 0.17136859204780736 ] ],
    	[ [ 0.2429386022500282 ], [ 0.3739106053731284 ], [ 0.3969525474770118 ] ]
    ]
    Outputs Statistics: {meanExponent=-0.6463691053444403, negative=0, min=0.3969525474770118, max=0.3969525474770118, mean=0.24661245070956306, count=6.0, positive=6, stdDev=0.10388318928280765, zeros=0}
    Feedback for input 0
    Inputs Values: [
    	[ [ -1.492 ], [ -1.336 ], [ 1.3 ] ],
    	[ [ 0.996 ], [ 0.36 ], [ -0.1 ] ]
    ]
    Value Statistics: {meanExponent=-0.17198658787100005, negative=3, min=-0.1, max=-0.1, mean=-0.045333333333333316, count=6.0, positive=3, stdDev=1.0661648819744325, zeros=0}
    Implemented Feedback: [ [ 0.19556684664583934, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, -0.2419668478410281, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.21833898254665943, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, -0.13460781793432625, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, -0.22277916966214956, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.03969525474770118 ] ]
    Implemented Statistics: {meanExponent=-0.8183556932154404, negative=3, min=0.03969525474770118, max=0.03969525474770118, mean=-0.004048687541591776, count=36.0, positive=3, stdDev=0.07695628038145597, zeros=30}
    Measured Feedback: [ [ 0.19557488183136185, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, -0.24196694401207308, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.2183453957793624, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, -0.1346240898802975, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, -0.22277325695901906, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.03967560539885895 ] ]
    Measured Statistics: {meanExponent=-0.818379574785551, negative=3, min=0.03967560539885895, max=0.03967560539885895, mean=-0.004049122440050179, count=36.0, positive=3, stdDev=0.07695737228793363, zeros=30}
    Feedback Error: [ [ 8.035185522503596E-6, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, -9.61710449887132E-8, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 6.413232702978666E-6, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, -1.6271945971257917E-5, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 5.912703130495078E-6, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, -1.9649348842229464E-5 ] ]
    Error Statistics: {meanExponent=-5.338051504539973, negative=3, min=-1.9649348842229464E-5, max=-1.9649348842229464E-5, mean=-4.348984584027432E-7, count=36.0, positive=3, stdDev=4.668830697025842E-6, zeros=30}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 1.5661e-06 +- 4.4198e-06 [0.0000e+00 - 1.9649e-05] (36#)
    relativeTol: 5.9450e-05 +- 8.6167e-05 [1.9873e-07 - 2.4756e-04] (6#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=1.5661e-06 +- 4.4198e-06 [0.0000e+00 - 1.9649e-05] (36#), relativeTol=5.9450e-05 +- 8.6167e-05 [1.9873e-07 - 2.4756e-04] (6#)}
```



### Performance
Code from [LayerTestBase.java:149](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L149) executed in 0.00 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 0.1621 +- 0.0343 [0.1054 - 0.2850]
    Learning performance: 0.0033 +- 0.0038 [0.0000 - 0.0285]
    
```

### Function Plots
Code from [ActivationLayerTestBase.java:74](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/java/ActivationLayerTestBase.java#L74) executed in 0.01 seconds: 
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



