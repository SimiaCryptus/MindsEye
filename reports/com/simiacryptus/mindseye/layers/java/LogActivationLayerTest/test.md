# LogActivationLayer
## LogActivationLayerTest
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
      "class": "com.simiacryptus.mindseye.layers.java.LogActivationLayer",
      "id": "a864e734-2f23-44db-97c1-504000002c14",
      "isFrozen": true,
      "name": "LogActivationLayer/a864e734-2f23-44db-97c1-504000002c14"
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
    	[ [ -0.316 ], [ -1.424 ], [ 0.008 ] ],
    	[ [ -0.144 ], [ -0.268 ], [ 1.008 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ -1.152013065395225 ], [ 0.35346981298978397 ], [ -4.8283137373023015 ] ],
    	[ [ -1.9379419794061366 ], [ -1.3167682984712803 ], [ 0.007968169649176881 ] ]
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
    	[ [ -0.316 ], [ -1.424 ], [ 0.008 ] ],
    	[ [ -0.144 ], [ -0.268 ], [ 1.008 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.6422925204758783, negative=4, min=1.008, max=1.008, mean=-0.18933333333333335, count=6.0, positive=2, stdDev=0.7096052110543971, zeros=0}
    Output: [
    	[ [ -1.152013065395225 ], [ 0.35346981298978397 ], [ -4.8283137373023015 ] ],
    	[ [ -1.9379419794061366 ], [ -1.3167682984712803 ], [ 0.007968169649176881 ] ]
    ]
    Outputs Statistics: {meanExponent=-0.23303101305935736, negative=4, min=0.007968169649176881, max=0.007968169649176881, mean=-1.4789331829893302, count=6.0, positive=2, stdDev=1.6897134746265823, zeros=0}
    Feedback for input 0
    Inputs Values: [
    	[ [ -0.316 ], [ -1.424 ], [ 0.008 ] ],
    	[ [ -0.144 ], [ -0.268 ], [ 1.008 ] ]
    ]
    Value Statistics: {meanExponent=-0.6422925204758783, negative=4, min=1.008, max=1.008, mean=-0.18933333333333335, count=6.0, positive=2, stdDev=0.7096052110543971, zeros=0}
    Implemented Feedback: [ [ -3.1645569620253164, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, -6.944444444444445, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, -0.702247191011236, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, -3.731343283582089, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 125.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.9920634920634921 ] ]
    Implemented Statistics: {meanExponent=0.6422925204758784, negative=4, min=0.9920634920634921, max=0.9920634920634921, mean=3.0958186558611223, count=36.0, positive=2, stdDev=20.65161591726889, zeros=30}
    Measured Feedback: [ [ -3.164557016432923, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, -6.944444685963447, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, -0.7022471881779069, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, -3.731343345414473, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 124.99992187997577, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.9920634810173246 ] ]
    Measured Statistics: {meanExponent=0.6422924791030339, negative=4, min=0.9920634810173246, max=0.9920634810173246, mean=3.095816475694565, count=36.0, positive=2, stdDev=20.651603112301796, zeros=30}
    Feedback Error: [ [ -5.4407606420170396E-8, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, -2.4151900213098543E-7, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 2.833329126872286E-9, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, -6.18323836576451E-8, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, -7.812002422724618E-5, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, -1.1046167536221674E-8 ] ]
    Error Statistics: {meanExponent=-6.950317010624405, negative=5, min=-1.1046167536221674E-8, max=-1.1046167536221674E-8, mean=-2.180166557162898E-6, count=36.0, positive=1, stdDev=1.2836245217655076E-5, zeros=30}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 2.1803e-06 +- 1.2836e-05 [0.0000e+00 - 7.8120e-05] (36#)
    relativeTol: 5.9056e-08 +- 1.1343e-07 [2.0173e-09 - 3.1248e-07] (6#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=2.1803e-06 +- 1.2836e-05 [0.0000e+00 - 7.8120e-05] (36#), relativeTol=5.9056e-08 +- 1.1343e-07 [2.0173e-09 - 3.1248e-07] (6#)}
```



### Performance
Code from [LayerTestBase.java:149](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L149) executed in 0.00 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 0.1390 +- 0.0520 [0.0940 - 0.4759]
    Learning performance: 0.0016 +- 0.0024 [0.0000 - 0.0200]
    
```

### Function Plots
Code from [ActivationLayerTestBase.java:74](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/java/ActivationLayerTestBase.java#L74) executed in 0.01 seconds: 
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



