# GaussianActivationLayer
## GaussianActivationLayerTest
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
      "class": "com.simiacryptus.mindseye.layers.java.GaussianActivationLayer",
      "id": "a864e734-2f23-44db-97c1-504000002bb4",
      "isFrozen": true,
      "name": "GaussianActivationLayer/a864e734-2f23-44db-97c1-504000002bb4",
      "mean": 0.0,
      "stddev": 1.0
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
    	[ [ -0.368 ], [ 1.888 ], [ 0.904 ] ],
    	[ [ -1.936 ], [ 0.336 ], [ -1.76 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 0.3728233614856777 ], [ 0.06712420745634257 ], [ 0.26512694414028337 ] ],
    	[ [ 0.061238051067276554 ], [ 0.3770465843680881 ], [ 0.08477636130802224 ] ]
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
    	[ [ -0.368 ], [ 1.888 ], [ 0.904 ] ],
    	[ [ -1.936 ], [ 0.336 ], [ -1.76 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.02387074378545002, negative=3, min=-1.76, max=-1.76, mean=-0.156, count=6.0, positive=3, stdDev=1.3741518596331823, zeros=0}
    Output: [
    	[ [ 0.3728233614856777 ], [ 0.06712420745634257 ], [ 0.26512694414028337 ] ],
    	[ [ 0.061238051067276554 ], [ 0.3770465843680881 ], [ 0.08477636130802224 ] ]
    ]
    Outputs Statistics: {meanExponent=-0.8144121168755385, negative=0, min=0.08477636130802224, max=0.08477636130802224, mean=0.2046892516376151, count=6.0, positive=6, stdDev=0.13875057555933135, zeros=0}
    Feedback for input 0
    Inputs Values: [
    	[ [ -0.368 ], [ 1.888 ], [ 0.904 ] ],
    	[ [ -1.936 ], [ 0.336 ], [ -1.76 ] ]
    ]
    Value Statistics: {meanExponent=-0.02387074378545002, negative=3, min=-1.76, max=-1.76, mean=-0.156, count=6.0, positive=3, stdDev=1.3741518596331823, zeros=0}
    Implemented Feedback: [ [ 0.13719899702672939, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.11855686686624742, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, -0.12673050367757474, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, -0.12668765234767762, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, -0.23967475750281617, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.14920639590211912 ] ]
    Implemented Statistics: {meanExponent=-0.8382828606609883, negative=3, min=0.14920639590211912, max=0.14920639590211912, mean=-0.0024480737148047955, count=36.0, positive=3, stdDev=0.06335097668967368, zeros=30}
    Measured Feedback: [ [ 0.13718287966535403, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.11856528141625244, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, -0.1267218966474748, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, -0.1267043757141595, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, -0.23967717967865454, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.14921528727104838 ] ]
    Measured Statistics: {meanExponent=-0.8382765442775376, negative=3, min=0.14921528727104838, max=0.14921528727104838, mean=-0.0024483334357676104, count=36.0, positive=3, stdDev=0.06335172159409384, zeros=30}
    Feedback Error: [ [ -1.6117361375356243E-5, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 8.41455000502589E-6, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 8.607030099944124E-6, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, -1.6723366481868895E-5, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, -2.4221758383735636E-6, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 8.891368929259125E-6 ] ]
    Error Statistics: {meanExponent=-5.0627206389090205, negative=3, min=8.891368929259125E-6, max=8.891368929259125E-6, mean=-2.597209628158212E-7, count=36.0, positive=3, stdDev=4.6152553231418385E-6, zeros=30}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 1.6993e-06 +- 4.2989e-06 [0.0000e+00 - 1.6723e-05] (36#)
    relativeTol: 3.8172e-05 +- 1.9956e-05 [5.0530e-06 - 6.5998e-05] (6#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=1.6993e-06 +- 4.2989e-06 [0.0000e+00 - 1.6723e-05] (36#), relativeTol=3.8172e-05 +- 1.9956e-05 [5.0530e-06 - 6.5998e-05] (6#)}
```



### Performance
Code from [LayerTestBase.java:149](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L149) executed in 0.00 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 0.1600 +- 0.1305 [0.1083 - 1.3223]
    Learning performance: 0.0026 +- 0.0038 [0.0000 - 0.0200]
    
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



