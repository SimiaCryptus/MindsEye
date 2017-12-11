# LogActivationLayer
## LogActivationLayerTest
### Json Serialization
Code from [LayerTestBase.java:121](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L121) executed in 0.00 seconds: 
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
      "id": "e2a3bda5-e7e7-4c05-aeb3-4ede00001ebd",
      "isFrozen": true,
      "name": "LogActivationLayer/e2a3bda5-e7e7-4c05-aeb3-4ede00001ebd"
    }
```



### Example Input/Output Pair
Code from [LayerTestBase.java:159](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L159) executed in 0.00 seconds: 
```java
    SimpleEval eval = SimpleEval.run(layer, inputPrototype);
    return String.format("--------------------\nInput: \n[%s]\n--------------------\nOutput: \n%s",
      Arrays.stream(inputPrototype).map(t -> t.prettyPrint()).reduce((a, b) -> a + ",\n" + b).get(),
      eval.getOutput().prettyPrint());
```

Returns: 

```
    --------------------
    Input: 
    [[
    	[ [ 1.04 ], [ -1.832 ], [ -1.448 ] ],
    	[ [ 0.504 ], [ 0.624 ], [ 0.164 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 0.03922071315328133 ], [ 0.6054082662519386 ], [ 0.3701832939635246 ] ],
    	[ [ -0.6851790109107684 ], [ -0.47160491061270937 ], [ -1.8078888511579385 ] ]
    ]
```



### Batch Execution
Code from [LayerTestBase.java:178](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L178) executed in 0.00 seconds: 
```java
    BatchingTester batchingTester = getBatchingTester();
    return batchingTester == null ? null : batchingTester.test(layer, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (120#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (118#)}
```



### Differential Validation
Code from [LayerTestBase.java:186](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L186) executed in 0.00 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ 1.04 ], [ -1.832 ], [ -1.448 ] ],
    	[ [ 0.504 ], [ 0.624 ], [ 0.164 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.14113560922210214, negative=2, min=0.164, max=0.164, mean=-0.158, count=6.0, positive=4, stdDev=1.084360948516068, zeros=0}
    Output: [
    	[ [ 0.03922071315328133 ], [ 0.6054082662519386 ], [ 0.3701832939635246 ] ],
    	[ [ -0.6851790109107684 ], [ -0.47160491061270937 ], [ -1.8078888511579385 ] ]
    ]
    Outputs Statistics: {meanExponent=-0.3815775426381234, negative=3, min=-1.8078888511579385, max=-1.8078888511579385, mean=-0.3249767498854453, count=6.0, positive=3, stdDev=0.7991430142520719, zeros=0}
    Feedback for input 0
    Inputs Values: [
    	[ [ 1.04 ], [ -1.832 ], [ -1.448 ] ],
    	[ [ 0.504 ], [ 0.624 ], [ 0.164 ] ]
    ]
    Value Statistics: {meanExponent=-0.14113560922210214, negative=2, min=0.164, max=0.164, mean=-0.158, count=6.0, positive=4, stdDev=1.084360948516068, zeros=0}
    Implemented Feedback: [ [ 0.9615384615384615, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 1.9841269841269842, 0.0, 0.0, 0.0, 0.0 ], 
```
...[skipping 627 bytes](etc/66.txt)...
```
    , 0.0, 0.0, 0.0, -0.6906077376633846, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 6.097560789619649 ] ]
    Measured Statistics: {meanExponent=0.1411356062379429, negative=2, min=6.097560789619649, max=6.097560789619649, mean=0.2613703067827797, count=36.0, positive=4, stdDev=1.0919573215066518, zeros=30}
    Feedback Error: [ [ -1.0594195520852168E-8, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, -1.029621543580106E-8, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, -2.257959041962465E-9, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, -4.473094339374484E-9, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, -2.8567547216695743E-9, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, -1.8599010687125883E-7 ] ]
    Error Statistics: {meanExponent=-8.038761246629571, negative=6, min=-1.8599010687125883E-7, max=-1.8599010687125883E-7, mean=-6.013009053636627E-9, count=36.0, positive=0, stdDev=3.0524221202601497E-8, zeros=30}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 6.0130e-09 +- 3.0524e-08 [0.0000e+00 - 1.8599e-07] (36#)
    relativeTol: 4.8145e-09 +- 4.8495e-09 [1.3956e-09 - 1.5251e-08] (6#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=6.0130e-09 +- 3.0524e-08 [0.0000e+00 - 1.8599e-07] (36#), relativeTol=4.8145e-09 +- 4.8495e-09 [1.3956e-09 - 1.5251e-08] (6#)}
```



### Performance
Code from [LayerTestBase.java:192](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L192) executed in 0.00 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 0.1412 +- 0.0653 [0.0969 - 0.6441]
    Learning performance: 0.0016 +- 0.0020 [0.0000 - 0.0142]
    
```

### Function Plots
Code from [ActivationLayerTestBase.java:103](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/java/ActivationLayerTestBase.java#L103) executed in 0.07 seconds: 
```java
    return plot("Value Plot", plotData, x -> new double[]{x[0], x[1]});
```

Returns: 

![Result](etc/test.26.png)



Code from [ActivationLayerTestBase.java:107](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/java/ActivationLayerTestBase.java#L107) executed in 0.00 seconds: 
```java
    return plot("Derivative Plot", plotData, x -> new double[]{x[0], x[2]});
```

Returns: 

![Result](etc/test.27.png)



