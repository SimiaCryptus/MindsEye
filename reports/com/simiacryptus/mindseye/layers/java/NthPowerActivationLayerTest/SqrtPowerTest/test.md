# NthPowerActivationLayer
## SqrtPowerTest
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
      "id": "a864e734-2f23-44db-97c1-504000002c59",
      "isFrozen": false,
      "name": "NthPowerActivationLayer/a864e734-2f23-44db-97c1-504000002c59",
      "power": 0.5
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
    	[ [ 1.616 ], [ -1.876 ], [ -0.86 ] ],
    	[ [ 1.044 ], [ -1.356 ], [ 1.888 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 1.2712198865656563 ], [ 0.0 ], [ 0.0 ] ],
    	[ [ 1.0217631819555841 ], [ 0.0 ], [ 1.3740451229854134 ] ]
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
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (120#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (72#)}
```



### Differential Validation
Code from [LayerTestBase.java:144](../../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.00 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ 1.616 ], [ -1.876 ], [ -0.86 ] ],
    	[ [ 1.044 ], [ -1.356 ], [ 1.888 ] ]
    ]
    Inputs Statistics: {meanExponent=0.14052246998075313, negative=3, min=1.888, max=1.888, mean=0.076, count=6.0, positive=3, stdDev=1.4904692773306891, zeros=0}
    Output: [
    	[ [ 1.2712198865656563 ], [ 0.0 ], [ 0.0 ] ],
    	[ [ 1.0217631819555841 ], [ 0.0 ], [ 1.3740451229854134 ] ]
    ]
    Outputs Statistics: {meanExponent=0.0838573075111435, negative=0, min=1.3740451229854134, max=1.3740451229854134, mean=0.611171365251109, count=6.0, positive=3, stdDev=0.6200560960889712, zeros=3}
    Feedback for input 0
    Inputs Values: [
    	[ [ 1.616 ], [ -1.876 ], [ -0.86 ] ],
    	[ [ 1.044 ], [ -1.356 ], [ 1.888 ] ]
    ]
    Value Statistics: {meanExponent=0.14052246998075313, negative=3, min=1.888, max=1.888, mean=0.076, count=6.0, positive=3, stdDev=1.4904692773306891, zeros=0}
    Implemented Feedback: [ [ 0.3933229847047204, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.4893501829289195, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.36388906858723874 ] ]
    Implemented Statistics: {meanExponent=-0.3848873031751247, negative=0, min=0.36388906858723874, max=0.36388906858723874, mean=0.03462672878391329, count=36.0, positive=3, stdDev=0.11588038056559734, zeros=33}
    Measured Feedback: [ [ 0.39331690006916276, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.48933846533483205, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.36388425026956384 ] ]
    Measured Statistics: {meanExponent=-0.3848949260128743, negative=0, min=0.36388425026956384, max=0.36388425026956384, mean=0.03462610043537663, count=36.0, positive=3, stdDev=0.11587819984812579, zeros=33}
    Feedback Error: [ [ -6.084635557623841E-6, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, -1.1717594087468086E-5, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, -4.818317674903927E-6 ] ]
    Error Statistics: {meanExponent=-5.154677183200687, negative=3, min=-4.818317674903927E-6, max=-4.818317674903927E-6, mean=-6.283485366665515E-7, count=36.0, positive=0, stdDev=2.256640807809396E-6, zeros=33}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 6.2835e-07 +- 2.2566e-06 [0.0000e+00 - 1.1718e-05] (36#)
    relativeTol: 8.7761e-06 +- 2.3057e-06 [6.6206e-06 - 1.1973e-05] (3#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=6.2835e-07 +- 2.2566e-06 [0.0000e+00 - 1.1718e-05] (36#), relativeTol=8.7761e-06 +- 2.3057e-06 [6.6206e-06 - 1.1973e-05] (3#)}
```



### Performance
Code from [LayerTestBase.java:149](../../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L149) executed in 0.00 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 0.1407 +- 0.0276 [0.1026 - 0.3249]
    Learning performance: 0.0016 +- 0.0019 [0.0000 - 0.0142]
    
```

### Function Plots
Code from [ActivationLayerTestBase.java:74](../../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/java/ActivationLayerTestBase.java#L74) executed in 0.01 seconds: 
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



