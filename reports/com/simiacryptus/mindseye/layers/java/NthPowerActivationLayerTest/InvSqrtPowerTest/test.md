# NthPowerActivationLayer
## InvSqrtPowerTest
### Json Serialization
Code from [LayerTestBase.java:83](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L83) executed in 0.00 seconds: 
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
      "id": "370a9587-74a1-4959-b406-fa4500002c4b",
      "isFrozen": false,
      "name": "NthPowerActivationLayer/370a9587-74a1-4959-b406-fa4500002c4b",
      "power": -0.5
    }
```



### Example Input/Output Pair
Code from [LayerTestBase.java:120](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L120) executed in 0.00 seconds: 
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
    	[ [ 0.812 ], [ -1.648 ], [ 0.368 ] ],
    	[ [ 0.92 ], [ 1.3 ], [ 1.112 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 1.1097419040461882 ], [ 0.0 ], [ 1.6484511834894675 ] ],
    	[ [ 1.0425720702853738 ], [ 0.8770580193070292 ], [ 0.9483040522636019 ] ]
    ]
```



### Batch Execution
Code from [LayerTestBase.java:138](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L138) executed in 0.00 seconds: 
```java
    BatchingTester batchingTester = getBatchingTester();
    return batchingTester==null?null:batchingTester.test(layer, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (120#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (58#)}
```



### Differential Validation
Code from [LayerTestBase.java:144](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.00 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ 0.812 ], [ -1.648 ], [ 0.368 ] ],
    	[ [ 0.92 ], [ 1.3 ], [ 1.112 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.03063382963762988, negative=1, min=1.112, max=1.112, mean=0.4773333333333334, count=6.0, positive=5, stdDev=0.9930200848366004, zeros=0}
    Output: [
    	[ [ 1.1097419040461882 ], [ 0.0 ], [ 1.6484511834894675 ] ],
    	[ [ 1.0425720702853738 ], [ 0.8770580193070292 ], [ 0.9483040522636019 ] ]
    ]
    Outputs Statistics: {meanExponent=0.040076018518687626, negative=0, min=0.9483040522636019, max=0.9483040522636019, mean=0.9376878715652768, count=6.0, positive=5, stdDev=0.4879950473740553, zeros=1}
    Feedback for input 0
    Inputs Values: [
    	[ [ 0.812 ], [ -1.648 ], [ 0.368 ] ],
    	[ [ 0.92 ], [ 1.3 ], [ 1.112 ] ]
    ]
    Value Statistics: {meanExponent=-0.03063382963762988, negative=1, min=1.112, max=1.112, mean=0.4773333333333334, count=6.0, positive=5, stdDev=0.9930200848366004, zeros=0}
    Implemented Feedback: [ [ -0.6833386108658793, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, -0.566615255589877, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, -0.33733000742578045, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, -2.239743455828081, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, -0.4263957069530584 ] ]
    Implemented Statistics: {meanExponent=-0.18080194010791834, negative=5, min=-0.4263957069530584, max=-0.4263957069530584, mean=-0.11815063990729656, count=36.0, positive=0, stdDev=0.3943178157038273, zeros=31}
    Measured Feedback: [ [ -0.683275501092151, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, -0.566569068312095, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, -0.3373105473269966, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, -2.23928708971588, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, -0.4263669504045442 ] ]
    Measured Statistics: {meanExponent=-0.18084561184923945, negative=5, min=-0.4263669504045442, max=-0.4263669504045442, mean=-0.11813358769032407, count=36.0, positive=0, stdDev=0.3942447127808346, zeros=31}
    Feedback Error: [ [ 6.310977372836479E-5, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 4.618727778205045E-5, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 1.946009878384114E-5, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 4.5636611220079004E-4, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 2.8756548514197178E-5 ] ]
    Error Statistics: {meanExponent=-4.2256371646287345, negative=0, min=2.8756548514197178E-5, max=2.8756548514197178E-5, mean=1.705221697247899E-5, count=36.0, positive=5, stdDev=7.54843176003021E-5, zeros=31}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 1.7052e-05 +- 7.5484e-05 [0.0000e+00 - 4.5637e-04] (36#)
    relativeTol: 5.0279e-05 +- 2.6475e-05 [2.8845e-05 - 1.0189e-04] (5#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=1.7052e-05 +- 7.5484e-05 [0.0000e+00 - 4.5637e-04] (36#), relativeTol=5.0279e-05 +- 2.6475e-05 [2.8845e-05 - 1.0189e-04] (5#)}
```



### Performance
Code from [LayerTestBase.java:149](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L149) executed in 0.00 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 0.1760 +- 0.0499 [0.1254 - 0.5073]
    Learning performance: 0.0028 +- 0.0067 [0.0000 - 0.0655]
    
```

### Function Plots
Code from [ActivationLayerTestBase.java:74](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/java/ActivationLayerTestBase.java#L74) executed in 0.00 seconds: 
```java
    return plot("Value Plot", plotData, x -> new double[]{x[0], x[1]});
```

Returns: 

![Result](etc/test.1.png)



Code from [ActivationLayerTestBase.java:78](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/java/ActivationLayerTestBase.java#L78) executed in 0.01 seconds: 
```java
    return plot("Derivative Plot", plotData, x -> new double[]{x[0], x[2]});
```

Returns: 

![Result](etc/test.2.png)



