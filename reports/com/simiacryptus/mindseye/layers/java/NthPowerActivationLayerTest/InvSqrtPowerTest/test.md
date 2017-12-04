# NthPowerActivationLayer
## InvSqrtPowerTest
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
      "id": "a864e734-2f23-44db-97c1-504000002c4b",
      "isFrozen": false,
      "name": "NthPowerActivationLayer/a864e734-2f23-44db-97c1-504000002c4b",
      "power": -0.5
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
    	[ [ 1.864 ], [ -1.292 ], [ -1.548 ] ],
    	[ [ -0.028 ], [ 1.756 ], [ -1.684 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 0.7324484191363095 ], [ 0.0 ], [ 0.0 ] ],
    	[ [ 0.0 ], [ 0.7546363905912276 ], [ 0.0 ] ]
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
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (120#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (66#)}
```



### Differential Validation
Code from [LayerTestBase.java:144](../../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.00 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ 1.864 ], [ -1.292 ], [ -1.548 ] ],
    	[ [ -0.028 ], [ 1.756 ], [ -1.684 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.0850826653166941, negative=4, min=-1.684, max=-1.684, mean=-0.15533333333333335, count=6.0, positive=2, stdDev=1.4897130223264106, zeros=0}
    Output: [
    	[ [ 0.7324484191363095 ], [ 0.0 ], [ 0.0 ] ],
    	[ [ 0.0 ], [ 0.7546363905912276 ], [ 0.0 ] ]
    ]
    Outputs Statistics: {meanExponent=-0.1287426048970116, negative=0, min=0.0, max=0.0, mean=0.24784746828792284, count=6.0, positive=2, stdDev=0.35056776888343033, zeros=4}
    Feedback for input 0
    Inputs Values: [
    	[ [ 1.864 ], [ -1.292 ], [ -1.548 ] ],
    	[ [ -0.028 ], [ 1.756 ], [ -1.684 ] ]
    ]
    Value Statistics: {meanExponent=-0.0850826653166941, negative=4, min=-1.684, max=-1.684, mean=-0.15533333333333335, count=6.0, positive=2, stdDev=1.4897130223264106, zeros=0}
    Implemented Feedback: [ [ -0.19647221543355942, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, -0.21487368752597596, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ] ]
    Implemented Statistics: {meanExponent=-0.687257810355016, negative=2, min=0.0, max=0.0, mean=-0.011426275082209315, count=36.0, positive=0, stdDev=0.04716162563969258, zeros=34}
    Measured Feedback: [ [ -0.19646431052122715, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, -0.21486451055374722, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ] ]
    Measured Statistics: {meanExponent=-0.687275821558016, negative=2, min=0.0, max=0.0, mean=-0.011425800585415955, count=36.0, positive=0, stdDev=0.04715966441528585, zeros=34}
    Feedback Error: [ [ 7.90491233226831E-6, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 9.176972228741587E-6, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ] ]
    Error Statistics: {meanExponent=-5.069701762352919, negative=0, min=0.0, max=0.0, mean=4.74496793361386E-7, count=36.0, positive=2, stdDev=1.9621357327835214E-6, zeros=34}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 4.7450e-07 +- 1.9621e-06 [0.0000e+00 - 9.1770e-06] (36#)
    relativeTol: 2.0736e-05 +- 6.1863e-07 [2.0118e-05 - 2.1355e-05] (2#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=4.7450e-07 +- 1.9621e-06 [0.0000e+00 - 9.1770e-06] (36#), relativeTol=2.0736e-05 +- 6.1863e-07 [2.0118e-05 - 2.1355e-05] (2#)}
```



### Performance
Code from [LayerTestBase.java:149](../../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L149) executed in 0.00 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 0.1492 +- 0.0772 [0.1111 - 0.6925]
    Learning performance: 0.0021 +- 0.0064 [0.0000 - 0.0627]
    
```

### Function Plots
Code from [ActivationLayerTestBase.java:74](../../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/java/ActivationLayerTestBase.java#L74) executed in 0.01 seconds: 
```java
    return plot("Value Plot", plotData, x -> new double[]{x[0], x[1]});
```

Returns: 

![Result](etc/test.1.png)



Code from [ActivationLayerTestBase.java:78](../../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/java/ActivationLayerTestBase.java#L78) executed in 0.01 seconds: 
```java
    return plot("Derivative Plot", plotData, x -> new double[]{x[0], x[2]});
```

Returns: 

![Result](etc/test.2.png)



