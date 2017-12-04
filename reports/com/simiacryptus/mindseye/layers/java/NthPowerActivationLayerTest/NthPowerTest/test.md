# NthPowerActivationLayer
## NthPowerTest
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
      "id": "370a9587-74a1-4959-b406-fa4500002c52",
      "isFrozen": false,
      "name": "NthPowerActivationLayer/370a9587-74a1-4959-b406-fa4500002c52",
      "power": 2.5
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
    	[ [ 0.984 ], [ 0.772 ], [ -0.888 ] ],
    	[ [ 1.624 ], [ -1.708 ], [ -1.912 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 0.9604787174276294 ], [ 0.5236525838890056 ], [ 0.0 ] ],
    	[ [ 3.360973298996978 ], [ 0.0 ], [ 0.0 ] ]
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
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (120#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (46#)}
```



### Differential Validation
Code from [LayerTestBase.java:144](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.00 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ 0.984 ], [ 0.772 ], [ -0.888 ] ],
    	[ [ 1.624 ], [ -1.708 ], [ -1.912 ] ]
    ]
    Inputs Statistics: {meanExponent=0.0922645239573171, negative=3, min=-1.912, max=-1.912, mean=-0.18800000000000003, count=6.0, positive=3, stdDev=1.3754456247582696, zeros=0}
    Output: [
    	[ [ 0.9604787174276294 ], [ 0.5236525838890056 ], [ 0.0 ] ],
    	[ [ 3.360973298996978 ], [ 0.0 ], [ 0.0 ] ]
    ]
    Outputs Statistics: {meanExponent=0.07599868639352851, negative=0, min=0.0, max=0.0, mean=0.8075174333856022, count=6.0, positive=3, stdDev=1.195851612902263, zeros=3}
    Feedback for input 0
    Inputs Values: [
    	[ [ 0.984 ], [ 0.772 ], [ -0.888 ] ],
    	[ [ 1.624 ], [ -1.708 ], [ -1.912 ] ]
    ]
    Value Statistics: {meanExponent=0.0922645239573171, negative=3, min=-1.912, max=-1.912, mean=-0.18800000000000003, count=6.0, positive=3, stdDev=1.3754456247582696, zeros=0}
    Implemented Feedback: [ [ 2.4402406438710096, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 5.173912098209633, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 1.6957661395369352, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ] ]
    Implemented Statistics: {meanExponent=0.4435392205081547, negative=0, min=0.0, max=0.0, mean=0.25860885782271054, count=36.0, positive=3, stdDev=0.9602102795106162, zeros=33}
    Measured Feedback: [ [ 2.4404266409727704, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 5.174151043645381, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 1.695930887213315, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ] ]
    Measured Statistics: {meanExponent=0.443571003221986, negative=0, min=0.0, max=0.0, mean=0.2586252381064296, count=36.0, positive=3, stdDev=0.9602628443751524, zeros=33}
    Feedback Error: [ [ 1.859971017608153E-4, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 2.3894543574787264E-4, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 1.6474767637975063E-4, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ] ]
    Error Statistics: {meanExponent=-3.7117919286056673, negative=0, min=0.0, max=0.0, mean=1.6380283719123295E-5, count=36.0, positive=3, stdDev=5.506873154694144E-5, zeros=33}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 1.6380e-05 +- 5.5069e-05 [0.0000e+00 - 2.3895e-04] (36#)
    relativeTol: 3.6591e-05 +- 1.0459e-05 [2.3091e-05 - 4.8574e-05] (3#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=1.6380e-05 +- 5.5069e-05 [0.0000e+00 - 2.3895e-04] (36#), relativeTol=3.6591e-05 +- 1.0459e-05 [2.3091e-05 - 4.8574e-05] (3#)}
```



### Performance
Code from [LayerTestBase.java:149](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L149) executed in 0.00 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 0.1509 +- 0.0277 [0.0998 - 0.2565]
    Learning performance: 0.0024 +- 0.0050 [0.0000 - 0.0484]
    
```

### Function Plots
Code from [ActivationLayerTestBase.java:74](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/java/ActivationLayerTestBase.java#L74) executed in 0.01 seconds: 
```java
    return plot("Value Plot", plotData, x -> new double[]{x[0], x[1]});
```

Returns: 

![Result](etc/test.1.png)



Code from [ActivationLayerTestBase.java:78](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/java/ActivationLayerTestBase.java#L78) executed in 0.00 seconds: 
```java
    return plot("Derivative Plot", plotData, x -> new double[]{x[0], x[2]});
```

Returns: 

![Result](etc/test.2.png)



