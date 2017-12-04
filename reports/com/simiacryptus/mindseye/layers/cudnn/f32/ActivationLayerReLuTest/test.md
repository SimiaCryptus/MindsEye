# ActivationLayer
## ActivationLayerReLuTest
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
      "class": "com.simiacryptus.mindseye.layers.cudnn.f32.ActivationLayer",
      "id": "370a9587-74a1-4959-b406-fa450000001c",
      "isFrozen": false,
      "name": "ActivationLayer/370a9587-74a1-4959-b406-fa450000001c",
      "mode": 1
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
    	[ [ 0.48, 0.764, 1.428 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 0.47999998927116394, 0.7639999985694885, 1.4279999732971191 ] ]
    ]
```



### Batch Execution
Code from [LayerTestBase.java:138](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L138) executed in 0.01 seconds: 
```java
    BatchingTester batchingTester = getBatchingTester();
    return batchingTester==null?null:batchingTester.test(layer, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (60#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (32#)}
```



### Differential Validation
Code from [LayerTestBase.java:144](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.01 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ 0.48, 0.764, 1.428 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.09364573220285577, negative=0, min=1.428, max=1.428, mean=0.8906666666666666, count=3.0, positive=3, stdDev=0.3972483130514492, zeros=0}
    Output: [
    	[ [ 0.47999998927116394, 0.7639999985694885, 1.4279999732971191 ] ]
    ]
    Outputs Statistics: {meanExponent=-0.09364573841668848, negative=0, min=1.4279999732971191, max=1.4279999732971191, mean=0.8906666537125906, count=3.0, positive=3, stdDev=0.3972483048607918, zeros=0}
    Feedback for input 0
    Inputs Values: [
    	[ [ 0.48, 0.764, 1.428 ] ]
    ]
    Value Statistics: {meanExponent=-0.09364573220285577, negative=0, min=1.428, max=1.428, mean=0.8906666666666666, count=3.0, positive=3, stdDev=0.3972483130514492, zeros=0}
    Implemented Feedback: [ [ 1.0, 0.0, 0.0 ], [ 0.0, 1.0, 0.0 ], [ 0.0, 0.0, 1.0 ] ]
    Implemented Statistics: {meanExponent=0.0, negative=0, min=1.0, max=1.0, mean=0.3333333333333333, count=9.0, positive=3, stdDev=0.4714045207910317, zeros=6}
    Measured Feedback: [ [ 1.0001659393310547, 0.0, 0.0 ], [ 0.0, 1.0001659393310547, 0.0 ], [ 0.0, 0.0, 1.0001659393310547 ] ]
    Measured Statistics: {meanExponent=7.206055713278508E-5, negative=0, min=1.0001659393310547, max=1.0001659393310547, mean=0.3333886464436849, count=9.0, positive=3, stdDev=0.4714827453418679, zeros=6}
    Feedback Error: [ [ 1.659393310546875E-4, 0.0, 0.0 ], [ 0.0, 1.659393310546875E-4, 0.0 ], [ 0.0, 0.0, 1.659393310546875E-4 ] ]
    Error Statistics: {meanExponent=-3.7800506649970242, negative=0, min=1.659393310546875E-4, max=1.659393310546875E-4, mean=5.53131103515625E-5, count=9.0, positive=3, stdDev=7.822455083621932E-5, zeros=6}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 5.5313e-05 +- 7.8225e-05 [0.0000e+00 - 1.6594e-04] (9#)
    relativeTol: 8.2963e-05 +- 0.0000e+00 [8.2963e-05 - 8.2963e-05] (3#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=5.5313e-05 +- 7.8225e-05 [0.0000e+00 - 1.6594e-04] (9#), relativeTol=8.2963e-05 +- 0.0000e+00 [8.2963e-05 - 8.2963e-05] (3#)}
```



### Performance
Code from [LayerTestBase.java:149](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L149) executed in 0.08 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 2.8485 +- 0.3035 [2.3625 - 3.7218]
    Learning performance: 2.3997 +- 1.0086 [1.7868 - 10.5072]
    
```

