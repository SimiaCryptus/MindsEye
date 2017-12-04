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
      "class": "com.simiacryptus.mindseye.layers.cudnn.f64.ActivationLayer",
      "id": "370a9587-74a1-4959-b406-fa4500000447",
      "isFrozen": false,
      "name": "ActivationLayer/370a9587-74a1-4959-b406-fa4500000447",
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
    	[ [ -1.248, 0.392, 1.484 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 0.0, 0.392, 1.484 ] ]
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
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (60#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (24#)}
```



### Differential Validation
Code from [LayerTestBase.java:144](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.01 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ -1.248, 0.392, 1.484 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.046355148896709754, negative=1, min=1.484, max=1.484, mean=0.20933333333333334, count=3.0, positive=2, stdDev=1.1227885919540785, zeros=0}
    Output: [
    	[ [ 0.0, 0.392, 1.484 ] ]
    ]
    Outputs Statistics: {meanExponent=-0.11764001601826723, negative=0, min=1.484, max=1.484, mean=0.6253333333333333, count=3.0, positive=2, stdDev=0.6279051591513554, zeros=1}
    Feedback for input 0
    Inputs Values: [
    	[ [ -1.248, 0.392, 1.484 ] ]
    ]
    Value Statistics: {meanExponent=-0.046355148896709754, negative=1, min=1.484, max=1.484, mean=0.20933333333333334, count=3.0, positive=2, stdDev=1.1227885919540785, zeros=0}
    Implemented Feedback: [ [ 0.0, 0.0, 0.0 ], [ 0.0, 1.0, 0.0 ], [ 0.0, 0.0, 1.0 ] ]
    Implemented Statistics: {meanExponent=0.0, negative=0, min=1.0, max=1.0, mean=0.2222222222222222, count=9.0, positive=2, stdDev=0.41573970964154905, zeros=7}
    Measured Feedback: [ [ 0.0, 0.0, 0.0 ], [ 0.0, 0.9999999999998899, 0.0 ], [ 0.0, 0.0, 0.9999999999998899 ] ]
    Measured Statistics: {meanExponent=-4.7830642341045674E-14, negative=0, min=0.9999999999998899, max=0.9999999999998899, mean=0.22222222222219776, count=9.0, positive=2, stdDev=0.41573970964150325, zeros=7}
    Feedback Error: [ [ 0.0, 0.0, 0.0 ], [ 0.0, -1.1013412404281553E-13, 0.0 ], [ 0.0, 0.0, -1.1013412404281553E-13 ] ]
    Error Statistics: {meanExponent=-12.958078098036825, negative=2, min=-1.1013412404281553E-13, max=-1.1013412404281553E-13, mean=-2.447424978729234E-14, count=9.0, positive=0, stdDev=4.5787128751186467E-14, zeros=7}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 2.4474e-14 +- 4.5787e-14 [0.0000e+00 - 1.1013e-13] (9#)
    relativeTol: 5.5067e-14 +- 0.0000e+00 [5.5067e-14 - 5.5067e-14] (2#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=2.4474e-14 +- 4.5787e-14 [0.0000e+00 - 1.1013e-13] (9#), relativeTol=5.5067e-14 +- 0.0000e+00 [5.5067e-14 - 5.5067e-14] (2#)}
```



### Performance
Code from [LayerTestBase.java:149](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L149) executed in 0.07 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 2.7153 +- 0.3322 [2.4052 - 3.9242]
    Learning performance: 1.3613 +- 0.7255 [1.0715 - 8.4553]
    
```

