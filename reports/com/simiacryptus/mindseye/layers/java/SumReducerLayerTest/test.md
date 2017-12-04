# SumReducerLayer
## SumReducerLayerTest
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
      "class": "com.simiacryptus.mindseye.layers.java.SumReducerLayer",
      "id": "370a9587-74a1-4959-b406-fa4500002cbb",
      "isFrozen": false,
      "name": "SumReducerLayer/370a9587-74a1-4959-b406-fa4500002cbb"
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
    [[ -1.008, -1.336, -1.216 ]]
    --------------------
    Output: 
    [ -3.5600000000000005 ]
```



### Batch Execution
Code from [LayerTestBase.java:138](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L138) executed in 0.00 seconds: 
```java
    BatchingTester batchingTester = getBatchingTester();
    return batchingTester==null?null:batchingTester.test(layer, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (40#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (40#)}
```



### Differential Validation
Code from [LayerTestBase.java:144](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.00 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [ -1.008, -1.336, -1.216 ]
    Inputs Statistics: {meanExponent=0.07140018839524982, negative=3, min=-1.216, max=-1.216, mean=-1.1866666666666668, count=3.0, positive=0, stdDev=0.1355023575030666, zeros=0}
    Output: [ -3.5600000000000005 ]
    Outputs Statistics: {meanExponent=0.5514499979728752, negative=1, min=-3.5600000000000005, max=-3.5600000000000005, mean=-3.5600000000000005, count=1.0, positive=0, stdDev=0.0, zeros=0}
    Feedback for input 0
    Inputs Values: [ -1.008, -1.336, -1.216 ]
    Value Statistics: {meanExponent=0.07140018839524982, negative=3, min=-1.216, max=-1.216, mean=-1.1866666666666668, count=3.0, positive=0, stdDev=0.1355023575030666, zeros=0}
    Implemented Feedback: [ [ 1.0 ], [ 1.0 ], [ 1.0 ] ]
    Implemented Statistics: {meanExponent=0.0, negative=0, min=1.0, max=1.0, mean=1.0, count=3.0, positive=3, stdDev=0.0, zeros=0}
    Measured Feedback: [ [ 1.0000000000065512 ], [ 1.0000000000065512 ], [ 1.0000000000021103 ] ]
    Measured Statistics: {meanExponent=2.2022667796100914E-12, negative=0, min=1.0000000000021103, max=1.0000000000021103, mean=1.0000000000050708, count=3.0, positive=3, stdDev=1.4901161193847656E-8, zeros=0}
    Feedback Error: [ [ 6.551204023708124E-12 ], [ 6.551204023708124E-12 ], [ 2.1103119252074976E-12 ] ]
    Error Statistics: {meanExponent=-11.347670365732291, negative=0, min=2.1103119252074976E-12, max=2.1103119252074976E-12, mean=5.070906657541248E-12, count=3.0, positive=3, stdDev=2.093456611578367E-12, zeros=0}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 5.0709e-12 +- 2.0935e-12 [2.1103e-12 - 6.5512e-12] (3#)
    relativeTol: 2.5355e-12 +- 1.0467e-12 [1.0552e-12 - 3.2756e-12] (3#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=5.0709e-12 +- 2.0935e-12 [2.1103e-12 - 6.5512e-12] (3#), relativeTol=2.5355e-12 +- 1.0467e-12 [1.0552e-12 - 3.2756e-12] (3#)}
```



### Performance
Code from [LayerTestBase.java:149](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L149) executed in 0.00 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 0.1730 +- 0.0395 [0.1140 - 0.3420]
    Learning performance: 0.0026 +- 0.0015 [0.0000 - 0.0086]
    
```

