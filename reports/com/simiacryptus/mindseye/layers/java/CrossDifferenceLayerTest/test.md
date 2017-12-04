# CrossDifferenceLayer
## CrossDifferenceLayerTest
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
      "class": "com.simiacryptus.mindseye.layers.java.CrossDifferenceLayer",
      "id": "370a9587-74a1-4959-b406-fa4500002b96",
      "isFrozen": false,
      "name": "CrossDifferenceLayer/370a9587-74a1-4959-b406-fa4500002b96"
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
    [[ -0.836, -1.4, -1.656, -1.968 ]]
    --------------------
    Output: 
    [ 0.564, 0.82, 1.1320000000000001, 0.256, 0.5680000000000001, 0.31200000000000006 ]
```



### Batch Execution
Code from [LayerTestBase.java:138](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L138) executed in 0.00 seconds: 
```java
    BatchingTester batchingTester = getBatchingTester();
    return batchingTester==null?null:batchingTester.test(layer, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (100#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (100#)}
```



### Differential Validation
Code from [LayerTestBase.java:144](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.00 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [ -0.836, -1.4, -1.656, -1.968 ]
    Inputs Statistics: {meanExponent=0.1453549349153596, negative=4, min=-1.968, max=-1.968, mean=-1.4649999999999999, count=4.0, positive=0, stdDev=0.41513732667636677, zeros=0}
    Output: [ 0.564, 0.82, 1.1320000000000001, 0.256, 0.5680000000000001, 0.31200000000000006 ]
    Outputs Statistics: {meanExponent=-0.27071962028989616, negative=0, min=0.31200000000000006, max=0.31200000000000006, mean=0.6086666666666667, count=6.0, positive=6, stdDev=0.2984887863145877, zeros=0}
    Feedback for input 0
    Inputs Values: [ -0.836, -1.4, -1.656, -1.968 ]
    Value Statistics: {meanExponent=0.1453549349153596, negative=4, min=-1.968, max=-1.968, mean=-1.4649999999999999, count=4.0, positive=0, stdDev=0.41513732667636677, zeros=0}
    Implemented Feedback: [ [ 1.0, 1.0, 1.0, 0.0, 0.0, 0.0 ], [ -1.0, 0.0, 0.0, 1.0, 1.0, 0.0 ], [ 0.0, -1.0, 0.0, -1.0, 0.0, 1.0 ], [ 0.0, 0.0, -1.0, 0.0, -1.0, -1.0 ] ]
    Implemented Statistics: {meanExponent=0.0, negative=6, min=-1.0, max=-1.0, mean=0.0, count=24.0, positive=6, stdDev=0.7071067811865476, zeros=12}
    Measured Feedback: [ [ 0.9999999999998899, 0.9999999999998899, 0.9999999999976694, 0.0, 0.0, 0.0 ], [ -0.9999999999998899, 0.0, 0.0, 0.9999999999998899, 0.9999999999998899, 0.0 ], [ 0.0, -0.9999999999998899, 0.0, -0.9999999999998899, 0.0, 0.9999999999998899 ], [ 0.0, 0.0, -1.0000000000021103, 0.0, -0.9999999999998899, -0.9999999999998899 ] ]
    Measured Statistics: {meanExponent=-4.78306423412241E-14, negative=6, min=-0.9999999999998899, max=-0.9999999999998899, mean=-1.8503717077085943E-13, count=24.0, positive=6, stdDev=0.7071067811864696, zeros=12}
    Feedback Error: [ [ -1.1013412404281553E-13, -1.1013412404281553E-13, -2.3305801732931286E-12, 0.0, 0.0, 0.0 ], [ 1.1013412404281553E-13, 0.0, 0.0, -1.1013412404281553E-13, -1.1013412404281553E-13, 0.0 ], [ 0.0, 1.1013412404281553E-13, 0.0, 1.1013412404281553E-13, 0.0, -1.1013412404281553E-13 ], [ 0.0, 0.0, -2.1103119252074976E-12, 0.0, 1.1013412404281553E-13, 1.1013412404281553E-13 ] ]
    Error Statistics: {meanExponent=-12.740747523312798, negative=7, min=1.1013412404281553E-13, max=1.1013412404281553E-13, mean=-1.8503717077085943E-13, count=24.0, positive=5, stdDev=6.186202897376268E-13, zeros=12}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 2.3093e-13 +- 6.0299e-13 [0.0000e+00 - 2.3306e-12] (24#)
    relativeTol: 2.3093e-13 +- 3.9388e-13 [5.5067e-14 - 1.1653e-12] (12#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=2.3093e-13 +- 6.0299e-13 [0.0000e+00 - 2.3306e-12] (24#), relativeTol=2.3093e-13 +- 3.9388e-13 [5.5067e-14 - 1.1653e-12] (12#)}
```



### Performance
Code from [LayerTestBase.java:149](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L149) executed in 0.00 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 0.2202 +- 0.0861 [0.1510 - 0.6612]
    Learning performance: 0.0037 +- 0.0029 [0.0000 - 0.0228]
    
```

