# SumInputsLayer
## NNTest
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
      "class": "com.simiacryptus.mindseye.layers.java.SumInputsLayer",
      "id": "370a9587-74a1-4959-b406-fa4500002cb4",
      "isFrozen": false,
      "name": "SumInputsLayer/370a9587-74a1-4959-b406-fa4500002cb4"
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
    [[ 0.456, 1.044, 1.888 ],
    [ -1.256, -1.04, 1.404 ]]
    --------------------
    Output: 
    [ -0.8, 0.0040000000000000036, 3.292 ]
```



### Batch Execution
Code from [LayerTestBase.java:138](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L138) executed in 0.00 seconds: 
```java
    BatchingTester batchingTester = getBatchingTester();
    return batchingTester==null?null:batchingTester.test(layer, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (90#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (90#)}
```



### Differential Validation
Code from [LayerTestBase.java:144](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.00 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [ 0.456, 1.044, 1.888 ],
    [ -1.256, -1.04, 1.404 ]
    Inputs Statistics: {meanExponent=-0.015444222902423829, negative=0, min=1.888, max=1.888, mean=1.1293333333333333, count=3.0, positive=3, stdDev=0.5877172411590079, zeros=0},
    {meanExponent=0.0877966954979147, negative=2, min=1.404, max=1.404, mean=-0.29733333333333345, count=3.0, positive=1, stdDev=1.2062518624050103, zeros=0}
    Output: [ -0.8, 0.0040000000000000036, 3.292 ]
    Outputs Statistics: {meanExponent=-0.6591300650466203, negative=1, min=3.292, max=3.292, mean=0.8319999999999999, count=3.0, positive=2, stdDev=1.7701796518997726, zeros=0}
    Feedback for input 0
    Inputs Values: [ 0.456, 1.044, 1.888 ]
    Value Statistics: {meanExponent=-0.015444222902423829, negative=0, min=1.888, max=1.888, mean=1.1293333333333333, count=3.0, positive=3, stdDev=0.5877172411590079, zeros=0}
    Implemented Feedback: [ [ 1.0, 0.0, 0.0 ], [ 0.0, 1.0, 0.0 ], [ 0.0, 0.0, 1.0 ] ]
    Implemented Statistics: {meanExponent=0.0, negative=0, min=1.0, max=1.0, mean=0.3333333333333333, count=9.0, positive=3, stdDev=0.4714045207910317, zeros=6}
    Measured Feedback: [ [ 0.9999999999998899, 0.0, 0.0 ], [ 0.0, 0.9999999999998899, 0.0 ], [ 0.0, 0.0, 0.9999999999976694 ] ]
    Measured Statistics: {meanExponent=-3.692731311925336E-13, negative=0, min=0.9999999999976694, max=0.9999999999976694, mean=0.33333333333304993, count=9.0, positive=3, stdDev=0.47140452079063083, zeros=6}
    Feedback Error: [ [ -1.1013412404281553E-13, 0.0, 0.0 ], [ 0.0, -1.1013412404281553E-13, 0.0 ], [ 0.0, 0.0, -2.3305801732931286E-12 ] ]
    Error Statistics: {meanExponent=-12.516230716189696, negative=3, min=-2.3305801732931286E-12, max=-2.3305801732931286E-12, mean=-2.834276023754177E-13, count=9.0, positive=0, stdDev=7.251729404930389E-13, zeros=6}
    Feedback for input 1
    Inputs Values: [ -1.256, -1.04, 1.404 ]
    Value Statistics: {meanExponent=0.0877966954979147, negative=2, min=1.404, max=1.404, mean=-0.29733333333333345, count=3.0, positive=1, stdDev=1.2062518624050103, zeros=0}
    Implemented Feedback: [ [ 1.0, 0.0, 0.0 ], [ 0.0, 1.0, 0.0 ], [ 0.0, 0.0, 1.0 ] ]
    Implemented Statistics: {meanExponent=0.0, negative=0, min=1.0, max=1.0, mean=0.3333333333333333, count=9.0, positive=3, stdDev=0.4714045207910317, zeros=6}
    Measured Feedback: [ [ 0.9999999999998899, 0.0, 0.0 ], [ 0.0, 0.9999999999998899, 0.0 ], [ 0.0, 0.0, 0.9999999999976694 ] ]
    Measured Statistics: {meanExponent=-3.692731311925336E-13, negative=0, min=0.9999999999976694, max=0.9999999999976694, mean=0.33333333333304993, count=9.0, positive=3, stdDev=0.47140452079063083, zeros=6}
    Feedback Error: [ [ -1.1013412404281553E-13, 0.0, 0.0 ], [ 0.0, -1.1013412404281553E-13, 0.0 ], [ 0.0, 0.0, -2.3305801732931286E-12 ] ]
    Error Statistics: {meanExponent=-12.516230716189696, negative=3, min=-2.3305801732931286E-12, max=-2.3305801732931286E-12, mean=-2.834276023754177E-13, count=9.0, positive=0, stdDev=7.251729404930389E-13, zeros=6}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 2.8343e-13 +- 7.2517e-13 [0.0000e+00 - 2.3306e-12] (18#)
    relativeTol: 4.2514e-13 +- 5.2336e-13 [5.5067e-14 - 1.1653e-12] (6#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=2.8343e-13 +- 7.2517e-13 [0.0000e+00 - 2.3306e-12] (18#), relativeTol=4.2514e-13 +- 5.2336e-13 [5.5067e-14 - 1.1653e-12] (6#)}
```



### Performance
Code from [LayerTestBase.java:149](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L149) executed in 0.00 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 0.2308 +- 0.0476 [0.1567 - 0.4645]
    Learning performance: 0.0183 +- 0.0086 [0.0114 - 0.0826]
    
```

