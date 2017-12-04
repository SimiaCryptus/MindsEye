# SumInputsLayer
## NNTest
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
      "class": "com.simiacryptus.mindseye.layers.java.SumInputsLayer",
      "id": "a864e734-2f23-44db-97c1-504000002cb4",
      "isFrozen": false,
      "name": "SumInputsLayer/a864e734-2f23-44db-97c1-504000002cb4"
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
    [[ -1.324, 1.548, 0.712 ],
    [ -1.7, 0.508, -0.288 ]]
    --------------------
    Output: 
    [ -3.024, 2.056, 0.424 ]
```



### Batch Execution
Code from [LayerTestBase.java:138](../../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L138) executed in 0.00 seconds: 
```java
    BatchingTester batchingTester = getBatchingTester();
    return batchingTester==null?null:batchingTester.test(layer, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (90#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (90#)}
```



### Differential Validation
Code from [LayerTestBase.java:144](../../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.00 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [ -1.324, 1.548, 0.712 ],
    [ -1.7, 0.508, -0.288 ]
    Inputs Statistics: {meanExponent=0.05471297836247043, negative=1, min=0.712, max=0.712, mean=0.312, count=3.0, positive=2, stdDev=1.206122160755977, zeros=0},
    {meanExponent=-0.20143162619285868, negative=2, min=-0.288, max=-0.288, mean=-0.49333333333333335, count=3.0, positive=1, stdDev=0.9130306067645754, zeros=0}
    Output: [ -3.024, 2.056, 0.424 ]
    Outputs Statistics: {meanExponent=0.1403235845817132, negative=1, min=0.424, max=0.424, mean=-0.18133333333333335, count=3.0, positive=2, stdDev=2.117612072332628, zeros=0}
    Feedback for input 0
    Inputs Values: [ -1.324, 1.548, 0.712 ]
    Value Statistics: {meanExponent=0.05471297836247043, negative=1, min=0.712, max=0.712, mean=0.312, count=3.0, positive=2, stdDev=1.206122160755977, zeros=0}
    Implemented Feedback: [ [ 1.0, 0.0, 0.0 ], [ 0.0, 1.0, 0.0 ], [ 0.0, 0.0, 1.0 ] ]
    Implemented Statistics: {meanExponent=0.0, negative=0, min=1.0, max=1.0, mean=0.3333333333333333, count=9.0, positive=3, stdDev=0.4714045207910317, zeros=6}
    Measured Feedback: [ [ 0.9999999999976694, 0.0, 0.0 ], [ 0.0, 0.9999999999976694, 0.0 ], [ 0.0, 0.0, 0.9999999999998899 ] ]
    Measured Statistics: {meanExponent=-6.907156200440216E-13, negative=0, min=0.9999999999998899, max=0.9999999999998899, mean=0.3333333333328032, count=9.0, positive=3, stdDev=0.47140452079028194, zeros=6}
    Feedback Error: [ [ -2.3305801732931286E-12, 0.0, 0.0 ], [ 0.0, -2.3305801732931286E-12, 0.0 ], [ 0.0, 0.0, -1.1013412404281553E-13 ] ]
    Error Statistics: {meanExponent=-12.074383334342565, negative=3, min=-1.1013412404281553E-13, max=-1.1013412404281553E-13, mean=-5.30143830069897E-13, count=9.0, positive=0, stdDev=9.62973698067151E-13, zeros=6}
    Feedback for input 1
    Inputs Values: [ -1.7, 0.508, -0.288 ]
    Value Statistics: {meanExponent=-0.20143162619285868, negative=2, min=-0.288, max=-0.288, mean=-0.49333333333333335, count=3.0, positive=1, stdDev=0.9130306067645754, zeros=0}
    Implemented Feedback: [ [ 1.0, 0.0, 0.0 ], [ 0.0, 1.0, 0.0 ], [ 0.0, 0.0, 1.0 ] ]
    Implemented Statistics: {meanExponent=0.0, negative=0, min=1.0, max=1.0, mean=0.3333333333333333, count=9.0, positive=3, stdDev=0.4714045207910317, zeros=6}
    Measured Feedback: [ [ 0.9999999999976694, 0.0, 0.0 ], [ 0.0, 0.9999999999976694, 0.0 ], [ 0.0, 0.0, 0.9999999999998899 ] ]
    Measured Statistics: {meanExponent=-6.907156200440216E-13, negative=0, min=0.9999999999998899, max=0.9999999999998899, mean=0.3333333333328032, count=9.0, positive=3, stdDev=0.47140452079028194, zeros=6}
    Feedback Error: [ [ -2.3305801732931286E-12, 0.0, 0.0 ], [ 0.0, -2.3305801732931286E-12, 0.0 ], [ 0.0, 0.0, -1.1013412404281553E-13 ] ]
    Error Statistics: {meanExponent=-12.074383334342565, negative=3, min=-1.1013412404281553E-13, max=-1.1013412404281553E-13, mean=-5.30143830069897E-13, count=9.0, positive=0, stdDev=9.62973698067151E-13, zeros=6}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 5.3014e-13 +- 9.6297e-13 [0.0000e+00 - 2.3306e-12] (18#)
    relativeTol: 7.9522e-13 +- 5.2336e-13 [5.5067e-14 - 1.1653e-12] (6#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=5.3014e-13 +- 9.6297e-13 [0.0000e+00 - 2.3306e-12] (18#), relativeTol=7.9522e-13 +- 5.2336e-13 [5.5067e-14 - 1.1653e-12] (6#)}
```



### Performance
Code from [LayerTestBase.java:149](../../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L149) executed in 0.00 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 0.2062 +- 0.0412 [0.1624 - 0.4987]
    Learning performance: 0.0151 +- 0.0119 [0.0085 - 0.0997]
    
```

