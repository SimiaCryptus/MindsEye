# BiasMetaLayer
## BiasMetaLayerTest
### Json Serialization
Code from [LayerTestBase.java:83](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L83) executed in 0.00 seconds: 
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
      "class": "com.simiacryptus.mindseye.layers.java.BiasMetaLayer",
      "id": "a864e734-2f23-44db-97c1-504000002b8e",
      "isFrozen": false,
      "name": "BiasMetaLayer/a864e734-2f23-44db-97c1-504000002b8e"
    }
```



### Example Input/Output Pair
Code from [LayerTestBase.java:120](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L120) executed in 0.00 seconds: 
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
    [[ 0.892, 1.728, -0.12 ],
    [ -1.136, 0.596, -0.124 ]]
    --------------------
    Output: 
    [ -0.24399999999999988, 2.324, -0.244 ]
```



### Batch Execution
Code from [LayerTestBase.java:138](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L138) executed in 0.00 seconds: 
```java
    BatchingTester batchingTester = getBatchingTester();
    return batchingTester==null?null:batchingTester.test(layer, inputPrototype);
```

### Differential Validation
Code from [LayerTestBase.java:144](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.00 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [ 0.892, 1.728, -0.12 ],
    [ -1.136, 0.596, -0.124 ]
    Inputs Statistics: {meanExponent=-0.24430338714445923, negative=1, min=-0.12, max=-0.12, mean=0.8333333333333334, count=3.0, positive=2, stdDev=0.755582483180287, zeros=0},
    {meanExponent=-0.35865124124084286, negative=2, min=-0.124, max=-0.124, mean=-0.2213333333333333, count=3.0, positive=1, stdDev=0.7104277271866264, zeros=0}
    Output: [ -0.24399999999999988, 2.324, -0.244 ]
    Outputs Statistics: {meanExponent=-0.2863280745347494, negative=2, min=-0.244, max=-0.244, mean=0.612, count=3.0, positive=1, stdDev=1.210566809391369, zeros=0}
    Feedback for input 0
    Inputs Values: [ 0.892, 1.728, -0.12 ]
    Value Statistics: {meanExponent=-0.24430338714445923, negative=1, min=-0.12, max=-0.12, mean=0.8333333333333334, count=3.0, positive=2, stdDev=0.755582483180287, zeros=0}
    Implemented Feedback: [ [ 1.0, 0.0, 0.0 ], [ 0.0, 1.0, 0.0 ], [ 0.0, 0.0, 1.0 ] ]
    Implemented Statistics: {meanExponent=0.0, negative=0, min=1.0, max=1.0, mean=0.3333333333333333, count=9.0, positive=3, stdDev=0.4714045207910317, zeros=6}
    Measured Feedback: [ [ 0.9999999999998899, 0.0, 0.0 ], [ 0.0, 1.0000000000021103, 0.0 ], [ 0.0, 0.0, 0.9999999999998899 ] ]
    Measured Statistics: {meanExponent=2.7361184650972856E-13, negative=0, min=0.9999999999998899, max=0.9999999999998899, mean=0.3333333333335433, count=9.0, positive=3, stdDev=0.4714045207913287, zeros=6}
    Feedback Error: [ [ -1.1013412404281553E-13, 0.0, 0.0 ], [ 0.0, 2.1103119252074976E-12, 0.0 ], [ 0.0, 0.0, -1.1013412404281553E-13 ] ]
    Error Statistics: {meanExponent=-12.530603180987853, negative=2, min=-1.1013412404281553E-13, max=-1.1013412404281553E-13, mean=2.1000485301354073E-13, count=9.0, positive=1, stdDev=6.733627986644662E-13, zeros=6}
    Feedback for input 1
    Inputs Values: [ -1.136, 0.596, -0.124 ]
    Value Statistics: {meanExponent=-0.35865124124084286, negative=2, min=-0.124, max=-0.124, mean=-0.2213333333333333, count=3.0, positive=1, stdDev=0.7104277271866264, zeros=0}
    Implemented Feedback: [ [ 1.0, 0.0, 0.0 ], [ 0.0, 1.0, 0.0 ], [ 0.0, 0.0, 1.0 ] ]
    Implemented Statistics: {meanExponent=0.0, negative=0, min=1.0, max=1.0, mean=0.3333333333333333, count=9.0, positive=3, stdDev=0.4714045207910317, zeros=6}
    Measured Feedback: [ [ 0.9999999999998899, 0.0, 0.0 ], [ 0.0, 1.0000000000021103, 0.0 ], [ 0.0, 0.0, 0.9999999999998899 ] ]
    Measured Statistics: {meanExponent=2.7361184650972856E-13, negative=0, min=0.9999999999998899, max=0.9999999999998899, mean=0.3333333333335433, count=9.0, positive=3, stdDev=0.4714045207913287, zeros=6}
    Feedback Error: [ [ -1.1013412404281553E-13, 0.0, 0.0 ], [ 0.0, 2.1103119252074976E-12, 0.0 ], [ 0.0, 0.0, -1.1013412404281553E-13 ] ]
    Error Statistics: {meanExponent=-12.530603180987853, negative=2, min=-1.1013412404281553E-13, max=-1.1013412404281553E-13, mean=2.1000485301354073E-13, count=9.0, positive=1, stdDev=6.733627986644662E-13, zeros=6}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 2.5895e-13 +- 6.5610e-13 [0.0000e+00 - 2.1103e-12] (18#)
    relativeTol: 3.8843e-13 +- 4.7145e-13 [5.5067e-14 - 1.0552e-12] (6#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=2.5895e-13 +- 6.5610e-13 [0.0000e+00 - 2.1103e-12] (18#), relativeTol=3.8843e-13 +- 4.7145e-13 [5.5067e-14 - 1.0552e-12] (6#)}
```



### Performance
Code from [LayerTestBase.java:149](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L149) executed in 0.00 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 0.1926 +- 0.0504 [0.1567 - 0.4902]
    Learning performance: 0.0034 +- 0.0024 [0.0000 - 0.0228]
    
```

