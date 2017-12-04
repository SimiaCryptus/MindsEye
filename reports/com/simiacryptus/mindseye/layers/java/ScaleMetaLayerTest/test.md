# ScaleMetaLayer
## ScaleMetaLayerTest
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
      "class": "com.simiacryptus.mindseye.layers.java.ScaleMetaLayer",
      "id": "370a9587-74a1-4959-b406-fa4500002c86",
      "isFrozen": false,
      "name": "ScaleMetaLayer/370a9587-74a1-4959-b406-fa4500002c86"
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
    [[ 1.42, -1.076, -0.592 ],
    [ 1.76, 0.616, -1.648 ]]
    --------------------
    Output: 
    [ 2.4992, -0.6628160000000001, 0.9756159999999999 ]
```



### Batch Execution
Code from [LayerTestBase.java:138](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L138) executed in 0.00 seconds: 
```java
    BatchingTester batchingTester = getBatchingTester();
    return batchingTester==null?null:batchingTester.test(layer, inputPrototype);
```

### Differential Validation
Code from [LayerTestBase.java:144](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.00 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [ 1.42, -1.076, -0.592 ],
    [ 1.76, 0.616, -1.648 ]
    Inputs Statistics: {meanExponent=-0.014525892521217795, negative=2, min=-0.592, max=-0.592, mean=-0.08266666666666671, count=3.0, positive=1, stdDev=1.0807618711919025, zeros=0},
    {meanExponent=0.08401686244655741, negative=1, min=-1.648, max=-1.648, mean=0.24266666666666667, count=3.0, positive=2, stdDev=1.4161330759815225, zeros=0}
    Output: [ 2.4992, -0.6628160000000001, 0.9756159999999999 ]
    Outputs Statistics: {meanExponent=0.06949096992533964, negative=1, min=0.9756159999999999, max=0.9756159999999999, mean=0.9373333333333332, count=3.0, positive=2, stdDev=1.2911714236793743, zeros=0}
    Feedback for input 0
    Inputs Values: [ 1.42, -1.076, -0.592 ]
    Value Statistics: {meanExponent=-0.014525892521217795, negative=2, min=-0.592, max=-0.592, mean=-0.08266666666666671, count=3.0, positive=1, stdDev=1.0807618711919025, zeros=0}
    Implemented Feedback: [ [ 1.76, 0.0, 0.0 ], [ 0.0, 0.616, 0.0 ], [ 0.0, 0.0, -1.648 ] ]
    Implemented Statistics: {meanExponent=0.08401686244655741, negative=1, min=-1.648, max=-1.648, mean=0.08088888888888889, count=9.0, positive=2, stdDev=0.8255686854047867, zeros=6}
    Measured Feedback: [ [ 1.7599999999973193, 0.0, 0.0 ], [ 0.0, 0.6159999999999499, 0.0 ], [ 0.0, 0.0, -1.6479999999996497 ] ]
    Measured Statistics: {meanExponent=0.0840168624462944, negative=1, min=-1.6479999999996497, max=-1.6479999999996497, mean=0.08088888888862439, count=9.0, positive=2, stdDev=0.8255686854040958, zeros=6}
    Feedback Error: [ [ -2.680744515259903E-12, 0.0, 0.0 ], [ 0.0, -5.007105841059456E-14, 0.0 ], [ 0.0, 0.0, 3.5016434196677437E-13 ] ]
    Error Statistics: {meanExponent=-12.442628627801968, negative=2, min=3.5016434196677437E-13, max=3.5016434196677437E-13, mean=-2.645168035226359E-13, count=9.0, positive=1, stdDev=8.616386893755241E-13, zeros=6}
    Feedback for input 1
    Inputs Values: [ 1.76, 0.616, -1.648 ]
    Value Statistics: {meanExponent=0.08401686244655741, negative=1, min=-1.648, max=-1.648, mean=0.24266666666666667, count=3.0, positive=2, stdDev=1.4161330759815225, zeros=0}
    Implemented Feedback: [ [ 1.42, 0.0, 0.0 ], [ 0.0, -1.076, 0.0 ], [ 0.0, 0.0, -0.592 ] ]
    Implemented Statistics: {meanExponent=-0.014525892521217795, negative=2, min=-0.592, max=-0.592, mean=-0.02755555555555557, count=9.0, positive=1, stdDev=0.6251938563555238, zeros=6}
    Measured Feedback: [ [ 1.4199999999986446, 0.0, 0.0 ], [ 0.0, -1.0759999999998549, 0.0 ], [ 0.0, 0.0, -0.59200000000037 ] ]
    Measured Statistics: {meanExponent=-0.014525892521285019, negative=2, min=-0.59200000000037, max=-0.59200000000037, mean=-0.027555555555731144, count=9.0, positive=1, stdDev=0.6251938563551852, zeros=6}
    Feedback Error: [ [ -1.3553602684623911E-12, 0.0, 0.0 ], [ 0.0, 1.4521717162097048E-13, 0.0 ], [ 0.0, 0.0, -3.700373341075647E-13 ] ]
    Error Statistics: {meanExponent=-12.379227244104476, negative=2, min=-3.700373341075647E-13, max=-3.700373341075647E-13, mean=-1.7557560343877613E-13, count=9.0, positive=1, stdDev=4.3685441004570885E-13, zeros=6}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 2.7509e-13 +- 6.6435e-13 [0.0000e+00 - 2.6807e-12] (18#)
    relativeTol: 2.9428e-13 +- 2.5926e-13 [4.0642e-14 - 7.6158e-13] (6#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=2.7509e-13 +- 6.6435e-13 [0.0000e+00 - 2.6807e-12] (18#), relativeTol=2.9428e-13 +- 2.5926e-13 [4.0642e-14 - 7.6158e-13] (6#)}
```



### Performance
Code from [LayerTestBase.java:149](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L149) executed in 0.00 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 0.1848 +- 0.0461 [0.1197 - 0.3505]
    Learning performance: 0.0019 +- 0.0015 [0.0000 - 0.0057]
    
```

