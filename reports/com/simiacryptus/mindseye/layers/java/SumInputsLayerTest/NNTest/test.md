# SumInputsLayer
## NNTest
### Json Serialization
Code from [LayerTestBase.java:121](../../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L121) executed in 0.00 seconds: 
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
      "id": "e2a3bda5-e7e7-4c05-aeb3-4ede00003681",
      "isFrozen": false,
      "name": "SumInputsLayer/e2a3bda5-e7e7-4c05-aeb3-4ede00003681"
    }
```



### Example Input/Output Pair
Code from [LayerTestBase.java:159](../../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L159) executed in 0.00 seconds: 
```java
    SimpleEval eval = SimpleEval.run(layer, inputPrototype);
    return String.format("--------------------\nInput: \n[%s]\n--------------------\nOutput: \n%s",
      Arrays.stream(inputPrototype).map(t -> t.prettyPrint()).reduce((a, b) -> a + ",\n" + b).get(),
      eval.getOutput().prettyPrint());
```

Returns: 

```
    --------------------
    Input: 
    [[ -0.368, 1.3, 1.576 ],
    [ 0.388, 1.72, -0.268 ]]
    --------------------
    Output: 
    [ 0.020000000000000018, 3.02, 1.308 ]
```



### Batch Execution
Code from [LayerTestBase.java:178](../../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L178) executed in 0.00 seconds: 
```java
    BatchingTester batchingTester = getBatchingTester();
    return batchingTester == null ? null : batchingTester.test(layer, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (90#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (90#)}
```



### Differential Validation
Code from [LayerTestBase.java:186](../../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L186) executed in 0.00 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [ -0.368, 1.3, 1.576 ],
    [ 0.388, 1.72, -0.268 ]
    Inputs Statistics: {meanExponent=-0.04088420528870302, negative=1, min=1.576, max=1.576, mean=0.836, count=3.0, positive=2, stdDev=0.8587805307527646, zeros=0},
    {meanExponent=-0.2491683444898183, negative=1, min=-0.268, max=-0.268, mean=0.6133333333333334, count=3.0, positive=2, stdDev=0.8270902140771719, zeros=0}
    Output: [ 0.020000000000000018, 3.02, 1.308 ]
    Outputs Statistics: {meanExponent=-0.36745177246353977, negative=0, min=1.308, max=1.308, mean=1.4493333333333334, count=3.0, positive=3, stdDev=1.228815509161386, zeros=0}
    Feedback for input 0
    Inputs Values: [ -0.368, 1.3, 1.576 ]
    Value Statistics: {meanExponent=-0.04088420528870302, negative=1, min=1.576, max=1.576, mean=0.836, count=3.0, positive=2, stdDev=0.8587805307527646, zeros=0}
    Implemented Feedback: [ [ 1.0, 0.0, 0.0 ], [ 0.0, 1.0, 0.0 ], [ 0.0, 0.0, 1.0 ] ]
    Implemented Statistics: {meanExponent=0.0, negative=0, min=1.0, max=1.0, mean=0.3333333333333333, count=9.0, positive=3, stdD
```
...[skipping 1027 bytes](etc/93.txt)...
```
    ] ]
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
Code from [LayerTestBase.java:192](../../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L192) executed in 0.00 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 0.2113 +- 0.1922 [0.1510 - 1.7583]
    Learning performance: 0.0129 +- 0.0214 [0.0057 - 0.2194]
    
```

