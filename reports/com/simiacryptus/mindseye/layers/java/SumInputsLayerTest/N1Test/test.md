# SumInputsLayer
## N1Test
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
      "id": "e2a3bda5-e7e7-4c05-aeb3-4ede0000367a",
      "isFrozen": false,
      "name": "SumInputsLayer/e2a3bda5-e7e7-4c05-aeb3-4ede0000367a"
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
    [[ -0.492, -1.636, -0.2 ],
    [ 1.412 ]]
    --------------------
    Output: 
    [ 0.9199999999999999, -0.22399999999999998, 1.212 ]
```



### Batch Execution
Code from [LayerTestBase.java:178](../../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L178) executed in 0.00 seconds: 
```java
    BatchingTester batchingTester = getBatchingTester();
    return batchingTester == null ? null : batchingTester.test(layer, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (70#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (70#)}
```



### Differential Validation
Code from [LayerTestBase.java:186](../../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L186) executed in 0.00 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [ -0.492, -1.636, -0.2 ],
    [ 1.412 ]
    Inputs Statistics: {meanExponent=-0.2644072007444514, negative=3, min=-0.2, max=-0.2, mean=-0.7760000000000001, count=3.0, positive=0, stdDev=0.6196859419630774, zeros=0},
    {meanExponent=0.14983469671578492, negative=0, min=1.412, max=1.412, mean=1.412, count=1.0, positive=1, stdDev=0.0, zeros=0}
    Output: [ 0.9199999999999999, -0.22399999999999998, 1.212 ]
    Outputs Statistics: {meanExponent=-0.20082051149667154, negative=1, min=1.212, max=1.212, mean=0.636, count=3.0, positive=2, stdDev=0.6196859419630775, zeros=0}
    Feedback for input 0
    Inputs Values: [ -0.492, -1.636, -0.2 ]
    Value Statistics: {meanExponent=-0.2644072007444514, negative=3, min=-0.2, max=-0.2, mean=-0.7760000000000001, count=3.0, positive=0, stdDev=0.6196859419630774, zeros=0}
    Implemented Feedback: [ [ 1.0, 0.0, 0.0 ], [ 0.0, 1.0, 0.0 ], [ 0.0, 0.0, 1.0 ] ]
    Implemented Statistics: {meanExponent=0.0, negative=0, min=1.0, max=1.0, mean=0.3333333333333333, count=9.0, positive=3, stdDev=0.47140452079
```
...[skipping 796 bytes](etc/92.txt)...
```
    492, negative=0, min=1.412, max=1.412, mean=1.412, count=1.0, positive=1, stdDev=0.0, zeros=0}
    Implemented Feedback: [ [ 1.0, 1.0, 1.0 ] ]
    Implemented Statistics: {meanExponent=0.0, negative=0, min=1.0, max=1.0, mean=1.0, count=3.0, positive=3, stdDev=0.0, zeros=0}
    Measured Feedback: [ [ 0.9999999999998899, 0.9999999999998899, 0.9999999999998899 ] ]
    Measured Statistics: {meanExponent=-4.7830642341045674E-14, negative=0, min=0.9999999999998899, max=0.9999999999998899, mean=0.9999999999998899, count=3.0, positive=3, stdDev=0.0, zeros=0}
    Feedback Error: [ [ -1.1013412404281553E-13, -1.1013412404281553E-13, -1.1013412404281553E-13 ] ]
    Error Statistics: {meanExponent=-12.958078098036827, negative=3, min=-1.1013412404281553E-13, max=-1.1013412404281553E-13, mean=-1.1013412404281553E-13, count=3.0, positive=0, stdDev=0.0, zeros=0}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 5.5067e-14 +- 5.5067e-14 [0.0000e+00 - 1.1013e-13] (12#)
    relativeTol: 5.5067e-14 +- 0.0000e+00 [5.5067e-14 - 5.5067e-14] (6#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=5.5067e-14 +- 5.5067e-14 [0.0000e+00 - 1.1013e-13] (12#), relativeTol=5.5067e-14 +- 0.0000e+00 [5.5067e-14 - 5.5067e-14] (6#)}
```



### Performance
Code from [LayerTestBase.java:192](../../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L192) executed in 0.00 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 0.1936 +- 0.0853 [0.1197 - 0.8207]
    Learning performance: 0.0225 +- 0.0188 [0.0114 - 0.1282]
    
```

