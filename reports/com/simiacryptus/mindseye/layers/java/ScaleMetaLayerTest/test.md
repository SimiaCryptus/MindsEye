# ScaleMetaLayer
## ScaleMetaLayerTest
### Json Serialization
Code from [LayerTestBase.java:121](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L121) executed in 0.00 seconds: 
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
      "id": "e2a3bda5-e7e7-4c05-aeb3-4ede00003615",
      "isFrozen": false,
      "name": "ScaleMetaLayer/e2a3bda5-e7e7-4c05-aeb3-4ede00003615"
    }
```



### Example Input/Output Pair
Code from [LayerTestBase.java:159](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L159) executed in 0.00 seconds: 
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
    [[ 1.9, 1.924, 1.164 ],
    [ 0.216, 1.664, -0.42 ]]
    --------------------
    Output: 
    [ 0.4104, 3.201536, -0.4888799999999999 ]
```



### Batch Execution
Code from [LayerTestBase.java:178](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L178) executed in 0.00 seconds: 
```java
    BatchingTester batchingTester = getBatchingTester();
    return batchingTester == null ? null : batchingTester.test(layer, inputPrototype);
```

### Differential Validation
Code from [LayerTestBase.java:186](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L186) executed in 0.00 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [ 1.9, 1.924, 1.164 ],
    [ 0.216, 1.664, -0.42 ]
    Inputs Statistics: {meanExponent=0.20963721632283092, negative=0, min=1.164, max=1.164, mean=1.6626666666666665, count=3.0, positive=3, stdDev=0.35274668279407345, zeros=0},
    {meanExponent=-0.2737145454988212, negative=1, min=-0.42, max=-0.42, mean=0.48666666666666664, count=3.0, positive=2, stdDev=0.8720509669101278, zeros=0}
    Output: [ 0.4104, 3.201536, -0.4888799999999999 ]
    Outputs Statistics: {meanExponent=-0.06407732917599025, negative=1, min=-0.4888799999999999, max=-0.4888799999999999, mean=1.0410186666666668, count=3.0, positive=2, stdDev=1.5712102533451364, zeros=0}
    Feedback for input 0
    Inputs Values: [ 1.9, 1.924, 1.164 ]
    Value Statistics: {meanExponent=0.20963721632283092, negative=0, min=1.164, max=1.164, mean=1.6626666666666665, count=3.0, positive=3, stdDev=0.35274668279407345, zeros=0}
    Implemented Feedback: [ [ 0.216, 0.0, 0.0 ], [ 0.0, 1.664, 0.0 ], [ 0.0, 0.0, -0.42 ] ]
    Implemented Statistics: {meanExponent=-0.2737145454988212, nega
```
...[skipping 1114 bytes](etc/83.txt)...
```
    Implemented Statistics: {meanExponent=0.20963721632283092, negative=0, min=1.164, max=1.164, mean=0.5542222222222222, count=9.0, positive=3, stdDev=0.809815586384096, zeros=6}
    Measured Feedback: [ [ 1.8999999999996797, 0.0, 0.0 ], [ 0.0, 1.9239999999998147, 0.0 ], [ 0.0, 0.0, 1.1639999999996098 ] ]
    Measured Statistics: {meanExponent=0.20963721632274404, negative=0, min=1.1639999999996098, max=1.1639999999996098, mean=0.5542222222221227, count=9.0, positive=3, stdDev=0.8098155863839693, zeros=6}
    Feedback Error: [ [ -3.2018832030189515E-13, 0.0, 0.0 ], [ 0.0, -1.851852005074761E-13, 0.0 ], [ 0.0, 0.0, -3.9013237085328E-13 ] ]
    Error Statistics: {meanExponent=-12.545258750354874, negative=3, min=-3.9013237085328E-13, max=-3.9013237085328E-13, mean=-9.950065462918347E-14, count=9.0, positive=0, stdDev=1.4903913106132E-13, zeros=6}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 1.3449e-13 +- 2.5293e-13 [0.0000e+00 - 1.0003e-12] (18#)
    relativeTol: 2.2394e-13 +- 1.5060e-13 [4.8125e-14 - 5.0006e-13] (6#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=1.3449e-13 +- 2.5293e-13 [0.0000e+00 - 1.0003e-12] (18#), relativeTol=2.2394e-13 +- 1.5060e-13 [4.8125e-14 - 5.0006e-13] (6#)}
```



### Performance
Code from [LayerTestBase.java:192](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L192) executed in 0.00 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 0.1838 +- 0.1321 [0.1112 - 1.0402]
    Learning performance: 0.0026 +- 0.0031 [0.0000 - 0.0200]
    
```

