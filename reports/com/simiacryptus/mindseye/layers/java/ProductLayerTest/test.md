# ProductLayer
## ProductLayerTest
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
      "class": "com.simiacryptus.mindseye.layers.java.ProductLayer",
      "id": "e2a3bda5-e7e7-4c05-aeb3-4ede00001f3d",
      "isFrozen": false,
      "name": "ProductLayer/e2a3bda5-e7e7-4c05-aeb3-4ede00001f3d"
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
    [[ -0.848, 1.096, -0.68 ]]
    --------------------
    Output: 
    [ 0.63199744 ]
```



### Batch Execution
Code from [LayerTestBase.java:178](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L178) executed in 0.00 seconds: 
```java
    BatchingTester batchingTester = getBatchingTester();
    return batchingTester == null ? null : batchingTester.test(layer, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (40#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (40#)}
```



### Differential Validation
Code from [LayerTestBase.java:186](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L186) executed in 0.00 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [ -0.848, 1.096, -0.68 ]
    Inputs Statistics: {meanExponent=-0.06642822696289982, negative=2, min=-0.68, max=-0.68, mean=-0.144, count=3.0, positive=1, stdDev=0.8794907617479562, zeros=0}
    Output: [ 0.63199744 ]
    Outputs Statistics: {meanExponent=-0.1992846808886995, negative=0, min=0.63199744, max=0.63199744, mean=0.63199744, count=1.0, positive=1, stdDev=0.0, zeros=0}
    Feedback for input 0
    Inputs Values: [ -0.848, 1.096, -0.68 ]
    Value Statistics: {meanExponent=-0.06642822696289982, negative=2, min=-0.68, max=-0.68, mean=-0.144, count=3.0, positive=1, stdDev=0.8794907617479562, zeros=0}
    Implemented Feedback: [ [ -0.74528 ], [ 0.5766399999999999 ], [ -0.929408 ] ]
    Implemented Statistics: {meanExponent=-0.13285645392579967, negative=2, min=-0.929408, max=-0.929408, mean=-0.36601600000000006, count=3.0, positive=1, stdDev=0.6707836366757913, zeros=0}
    Measured Feedback: [ [ -0.7452799999996262 ], [ 0.5766400000006833 ], [ -0.9294079999999649 ] ]
    Measured Statistics: {meanExponent=-0.1328564539257062, negative=2, min=-0.9294079999999649, max=-0.9294079999999649, mean=-0.36601599999963597, count=3.0, positive=1, stdDev=0.6707836366760311, zeros=0}
    Feedback Error: [ [ 3.738120923912902E-13 ], [ 6.833422716567839E-13 ], [ 3.5083047578154947E-14 ] ]
    Error Statistics: {meanExponent=-12.682537018099728, negative=0, min=3.5083047578154947E-14, max=3.5083047578154947E-14, mean=3.64079137208743E-13, count=3.0, positive=3, stdDev=2.6474019114746644E-13, zeros=0}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 3.6408e-13 +- 2.6474e-13 [3.5083e-14 - 6.8334e-13] (3#)
    relativeTol: 2.8739e-13 +- 2.3562e-13 [1.8874e-14 - 5.9252e-13] (3#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=3.6408e-13 +- 2.6474e-13 [3.5083e-14 - 6.8334e-13] (3#), relativeTol=2.8739e-13 +- 2.3562e-13 [1.8874e-14 - 5.9252e-13] (3#)}
```



### Performance
Code from [LayerTestBase.java:192](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L192) executed in 0.00 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 0.1388 +- 0.0374 [0.0997 - 0.3733]
    Learning performance: 0.0027 +- 0.0038 [0.0000 - 0.0371]
    
```

