# DropoutNoiseLayer
## DropoutNoiseLayerTest
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
      "class": "com.simiacryptus.mindseye.layers.java.DropoutNoiseLayer",
      "id": "e2a3bda5-e7e7-4c05-aeb3-4ede00001e43",
      "isFrozen": false,
      "name": "DropoutNoiseLayer/e2a3bda5-e7e7-4c05-aeb3-4ede00001e43",
      "value": 0.5
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
    [[ 1.44, -1.56, -0.188 ]]
    --------------------
    Output: 
    [ 1.44, -1.56, -0.0 ]
```



### Batch Execution
Code from [LayerTestBase.java:178](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L178) executed in 0.00 seconds: 
```java
    BatchingTester batchingTester = getBatchingTester();
    return batchingTester == null ? null : batchingTester.test(layer, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (60#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (40#)}
```



### Differential Validation
Code from [LayerTestBase.java:186](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L186) executed in 0.00 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [ 1.44, -1.56, -0.188 ]
    Inputs Statistics: {meanExponent=-0.12478502009553631, negative=2, min=-0.188, max=-0.188, mean=-0.1026666666666667, count=3.0, positive=1, stdDev=1.2262303571877875, zeros=0}
    Output: [ 1.44, -1.56, -0.0 ]
    Outputs Statistics: {meanExponent=0.17574354522485564, negative=1, min=-0.0, max=-0.0, mean=-0.040000000000000036, count=3.0, positive=1, stdDev=1.2250714264890843, zeros=1}
    Feedback for input 0
    Inputs Values: [ 1.44, -1.56, -0.188 ]
    Value Statistics: {meanExponent=-0.12478502009553631, negative=2, min=-0.188, max=-0.188, mean=-0.1026666666666667, count=3.0, positive=1, stdDev=1.2262303571877875, zeros=0}
    Implemented Feedback: [ [ 1.0, 0.0, 0.0 ], [ 0.0, 1.0, 0.0 ], [ 0.0, 0.0, 0.0 ] ]
    Implemented Statistics: {meanExponent=0.0, negative=0, min=0.0, max=0.0, mean=0.2222222222222222, count=9.0, positive=2, stdDev=0.41573970964154905, zeros=7}
    Measured Feedback: [ [ 0.9999999999998899, 0.0, 0.0 ], [ 0.0, 0.9999999999998899, 0.0 ], [ 0.0, 0.0, 0.0 ] ]
    Measured Statistics: {meanExponent=-4.7830642341045674E-14, negative=0, min=0.0, max=0.0, mean=0.22222222222219776, count=9.0, positive=2, stdDev=0.41573970964150325, zeros=7}
    Feedback Error: [ [ -1.1013412404281553E-13, 0.0, 0.0 ], [ 0.0, -1.1013412404281553E-13, 0.0 ], [ 0.0, 0.0, 0.0 ] ]
    Error Statistics: {meanExponent=-12.958078098036825, negative=2, min=0.0, max=0.0, mean=-2.447424978729234E-14, count=9.0, positive=0, stdDev=4.5787128751186467E-14, zeros=7}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 2.4474e-14 +- 4.5787e-14 [0.0000e+00 - 1.1013e-13] (9#)
    relativeTol: 5.5067e-14 +- 0.0000e+00 [5.5067e-14 - 5.5067e-14] (2#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=2.4474e-14 +- 4.5787e-14 [0.0000e+00 - 1.1013e-13] (9#), relativeTol=5.5067e-14 +- 0.0000e+00 [5.5067e-14 - 5.5067e-14] (2#)}
```



### Performance
Code from [LayerTestBase.java:192](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L192) executed in 0.01 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 0.1859 +- 0.1452 [0.1339 - 1.3907]
    Learning performance: 0.0032 +- 0.0046 [0.0000 - 0.0370]
    
```

