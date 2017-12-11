# StaticScalarLossLayer
## StaticScalarLossLayerTest
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
      "class": "com.simiacryptus.mindseye.layers.java.StaticScalarLossLayer",
      "id": "e2a3bda5-e7e7-4c05-aeb3-4ede00003670",
      "isFrozen": false,
      "name": "StaticScalarLossLayer/e2a3bda5-e7e7-4c05-aeb3-4ede00003670"
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
    [[ -1.204 ]]
    --------------------
    Output: 
    [ 1.204 ]
```



### Differential Validation
Code from [LayerTestBase.java:186](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L186) executed in 0.00 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [ -1.204 ]
    Inputs Statistics: {meanExponent=0.08062648692180573, negative=1, min=-1.204, max=-1.204, mean=-1.204, count=1.0, positive=0, stdDev=0.0, zeros=0}
    Output: [ 1.204 ]
    Outputs Statistics: {meanExponent=0.08062648692180573, negative=0, min=1.204, max=1.204, mean=1.204, count=1.0, positive=1, stdDev=0.0, zeros=0}
    Feedback for input 0
    Inputs Values: [ -1.204 ]
    Value Statistics: {meanExponent=0.08062648692180573, negative=1, min=-1.204, max=-1.204, mean=-1.204, count=1.0, positive=0, stdDev=0.0, zeros=0}
    Implemented Feedback: [ [ -1.0 ] ]
    Implemented Statistics: {meanExponent=0.0, negative=1, min=-1.0, max=-1.0, mean=-1.0, count=1.0, positive=0, stdDev=0.0, zeros=0}
    Measured Feedback: [ [ -0.9999999999998899 ] ]
    Measured Statistics: {meanExponent=-4.7830642341045674E-14, negative=1, min=-0.9999999999998899, max=-0.9999999999998899, mean=-0.9999999999998899, count=1.0, positive=0, stdDev=0.0, zeros=0}
    Feedback Error: [ [ 1.1013412404281553E-13 ] ]
    Error Statistics: {meanExponent=-12.958078098036825, negative=0, min=1.1013412404281553E-13, max=1.1013412404281553E-13, mean=1.1013412404281553E-13, count=1.0, positive=1, stdDev=0.0, zeros=0}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 1.1013e-13 +- 0.0000e+00 [1.1013e-13 - 1.1013e-13] (1#)
    relativeTol: 5.5067e-14 +- 0.0000e+00 [5.5067e-14 - 5.5067e-14] (1#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=1.1013e-13 +- 0.0000e+00 [1.1013e-13 - 1.1013e-13] (1#), relativeTol=5.5067e-14 +- 0.0000e+00 [5.5067e-14 - 5.5067e-14] (1#)}
```



### Performance
Code from [LayerTestBase.java:192](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L192) executed in 0.00 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 0.1865 +- 0.0913 [0.1453 - 0.8549]
    Learning performance: 0.0100 +- 0.0029 [0.0057 - 0.0314]
    
```

