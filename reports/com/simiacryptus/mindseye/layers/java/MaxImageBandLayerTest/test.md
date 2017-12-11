# MaxImageBandLayer
## MaxImageBandLayerTest
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
      "class": "com.simiacryptus.mindseye.layers.java.MaxImageBandLayer",
      "id": "e2a3bda5-e7e7-4c05-aeb3-4ede00001ee3",
      "isFrozen": false,
      "name": "MaxImageBandLayer/e2a3bda5-e7e7-4c05-aeb3-4ede00001ee3"
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
    [[
    	[ [ 1.204, -1.436, 1.368 ], [ 1.812, -1.412, 1.552 ] ],
    	[ [ -0.424, -0.324, -1.564 ], [ -0.348, -0.524, -1.304 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ -0.348, -0.524, -1.304 ] ]
    ]
```



### Batch Execution
Code from [LayerTestBase.java:178](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L178) executed in 0.00 seconds: 
```java
    BatchingTester batchingTester = getBatchingTester();
    return batchingTester == null ? null : batchingTester.test(layer, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (150#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (60#)}
```



### Differential Validation
Code from [LayerTestBase.java:186](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L186) executed in 0.00 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ 1.204, -1.436, 1.368 ], [ 1.812, -1.412, 1.552 ] ],
    	[ [ -0.424, -0.324, -1.564 ], [ -0.348, -0.524, -1.304 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.026576052579973603, negative=8, min=-1.304, max=-1.304, mean=-0.11666666666666664, count=12.0, positive=4, stdDev=1.2156115973268582, zeros=0}
    Output: [
    	[ [ -0.348, -0.524, -1.304 ] ]
    ]
    Outputs Statistics: {meanExponent=-0.20793729255793034, negative=3, min=-1.304, max=-1.304, mean=-0.7253333333333334, count=3.0, positive=0, stdDev=0.41543979373938433, zeros=0}
    Feedback for input 0
    Inputs Values: [
    	[ [ 1.204, -1.436, 1.368 ], [ 1.812, -1.412, 1.552 ] ],
    	[ [ -0.424, -0.324, -1.564 ], [ -0.348, -0.524, -1.304 ] ]
    ]
    Value Statistics: {meanExponent=-0.026576052579973603, negative=8, min=-1.304, max=-1.304, mean=-0.11666666666666664, count=12.0, positive=4, stdDev=1.2156115973268582, zeros=0}
    Implemented Feedback: [ [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 1.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.
```
...[skipping 202 bytes](etc/70.txt)...
```
    eedback: [ [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.9999999999998899, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.9999999999998899, 0.0 ], ... ]
    Measured Statistics: {meanExponent=-4.7830642341045674E-14, negative=0, min=0.9999999999998899, max=0.9999999999998899, mean=0.08333333333332416, count=36.0, positive=3, stdDev=0.2763853991962529, zeros=33}
    Feedback Error: [ [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ -1.1013412404281553E-13, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, -1.1013412404281553E-13, 0.0 ], ... ]
    Error Statistics: {meanExponent=-12.958078098036827, negative=3, min=-1.1013412404281553E-13, max=-1.1013412404281553E-13, mean=-9.177843670234628E-15, count=36.0, positive=0, stdDev=3.0439463838706555E-14, zeros=33}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 9.1778e-15 +- 3.0439e-14 [0.0000e+00 - 1.1013e-13] (36#)
    relativeTol: 5.5067e-14 +- 0.0000e+00 [5.5067e-14 - 5.5067e-14] (3#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=9.1778e-15 +- 3.0439e-14 [0.0000e+00 - 1.1013e-13] (36#), relativeTol=5.5067e-14 +- 0.0000e+00 [5.5067e-14 - 5.5067e-14] (3#)}
```



### Performance
Code from [LayerTestBase.java:192](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L192) executed in 0.01 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 0.2783 +- 0.0831 [0.2052 - 0.7324]
    Learning performance: 0.0013 +- 0.0016 [0.0000 - 0.0086]
    
```

