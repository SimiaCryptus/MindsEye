# MaxSubsampleLayer
## MaxSubsampleLayerTest
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
      "class": "com.simiacryptus.mindseye.layers.java.MaxSubsampleLayer",
      "id": "a864e734-2f23-44db-97c1-504000002c2b",
      "isFrozen": false,
      "name": "MaxSubsampleLayer/a864e734-2f23-44db-97c1-504000002c2b",
      "inner": [
        2,
        2,
        1
      ]
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
    [[
    	[ [ 1.012, 0.392, -1.216 ], [ 0.092, -0.828, 1.272 ] ],
    	[ [ 0.26, -0.46, -0.316 ], [ 1.756, -0.596, 0.924 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 1.756, 0.392, 1.272 ] ]
    ]
```



### Batch Execution
Code from [LayerTestBase.java:138](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L138) executed in 0.00 seconds: 
```java
    BatchingTester batchingTester = getBatchingTester();
    return batchingTester==null?null:batchingTester.test(layer, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (150#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (60#)}
```



### Differential Validation
Code from [LayerTestBase.java:144](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.00 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ 1.012, 0.392, -1.216 ], [ 0.092, -0.828, 1.272 ] ],
    	[ [ 0.26, -0.46, -0.316 ], [ 1.756, -0.596, 0.924 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.23061946377458273, negative=5, min=0.924, max=0.924, mean=0.19100000000000003, count=12.0, positive=7, stdDev=0.8757048589564865, zeros=0}
    Output: [
    	[ [ 1.756, 0.392, 1.272 ] ]
    ]
    Outputs Statistics: {meanExponent=-0.019234103365687965, negative=0, min=1.272, max=1.272, mean=1.14, count=3.0, positive=3, stdDev=0.5646190456109916, zeros=0}
    Feedback for input 0
    Inputs Values: [
    	[ [ 1.012, 0.392, -1.216 ], [ 0.092, -0.828, 1.272 ] ],
    	[ [ 0.26, -0.46, -0.316 ], [ 1.756, -0.596, 0.924 ] ]
    ]
    Value Statistics: {meanExponent=-0.23061946377458273, negative=5, min=0.924, max=0.924, mean=0.19100000000000003, count=12.0, positive=7, stdDev=0.8757048589564865, zeros=0}
    Implemented Feedback: [ [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 1.0, 0.0, 0.0 ], [ 0.0, 1.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], ... ]
    Implemented Statistics: {meanExponent=0.0, negative=0, min=0.0, max=0.0, mean=0.08333333333333333, count=36.0, positive=3, stdDev=0.2763853991962833, zeros=33}
    Measured Feedback: [ [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.9999999999998899, 0.0, 0.0 ], [ 0.0, 0.9999999999998899, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], ... ]
    Measured Statistics: {meanExponent=-4.7830642341045674E-14, negative=0, min=0.0, max=0.0, mean=0.08333333333332416, count=36.0, positive=3, stdDev=0.2763853991962529, zeros=33}
    Feedback Error: [ [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ -1.1013412404281553E-13, 0.0, 0.0 ], [ 0.0, -1.1013412404281553E-13, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], ... ]
    Error Statistics: {meanExponent=-12.958078098036827, negative=3, min=0.0, max=0.0, mean=-9.177843670234628E-15, count=36.0, positive=0, stdDev=3.0439463838706555E-14, zeros=33}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 9.1778e-15 +- 3.0439e-14 [0.0000e+00 - 1.1013e-13] (36#)
    relativeTol: 5.5067e-14 +- 0.0000e+00 [5.5067e-14 - 5.5067e-14] (3#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=9.1778e-15 +- 3.0439e-14 [0.0000e+00 - 1.1013e-13] (36#), relativeTol=5.5067e-14 +- 0.0000e+00 [5.5067e-14 - 5.5067e-14] (3#)}
```



### Performance
Code from [LayerTestBase.java:149](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L149) executed in 0.01 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 0.4369 +- 0.2748 [0.2479 - 2.6275]
    Learning performance: 0.0042 +- 0.0140 [0.0000 - 0.1425]
    
```

