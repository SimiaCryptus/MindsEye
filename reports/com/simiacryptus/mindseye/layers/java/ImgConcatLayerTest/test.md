# ImgConcatLayer
## ImgConcatLayerTest
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
      "class": "com.simiacryptus.mindseye.layers.java.ImgConcatLayer",
      "id": "e2a3bda5-e7e7-4c05-aeb3-4ede00001e8f",
      "isFrozen": false,
      "name": "ImgConcatLayer/e2a3bda5-e7e7-4c05-aeb3-4ede00001e8f"
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
    	[ [ 1.72 ], [ 1.78 ] ],
    	[ [ -0.96 ], [ 0.66 ] ]
    ],
    [
    	[ [ -1.16 ], [ -0.264 ] ],
    	[ [ -1.644 ], [ -0.916 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 1.72, -1.16 ], [ 1.78, -0.264 ] ],
    	[ [ -0.96, -1.644 ], [ 0.66, -0.916 ] ]
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
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (160#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (160#)}
```



### Differential Validation
Code from [LayerTestBase.java:186](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L186) executed in 0.00 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ 1.72 ], [ 1.78 ] ],
    	[ [ -0.96 ], [ 0.66 ] ]
    ],
    [
    	[ [ -1.16 ], [ -0.264 ] ],
    	[ [ -1.644 ], [ -0.916 ] ]
    ]
    Inputs Statistics: {meanExponent=0.07194090444946999, negative=1, min=0.66, max=0.66, mean=0.8, count=4.0, positive=3, stdDev=1.1095043938624127, zeros=0},
    {meanExponent=-0.08403519925784213, negative=4, min=-0.916, max=-0.916, mean=-0.9959999999999999, count=4.0, positive=0, stdDev=0.4972484288562412, zeros=0}
    Output: [
    	[ [ 1.72, -1.16 ], [ 1.78, -0.264 ] ],
    	[ [ -0.96, -1.644 ], [ 0.66, -0.916 ] ]
    ]
    Outputs Statistics: {meanExponent=-0.0060471474041860675, negative=5, min=-0.916, max=-0.916, mean=-0.09799999999999999, count=8.0, positive=3, stdDev=1.2431942728310807, zeros=0}
    Feedback for input 0
    Inputs Values: [
    	[ [ 1.72 ], [ 1.78 ] ],
    	[ [ -0.96 ], [ 0.66 ] ]
    ]
    Value Statistics: {meanExponent=0.07194090444946999, negative=1, min=0.66, max=0.66, mean=0.8, count=4.0, positive=3, stdDev=1.1095043938624127, zeros=0}
    Implemented Feedback: [ [ 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
```
...[skipping 1921 bytes](etc/62.txt)...
```
    998899, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9999999999998899, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9999999999998899 ] ]
    Measured Statistics: {meanExponent=-4.7830642341045674E-14, negative=0, min=0.9999999999998899, max=0.9999999999998899, mean=0.12499999999998623, count=32.0, positive=4, stdDev=0.3307189138830374, zeros=28}
    Feedback Error: [ [ 0.0, 0.0, 0.0, 0.0, -1.1013412404281553E-13, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, -1.1013412404281553E-13, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.1013412404281553E-13, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.1013412404281553E-13 ] ]
    Error Statistics: {meanExponent=-12.958078098036825, negative=4, min=-1.1013412404281553E-13, max=-1.1013412404281553E-13, mean=-1.3766765505351941E-14, count=32.0, positive=0, stdDev=3.6423437884903677E-14, zeros=28}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 1.3767e-14 +- 3.6423e-14 [0.0000e+00 - 1.1013e-13] (64#)
    relativeTol: 5.5067e-14 +- 0.0000e+00 [5.5067e-14 - 5.5067e-14] (8#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=1.3767e-14 +- 3.6423e-14 [0.0000e+00 - 1.1013e-13] (64#), relativeTol=5.5067e-14 +- 0.0000e+00 [5.5067e-14 - 5.5067e-14] (8#)}
```



### Performance
Code from [LayerTestBase.java:192](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L192) executed in 0.01 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 0.2043 +- 0.0393 [0.1681 - 0.4018]
    Learning performance: 0.0388 +- 0.0201 [0.0257 - 0.1482]
    
```

