# ImgBandSelectLayer
## ImgBandSelectLayerTest
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
      "class": "com.simiacryptus.mindseye.layers.java.ImgBandSelectLayer",
      "id": "e2a3bda5-e7e7-4c05-aeb3-4ede00001e86",
      "isFrozen": false,
      "name": "ImgBandSelectLayer/e2a3bda5-e7e7-4c05-aeb3-4ede00001e86",
      "bands": [
        0,
        2
      ]
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
    	[ [ 0.028, -1.38, 1.152 ], [ -0.124, 1.528, -0.604 ] ],
    	[ [ -0.052, 1.692, 1.824 ], [ 1.596, 0.304, -1.068 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 0.028, 1.152 ], [ -0.124, -0.604 ] ],
    	[ [ -0.052, 1.824 ], [ 1.596, -1.068 ] ]
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
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (200#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (160#)}
```



### Differential Validation
Code from [LayerTestBase.java:186](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L186) executed in 0.00 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ 0.028, -1.38, 1.152 ], [ -0.124, 1.528, -0.604 ] ],
    	[ [ -0.052, 1.692, 1.824 ], [ 1.596, 0.304, -1.068 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.2810851804583425, negative=5, min=-1.068, max=-1.068, mean=0.4079999999999999, count=12.0, positive=7, stdDev=1.0771295186745187, zeros=0}
    Output: [
    	[ [ 0.028, 1.152 ], [ -0.124, -0.604 ] ],
    	[ [ -0.052, 1.824 ], [ 1.596, -1.068 ] ]
    ]
    Outputs Statistics: {meanExponent=-0.42603731855659704, negative=4, min=-1.068, max=-1.068, mean=0.34400000000000003, count=8.0, positive=4, stdDev=0.986085189017663, zeros=0}
    Feedback for input 0
    Inputs Values: [
    	[ [ 0.028, -1.38, 1.152 ], [ -0.124, 1.528, -0.604 ] ],
    	[ [ -0.052, 1.692, 1.824 ], [ 1.596, 0.304, -1.068 ] ]
    ]
    Value Statistics: {meanExponent=-0.2810851804583425, negative=5, min=-1.068, max=-1.068, mean=0.4079999999999999, count=12.0, positive=7, stdDev=1.0771295186745187, zeros=0}
    Implemented Feedback: [ [ 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0
```
...[skipping 901 bytes](etc/61.txt)...
```
    -2.7109655903480978E-14, negative=0, min=0.9999999999998899, max=0.9999999999998899, mean=0.08333333333332814, count=96.0, positive=8, stdDev=0.2763853991962661, zeros=88}
    Feedback Error: [ [ -5.995204332975845E-15, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 2.864375403532904E-14, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 2.864375403532904E-14, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, -1.1013412404281553E-13, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], ... ]
    Error Statistics: {meanExponent=-13.262315828625962, negative=6, min=-1.1013412404281553E-13, max=-1.1013412404281553E-13, mean=-5.201857463295785E-15, count=96.0, positive=2, stdDev=2.4943019091915518E-14, zeros=88}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 6.3953e-15 +- 2.4664e-14 [0.0000e+00 - 1.1013e-13] (96#)
    relativeTol: 3.8372e-14 +- 2.1800e-14 [2.9976e-15 - 5.5067e-14] (8#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=6.3953e-15 +- 2.4664e-14 [0.0000e+00 - 1.1013e-13] (96#), relativeTol=3.8372e-14 +- 2.1800e-14 [2.9976e-15 - 5.5067e-14] (8#)}
```



### Performance
Code from [LayerTestBase.java:192](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L192) executed in 0.00 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 0.1788 +- 0.0732 [0.1254 - 0.7837]
    Learning performance: 0.0022 +- 0.0019 [0.0000 - 0.0142]
    
```

