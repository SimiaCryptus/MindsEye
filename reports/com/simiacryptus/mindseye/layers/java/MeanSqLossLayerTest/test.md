# MeanSqLossLayer
## MeanSqLossLayerTest
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
      "class": "com.simiacryptus.mindseye.layers.java.MeanSqLossLayer",
      "id": "e2a3bda5-e7e7-4c05-aeb3-4ede00001eef",
      "isFrozen": false,
      "name": "MeanSqLossLayer/e2a3bda5-e7e7-4c05-aeb3-4ede00001eef"
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
    	[ [ 1.268 ], [ -1.932 ], [ -1.404 ] ],
    	[ [ -0.32 ], [ 1.792 ], [ 0.352 ] ]
    ],
    [
    	[ [ -1.276 ], [ 1.792 ], [ 1.284 ] ],
    	[ [ 0.504 ], [ -0.136 ], [ -0.244 ] ]
    ]]
    --------------------
    Output: 
    [ 5.3861386666666675 ]
```



### Batch Execution
Code from [LayerTestBase.java:178](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L178) executed in 0.00 seconds: 
```java
    BatchingTester batchingTester = getBatchingTester();
    return batchingTester == null ? null : batchingTester.test(layer, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (130#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (128#)}
```



### Differential Validation
Code from [LayerTestBase.java:186](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L186) executed in 0.00 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ 1.268 ], [ -1.932 ], [ -1.404 ] ],
    	[ [ -0.32 ], [ 1.792 ], [ 0.352 ] ]
    ],
    [
    	[ [ -1.276 ], [ 1.792 ], [ 1.284 ] ],
    	[ [ 0.504 ], [ -0.136 ], [ -0.244 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.026412644909480282, negative=3, min=0.352, max=0.352, mean=-0.04066666666666665, count=6.0, positive=3, stdDev=1.337865796292322, zeros=0},
    {meanExponent=-0.21814783756690725, negative=3, min=-0.244, max=-0.244, mean=0.3206666666666667, count=6.0, positive=3, stdDev=1.0167908120924163, zeros=0}
    Output: [ 5.3861386666666675 ]
    Outputs Statistics: {meanExponent=0.7312775301733264, negative=0, min=5.3861386666666675, max=5.3861386666666675, mean=5.3861386666666675, count=1.0, positive=1, stdDev=0.0, zeros=0}
    Feedback for input 0
    Inputs Values: [
    	[ [ 1.268 ], [ -1.932 ], [ -1.404 ] ],
    	[ [ -0.32 ], [ 1.792 ], [ 0.352 ] ]
    ]
    Value Statistics: {meanExponent=-0.026412644909480282, negative=3, min=0.352, max=0.352, mean=-0.04066666666666665, count=6.0, positive=3, stdDev=1.337865796292322, zeros=0}
    Implemen
```
...[skipping 1709 bytes](etc/72.txt)...
```
    0.12044444444444447, count=6.0, positive=3, stdDev=0.7641681671296712, zeros=0}
    Measured Feedback: [ [ -0.8479833333385756 ], [ 0.2746833333233667 ], [ 1.2413499999919253 ], [ -0.6426499999978574 ], [ 0.8960166666582836 ], [ -0.19864999999619215 ] ]
    Measured Statistics: {meanExponent=-0.2467511594497125, negative=3, min=-0.19864999999619215, max=-0.19864999999619215, mean=0.12046111110682507, count=6.0, positive=3, stdDev=0.7641681671264294, zeros=0}
    Feedback Error: [ [ 1.6666661424413753E-5 ], [ 1.6666656700026206E-5 ], [ 1.666665859190175E-5 ], [ 1.6666668809173224E-5 ], [ 1.666665828370384E-5 ], [ 1.666667047450776E-5 ] ]
    Error Statistics: {meanExponent=-4.778151362068036, negative=0, min=1.666667047450776E-5, max=1.666667047450776E-5, mean=1.666666238062109E-5, count=6.0, positive=6, stdDev=5.34207180156533E-12, zeros=0}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 1.6667e-05 +- 4.8925e-12 [1.6667e-05 - 1.6667e-05] (12#)
    relativeTol: 1.8516e-05 +- 1.3034e-05 [6.7132e-06 - 4.1948e-05] (12#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=1.6667e-05 +- 4.8925e-12 [1.6667e-05 - 1.6667e-05] (12#), relativeTol=1.8516e-05 +- 1.3034e-05 [6.7132e-06 - 4.1948e-05] (12#)}
```



### Performance
Code from [LayerTestBase.java:192](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L192) executed in 0.01 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 0.3791 +- 0.0874 [0.2365 - 0.6213]
    Learning performance: 0.0048 +- 0.0029 [0.0028 - 0.0228]
    
```

