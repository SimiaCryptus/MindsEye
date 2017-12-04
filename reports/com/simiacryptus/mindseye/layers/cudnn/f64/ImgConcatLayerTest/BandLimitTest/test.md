# ImgConcatLayer
## BandLimitTest
### Json Serialization
Code from [LayerTestBase.java:83](../../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L83) executed in 0.00 seconds: 
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
      "class": "com.simiacryptus.mindseye.layers.cudnn.f64.ImgConcatLayer",
      "id": "370a9587-74a1-4959-b406-fa4500002b28",
      "isFrozen": false,
      "name": "ImgConcatLayer/370a9587-74a1-4959-b406-fa4500002b28",
      "maxBands": 3
    }
```



### Example Input/Output Pair
Code from [LayerTestBase.java:120](../../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L120) executed in 0.00 seconds: 
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
    	[ [ 0.836, 0.264 ], [ -1.88, 1.32 ] ],
    	[ [ -1.72, -1.16 ], [ 1.108, -1.588 ] ]
    ],
    [
    	[ [ -1.052, 0.472 ], [ 1.288, 1.472 ] ],
    	[ [ 1.144, 1.28 ], [ 0.068, 0.8 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 0.836, 0.264, -1.052 ], [ -1.88, 1.32, 1.288 ] ],
    	[ [ -1.72, -1.16, 1.144 ], [ 1.108, -1.588, 0.068 ] ]
    ]
```



### Batch Execution
Code from [LayerTestBase.java:138](../../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L138) executed in 0.01 seconds: 
```java
    BatchingTester batchingTester = getBatchingTester();
    return batchingTester==null?null:batchingTester.test(layer, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (280#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (240#)}
```



### Differential Validation
Code from [LayerTestBase.java:144](../../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.12 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ 0.836, 0.264 ], [ -1.88, 1.32 ] ],
    	[ [ -1.72, -1.16 ], [ 1.108, -1.588 ] ]
    ],
    [
    	[ [ -1.052, 0.472 ], [ 1.288, 1.472 ] ],
    	[ [ 1.144, 1.28 ], [ 0.068, 0.8 ] ]
    ]
    Inputs Statistics: {meanExponent=0.035489834924541615, negative=4, min=-1.588, max=-1.588, mean=-0.3524999999999999, count=8.0, positive=4, stdDev=1.2798803655029636, zeros=0},
    {meanExponent=-0.1406229618399831, negative=1, min=0.8, max=0.8, mean=0.684, count=8.0, positive=7, stdDev=0.7908956947663832, zeros=0}
    Output: [
    	[ [ 0.836, 0.264, -1.052 ], [ -1.88, 1.32, 1.288 ] ],
    	[ [ -1.72, -1.16, 1.144 ], [ 1.108, -1.588, 0.068 ] ]
    ]
    Outputs Statistics: {meanExponent=-0.057767898383242644, negative=5, min=0.068, max=0.068, mean=-0.1143333333333333, count=12.0, positive=7, stdDev=1.2254571482602818, zeros=0}
    Feedback for input 0
    Inputs Values: [
    	[ [ 0.836, 0.264 ], [ -1.88, 1.32 ] ],
    	[ [ -1.72, -1.16 ], [ 1.108, -1.588 ] ]
    ]
    Value Statistics: {meanExponent=0.035489834924541615, negative=4, min=-1.588, max=-1.588, mean=-0.3524999999999999, count=8.0, positive=4, stdDev=1.2798803655029636, zeros=0}
    Implemented Feedback: [ [ 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, ... ] ]
    Implemented Statistics: {meanExponent=0.0, negative=0, min=0.0, max=0.0, mean=0.08333333333333333, count=96.0, positive=8, stdDev=0.2763853991962833, zeros=88}
    Measured Feedback: [ [ 0.9999999999998899, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.9999999999998899, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.9999999999998899, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.9999999999998899, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.9999999999998899, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.9999999
```
...[skipping 1261 bytes](etc/1.txt)...
```
    egative=1, min=0.8, max=0.8, mean=0.684, count=8.0, positive=7, stdDev=0.7908956947663832, zeros=0}
    Implemented Feedback: [ [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ] ]
    Implemented Statistics: {meanExponent=0.0, negative=0, min=0.0, max=0.0, mean=0.041666666666666664, count=96.0, positive=4, stdDev=0.19982631347136331, zeros=92}
    Measured Feedback: [ [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ] ]
    Measured Statistics: {meanExponent=-3.276302567614995E-14, negative=0, min=0.0, max=0.0, mean=0.04166666666666352, count=96.0, positive=4, stdDev=0.19982631347134824, zeros=92}
    Feedback Error: [ [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ] ]
    Error Statistics: {meanExponent=-13.104301089584563, negative=3, min=0.0, max=0.0, mean=-3.1433189384699745E-15, count=96.0, positive=1, stdDev=1.943485831517598E-14, zeros=92}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 6.4590e-15 +- 2.5641e-14 [0.0000e+00 - 1.1013e-13] (192#)
    relativeTol: 5.1672e-14 +- 1.1261e-14 [1.4322e-14 - 5.5067e-14] (12#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=6.4590e-15 +- 2.5641e-14 [0.0000e+00 - 1.1013e-13] (192#), relativeTol=5.1672e-14 +- 1.1261e-14 [1.4322e-14 - 5.5067e-14] (12#)}
```



### Performance
Code from [LayerTestBase.java:149](../../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L149) executed in 0.12 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 5.2504 +- 0.9287 [3.4340 - 8.1219]
    Learning performance: 1.6060 +- 0.4593 [0.9632 - 3.5423]
    
```

