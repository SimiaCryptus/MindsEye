# ImgConcatLayer
## BandLimitTest
### Json Serialization
Code from [LayerTestBase.java:83](../../../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L83) executed in 0.00 seconds: 
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
      "id": "a864e734-2f23-44db-97c1-504000002b28",
      "isFrozen": false,
      "name": "ImgConcatLayer/a864e734-2f23-44db-97c1-504000002b28",
      "maxBands": 3
    }
```



### Example Input/Output Pair
Code from [LayerTestBase.java:120](../../../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L120) executed in 0.00 seconds: 
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
    	[ [ -1.772, -1.792 ], [ 1.968, -1.36 ] ],
    	[ [ 0.424, -0.056 ], [ 1.796, 0.084 ] ]
    ],
    [
    	[ [ -1.8, 1.692 ], [ -0.836, -1.912 ] ],
    	[ [ -1.872, -1.624 ], [ -1.296, -1.984 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ -1.772, -1.792, -1.8 ], [ 1.968, -1.36, -0.836 ] ],
    	[ [ 0.424, -0.056, -1.872 ], [ 1.796, 0.084, -1.296 ] ]
    ]
```



### Batch Execution
Code from [LayerTestBase.java:138](../../../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L138) executed in 0.01 seconds: 
```java
    BatchingTester batchingTester = getBatchingTester();
    return batchingTester==null?null:batchingTester.test(layer, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (280#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (240#)}
```



### Differential Validation
Code from [LayerTestBase.java:144](../../../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.03 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ -1.772, -1.792 ], [ 1.968, -1.36 ] ],
    	[ [ 0.424, -0.056 ], [ 1.796, 0.084 ] ]
    ],
    [
    	[ [ -1.8, 1.692 ], [ -0.836, -1.912 ] ],
    	[ [ -1.872, -1.624 ], [ -1.296, -1.984 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.18956184658315264, negative=4, min=0.084, max=0.084, mean=-0.08850000000000004, count=8.0, positive=4, stdDev=1.3886503339574006, zeros=0},
    {meanExponent=0.1975506959806732, negative=7, min=-1.984, max=-1.984, mean=-1.2040000000000002, count=8.0, positive=1, stdDev=1.1520746503590813, zeros=0}
    Output: [
    	[ [ -1.772, -1.792, -1.8 ], [ 1.968, -1.36, -0.836 ] ],
    	[ [ 0.424, -0.056, -1.872 ], [ 1.796, 0.084, -1.296 ] ]
    ]
    Outputs Statistics: {meanExponent=-0.07950876201551982, negative=8, min=-1.296, max=-1.296, mean=-0.5426666666666667, count=12.0, positive=4, stdDev=1.3253460763974905, zeros=0}
    Feedback for input 0
    Inputs Values: [
    	[ [ -1.772, -1.792 ], [ 1.968, -1.36 ] ],
    	[ [ 0.424, -0.056 ], [ 1.796, 0.084 ] ]
    ]
    Value Statistics: {meanExponent=-0.18956184658315264, negative=4, min=0.084, max=0.084, mean=-0.08850000000000004, count=8.0, positive=4, stdDev=1.3886503339574006, zeros=0}
    Implemented Feedback: [ [ 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, ... ] ]
    Implemented Statistics: {meanExponent=0.0, negative=0, min=0.0, max=0.0, mean=0.08333333333333333, count=96.0, positive=8, stdDev=0.2763853991962833, zeros=88}
    Measured Feedback: [ [ 0.9999999999998899, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.9999999999998899, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.9999999999998899, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.9999999999998899, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.9999999999998899, 0.0, 0.0, 0.0, ... ], [
```
...[skipping 1317 bytes](etc/1.txt)...
```
    4, max=-1.984, mean=-1.2040000000000002, count=8.0, positive=1, stdDev=1.1520746503590813, zeros=0}
    Implemented Feedback: [ [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ] ]
    Implemented Statistics: {meanExponent=0.0, negative=0, min=0.0, max=0.0, mean=0.041666666666666664, count=96.0, positive=4, stdDev=0.19982631347136331, zeros=92}
    Measured Feedback: [ [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ] ]
    Measured Statistics: {meanExponent=-4.7830642341045674E-14, negative=0, min=0.0, max=0.0, mean=0.04166666666666208, count=96.0, positive=4, stdDev=0.1998263134713413, zeros=92}
    Feedback Error: [ [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ] ]
    Error Statistics: {meanExponent=-12.958078098036825, negative=4, min=0.0, max=0.0, mean=-4.588921835117314E-15, count=96.0, positive=0, stdDev=2.2007695994873667E-14, zeros=92}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 6.0345e-15 +- 2.4574e-14 [0.0000e+00 - 1.1013e-13] (192#)
    relativeTol: 4.8276e-14 +- 1.5185e-14 [1.4322e-14 - 5.5067e-14] (12#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=6.0345e-15 +- 2.4574e-14 [0.0000e+00 - 1.1013e-13] (192#), relativeTol=4.8276e-14 +- 1.5185e-14 [1.4322e-14 - 5.5067e-14] (12#)}
```



### Performance
Code from [LayerTestBase.java:149](../../../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L149) executed in 0.20 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 12.9685 +- 90.5470 [3.0436 - 913.8731]
    Learning performance: 1.5145 +- 0.3034 [0.9803 - 2.4736]
    
```

