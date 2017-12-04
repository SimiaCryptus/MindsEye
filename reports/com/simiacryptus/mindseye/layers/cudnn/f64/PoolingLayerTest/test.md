# PoolingLayer
## PoolingLayerTest
### Json Serialization
Code from [LayerTestBase.java:83](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L83) executed in 0.00 seconds: 
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
      "class": "com.simiacryptus.mindseye.layers.cudnn.f64.PoolingLayer",
      "id": "370a9587-74a1-4959-b406-fa4500002b41",
      "isFrozen": false,
      "name": "PoolingLayer/370a9587-74a1-4959-b406-fa4500002b41",
      "mode": 0,
      "windowX": 2,
      "windowY": 2,
      "paddingX": 0,
      "paddingY": 0,
      "strideX": 2,
      "strideY": 2
    }
```



### Example Input/Output Pair
Code from [LayerTestBase.java:120](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L120) executed in 0.00 seconds: 
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
    	[ [ -1.112, 0.236 ], [ -0.012, 0.616 ] ],
    	[ [ -1.644, 0.484 ], [ 1.936, -0.548 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 1.936, 0.616 ] ]
    ]
```



### Batch Execution
Code from [LayerTestBase.java:138](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L138) executed in 0.01 seconds: 
```java
    BatchingTester batchingTester = getBatchingTester();
    return batchingTester==null?null:batchingTester.test(layer, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (100#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (40#)}
```



### Differential Validation
Code from [LayerTestBase.java:144](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.01 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ -1.112, 0.236 ], [ -0.012, 0.616 ] ],
    	[ [ -1.644, 0.484 ], [ 1.936, -0.548 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.348223520658327, negative=4, min=-0.548, max=-0.548, mean=-0.0055000000000000465, count=8.0, positive=4, stdDev=1.0402517724089684, zeros=0}
    Output: [
    	[ [ 1.936, 0.616 ] ]
    ]
    Outputs Statistics: {meanExponent=0.038243032568400157, negative=0, min=0.616, max=0.616, mean=1.276, count=2.0, positive=2, stdDev=0.6599999999999998, zeros=0}
    Feedback for input 0
    Inputs Values: [
    	[ [ -1.112, 0.236 ], [ -0.012, 0.616 ] ],
    	[ [ -1.644, 0.484 ], [ 1.936, -0.548 ] ]
    ]
    Value Statistics: {meanExponent=-0.348223520658327, negative=4, min=-0.548, max=-0.548, mean=-0.0055000000000000465, count=8.0, positive=4, stdDev=1.0402517724089684, zeros=0}
    Implemented Feedback: [ [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 1.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 1.0 ], [ 0.0, 0.0 ] ]
    Implemented Statistics: {meanExponent=0.0, negative=0, min=0.0, max=0.0, mean=0.125, count=16.0, positive=2, stdDev=0.33071891388307384, zeros=14}
    Measured Feedback: [ [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.9999999999998899, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.9999999999998899 ], [ 0.0, 0.0 ] ]
    Measured Statistics: {meanExponent=-4.7830642341045674E-14, negative=0, min=0.0, max=0.0, mean=0.12499999999998623, count=16.0, positive=2, stdDev=0.3307189138830374, zeros=14}
    Feedback Error: [ [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ -1.1013412404281553E-13, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, -1.1013412404281553E-13 ], [ 0.0, 0.0 ] ]
    Error Statistics: {meanExponent=-12.958078098036825, negative=2, min=0.0, max=0.0, mean=-1.3766765505351941E-14, count=16.0, positive=0, stdDev=3.6423437884903677E-14, zeros=14}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 1.3767e-14 +- 3.6423e-14 [0.0000e+00 - 1.1013e-13] (16#)
    relativeTol: 5.5067e-14 +- 0.0000e+00 [5.5067e-14 - 5.5067e-14] (2#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=1.3767e-14 +- 3.6423e-14 [0.0000e+00 - 1.1013e-13] (16#), relativeTol=5.5067e-14 +- 0.0000e+00 [5.5067e-14 - 5.5067e-14] (2#)}
```



### Performance
Code from [LayerTestBase.java:149](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L149) executed in 0.10 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 4.4479 +- 5.7828 [2.6190 - 46.7622]
    Learning performance: 1.8906 +- 0.5533 [1.0915 - 4.1179]
    
```

