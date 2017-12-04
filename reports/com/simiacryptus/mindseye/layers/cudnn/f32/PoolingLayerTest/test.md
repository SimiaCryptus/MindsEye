# PoolingLayer
## PoolingLayerTest
### Json Serialization
Code from [LayerTestBase.java:83](../../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L83) executed in 0.00 seconds: 
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
      "class": "com.simiacryptus.mindseye.layers.cudnn.f32.PoolingLayer",
      "id": "a864e734-2f23-44db-97c1-5040000003d8",
      "isFrozen": false,
      "name": "PoolingLayer/a864e734-2f23-44db-97c1-5040000003d8",
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
Code from [LayerTestBase.java:120](../../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L120) executed in 0.00 seconds: 
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
    	[ [ 0.344, 0.52 ], [ 0.54, 1.98 ] ],
    	[ [ 1.352, -0.952 ], [ -0.24, 1.592 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 1.3519999980926514, 1.9800000190734863 ] ]
    ]
```



### Batch Execution
Code from [LayerTestBase.java:138](../../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L138) executed in 0.01 seconds: 
```java
    BatchingTester batchingTester = getBatchingTester();
    return batchingTester==null?null:batchingTester.test(layer, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (100#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (40#)}
```



### Differential Validation
Code from [LayerTestBase.java:144](../../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.01 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ 0.344, 0.52 ], [ 0.54, 1.98 ] ],
    	[ [ 1.352, -0.952 ], [ -0.24, 1.592 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.12832641482572793, negative=2, min=1.592, max=1.592, mean=0.642, count=8.0, positive=6, stdDev=0.9103362016310238, zeros=0}
    Output: [
    	[ [ 1.3519999980926514, 1.9800000190734863 ] ]
    ]
    Outputs Statistics: {meanExponent=0.21382094271902669, negative=0, min=1.9800000190734863, max=1.9800000190734863, mean=1.6660000085830688, count=2.0, positive=2, stdDev=0.3140000104904175, zeros=0}
    Feedback for input 0
    Inputs Values: [
    	[ [ 0.344, 0.52 ], [ 0.54, 1.98 ] ],
    	[ [ 1.352, -0.952 ], [ -0.24, 1.592 ] ]
    ]
    Value Statistics: {meanExponent=-0.12832641482572793, negative=2, min=1.592, max=1.592, mean=0.642, count=8.0, positive=6, stdDev=0.9103362016310238, zeros=0}
    Implemented Feedback: [ [ 0.0, 0.0 ], [ 1.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 1.0 ], [ 0.0, 0.0 ] ]
    Implemented Statistics: {meanExponent=0.0, negative=0, min=0.0, max=0.0, mean=0.125, count=16.0, positive=2, stdDev=0.33071891388307384, zeros=14}
    Measured Feedback: [ [ 0.0, 0.0 ], [ 1.0001659393310547, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 1.0001659393310547 ], [ 0.0, 0.0 ] ]
    Measured Statistics: {meanExponent=7.206055713278508E-5, negative=0, min=0.0, max=0.0, mean=0.12502074241638184, count=16.0, positive=2, stdDev=0.3307737931584107, zeros=14}
    Feedback Error: [ [ 0.0, 0.0 ], [ 1.659393310546875E-4, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 1.659393310546875E-4 ], [ 0.0, 0.0 ] ]
    Error Statistics: {meanExponent=-3.7800506649970242, negative=0, min=0.0, max=0.0, mean=2.0742416381835938E-5, count=16.0, positive=2, stdDev=5.4879275336890075E-5, zeros=14}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 2.0742e-05 +- 5.4879e-05 [0.0000e+00 - 1.6594e-04] (16#)
    relativeTol: 8.2963e-05 +- 0.0000e+00 [8.2963e-05 - 8.2963e-05] (2#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=2.0742e-05 +- 5.4879e-05 [0.0000e+00 - 1.6594e-04] (16#), relativeTol=8.2963e-05 +- 0.0000e+00 [8.2963e-05 - 8.2963e-05] (2#)}
```



### Performance
Code from [LayerTestBase.java:149](../../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L149) executed in 0.12 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 6.1582 +- 29.2877 [2.5135 - 297.4866]
    Learning performance: 1.8781 +- 0.6255 [1.1000 - 4.7050]
    
```

