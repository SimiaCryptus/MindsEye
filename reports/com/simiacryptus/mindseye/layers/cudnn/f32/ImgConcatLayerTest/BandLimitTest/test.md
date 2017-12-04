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
      "class": "com.simiacryptus.mindseye.layers.cudnn.f32.ImgConcatLayer",
      "id": "a864e734-2f23-44db-97c1-5040000003d1",
      "isFrozen": false,
      "name": "ImgConcatLayer/a864e734-2f23-44db-97c1-5040000003d1",
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
    	[ [ 0.824, -0.856 ] ]
    ],
    [
    	[ [ 1.536, 1.032 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 0.8240000009536743, -0.8560000061988831, 1.5360000133514404 ] ]
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
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (70#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (60#)}
```



### Differential Validation
Code from [LayerTestBase.java:144](../../../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.01 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ 0.824, -0.856 ] ]
    ],
    [
    	[ [ 1.536, 1.032 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.07579951181286551, negative=1, min=-0.856, max=-0.856, mean=-0.016000000000000014, count=2.0, positive=1, stdDev=0.84, zeros=0},
    {meanExponent=0.10003545649334288, negative=0, min=1.032, max=1.032, mean=1.284, count=2.0, positive=2, stdDev=0.25199999999999956, zeros=0}
    Output: [
    	[ [ 0.8240000009536743, -0.8560000061988831, 1.5360000133514404 ] ]
    ]
    Outputs Statistics: {meanExponent=0.011597399830821198, negative=1, min=1.5360000133514404, max=1.5360000133514404, mean=0.5013333360354105, count=3.0, positive=2, stdDev=1.0028297814582139, zeros=0}
    Feedback for input 0
    Inputs Values: [
    	[ [ 0.824, -0.856 ] ]
    ]
    Value Statistics: {meanExponent=-0.07579951181286551, negative=1, min=-0.856, max=-0.856, mean=-0.016000000000000014, count=2.0, positive=1, stdDev=0.84, zeros=0}
    Implemented Feedback: [ [ 1.0, 0.0, 0.0 ], [ 0.0, 1.0, 0.0 ] ]
    Implemented Statistics: {meanExponent=0.0, negative=0, min=0.0, max=0.0, mean=0.3333333333333333, count=6.0, positive=2, stdDev=0.4714045207910317, zeros=4}
    Measured Feedback: [ [ 1.0001659393310547, 0.0, 0.0 ], [ 0.0, 1.0001659393310547, 0.0 ] ]
    Measured Statistics: {meanExponent=7.206055713278508E-5, negative=0, min=0.0, max=0.0, mean=0.3333886464436849, count=6.0, positive=2, stdDev=0.4714827453418679, zeros=4}
    Feedback Error: [ [ 1.659393310546875E-4, 0.0, 0.0 ], [ 0.0, 1.659393310546875E-4, 0.0 ] ]
    Error Statistics: {meanExponent=-3.7800506649970242, negative=0, min=0.0, max=0.0, mean=5.53131103515625E-5, count=6.0, positive=2, stdDev=7.822455083621932E-5, zeros=4}
    Feedback for input 1
    Inputs Values: [
    	[ [ 1.536, 1.032 ] ]
    ]
    Value Statistics: {meanExponent=0.10003545649334288, negative=0, min=1.032, max=1.032, mean=1.284, count=2.0, positive=2, stdDev=0.25199999999999956, zeros=0}
    Implemented Feedback: [ [ 0.0, 0.0, 1.0 ], [ 0.0, 0.0, 0.0 ] ]
    Implemented Statistics: {meanExponent=0.0, negative=0, min=0.0, max=0.0, mean=0.16666666666666666, count=6.0, positive=1, stdDev=0.37267799624996495, zeros=5}
    Measured Feedback: [ [ 0.0, 0.0, 1.0001659393310547 ], [ 0.0, 0.0, 0.0 ] ]
    Measured Statistics: {meanExponent=7.206055713278508E-5, negative=0, min=0.0, max=0.0, mean=0.16669432322184244, count=6.0, positive=1, stdDev=0.37273983818736145, zeros=5}
    Feedback Error: [ [ 0.0, 0.0, 1.659393310546875E-4 ], [ 0.0, 0.0, 0.0 ] ]
    Error Statistics: {meanExponent=-3.7800506649970242, negative=0, min=0.0, max=0.0, mean=2.765655517578125E-5, count=6.0, positive=1, stdDev=6.184193739652052E-5, zeros=5}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 4.1485e-05 +- 7.1854e-05 [0.0000e+00 - 1.6594e-04] (12#)
    relativeTol: 8.2963e-05 +- 0.0000e+00 [8.2963e-05 - 8.2963e-05] (3#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=4.1485e-05 +- 7.1854e-05 [0.0000e+00 - 1.6594e-04] (12#), relativeTol=8.2963e-05 +- 0.0000e+00 [8.2963e-05 - 8.2963e-05] (3#)}
```



### Performance
Code from [LayerTestBase.java:149](../../../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L149) executed in 0.09 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 3.6772 +- 0.5982 [3.2288 - 8.3214]
    Learning performance: 1.0995 +- 0.4789 [0.9005 - 4.3089]
    
```

