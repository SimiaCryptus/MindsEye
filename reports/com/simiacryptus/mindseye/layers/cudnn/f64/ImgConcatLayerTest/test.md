# ImgConcatLayer
## ImgConcatLayerTest
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
      "class": "com.simiacryptus.mindseye.layers.cudnn.f64.ImgConcatLayer",
      "id": "370a9587-74a1-4959-b406-fa4500002b17",
      "isFrozen": false,
      "name": "ImgConcatLayer/370a9587-74a1-4959-b406-fa4500002b17",
      "maxBands": -1
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
    	[ [ -1.184 ], [ -1.204 ] ],
    	[ [ -1.732 ], [ 0.508 ] ]
    ],
    [
    	[ [ -0.172 ], [ -1.732 ] ],
    	[ [ 0.02 ], [ -1.804 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ -1.184, -0.172 ], [ -1.204, -1.732 ] ],
    	[ [ -1.732, 0.02 ], [ 0.508, -1.804 ] ]
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
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (160#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (160#)}
```



### Differential Validation
Code from [LayerTestBase.java:144](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.02 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ -1.184 ], [ -1.204 ] ],
    	[ [ -1.732 ], [ 0.508 ] ]
    ],
    [
    	[ [ -0.172 ], [ -1.732 ] ],
    	[ [ 0.02 ], [ -1.804 ] ]
    ]
    Inputs Statistics: {meanExponent=0.02459744731848844, negative=3, min=0.508, max=0.508, mean=-0.903, count=4.0, positive=1, stdDev=0.8437600369773386, zeros=0},
    {meanExponent=-0.4921642841353048, negative=3, min=-1.804, max=-1.804, mean=-0.9219999999999999, count=4.0, positive=1, stdDev=0.849100700741673, zeros=0}
    Output: [
    	[ [ -1.184, -0.172 ], [ -1.204, -1.732 ] ],
    	[ [ -1.732, 0.02 ], [ 0.508, -1.804 ] ]
    ]
    Outputs Statistics: {meanExponent=-0.23378341840840816, negative=6, min=-1.804, max=-1.804, mean=-0.9125000000000001, count=8.0, positive=2, stdDev=0.8464878912305833, zeros=0}
    Feedback for input 0
    Inputs Values: [
    	[ [ -1.184 ], [ -1.204 ] ],
    	[ [ -1.732 ], [ 0.508 ] ]
    ]
    Value Statistics: {meanExponent=0.02459744731848844, negative=3, min=0.508, max=0.508, mean=-0.903, count=4.0, positive=1, stdDev=0.8437600369773386, zeros=0}
    Implemented Feedback: [ [ 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0 ] ]
    Implemented Statistics: {meanExponent=0.0, negative=0, min=0.0, max=0.0, mean=0.125, count=32.0, positive=4, stdDev=0.33071891388307384, zeros=28}
    Measured Feedback: [ [ 0.9999999999998899, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.9999999999998899, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.9999999999998899, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.9999999999998899, 0.0, 0.0, 0.0, 0.0 ] ]
    Measured Statistics: {meanExponent=-4.7830642341045674E-14, negative=0, min=0.0, max=0.0, mean=0.12499999999998623, count=32.0, positive=4, stdDev=0.3307189138830374, zeros=28}
    Feedback Error: [ [ -1.1013412404281553E-13, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, -1.1013412404281553E-13, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, -1.1013412404281553E-13, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, -1.1013412404281553E-13, 0.0, 0.0, 0.0, 0.0 ] ]
    Error Statistics: {meanExponent=-12.958078098036825, negative=4, min=0.0, max=0.0, mean=-1.3766765505351941E-14, count=32.0, positive=0, stdDev=3.6423437884903677E-14, zeros=28}
    Feedback for input 1
    Inputs Values: [
    	[ [ -0.172 ], [ -1.732 ] ],
    	[ [ 0.02 ], [ -1.804 ] ]
    ]
    Value Statistics: {meanExponent=-0.4921642841353048, negative=3, min=-1.804, max=-1.804, mean=-0.9219999999999999, count=4.0, positive=1, stdDev=0.849100700741673, zeros=0}
    Implemented Feedback: [ [ 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0 ] ]
    Implemented Statistics: {meanExponent=0.0, negative=0, min=1.0, max=1.0, mean=0.125, count=32.0, positive=4, stdDev=0.33071891388307384, zeros=28}
    Measured Feedback: [ [ 0.0, 0.0, 0.0, 0.0, 0.9999999999998899, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.999999999999994, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9999999999998899, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9999999999998899 ] ]
    Measured Statistics: {meanExponent=-3.652390279570773E-14, negative=0, min=0.9999999999998899, max=0.9999999999998899, mean=0.12499999999998948, count=32.0, positive=4, stdDev=0.33071891388304603, zeros=28}
    Feedback Error: [ [ 0.0, 0.0, 0.0, 0.0, -1.1013412404281553E-13, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, -5.995204332975845E-15, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.1013412404281553E-13, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.1013412404281553E-13 ] ]
    Error Statistics: {meanExponent=-13.274107576119626, negative=4, min=-1.1013412404281553E-13, max=-1.1013412404281553E-13, mean=-1.0512424264419451E-14, count=32.0, positive=0, stdDev=3.205862026514086E-14, zeros=28}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 1.2140e-14 +- 3.4349e-14 [0.0000e+00 - 1.1013e-13] (64#)
    relativeTol: 4.8558e-14 +- 1.7220e-14 [2.9976e-15 - 5.5067e-14] (8#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=1.2140e-14 +- 3.4349e-14 [0.0000e+00 - 1.1013e-13] (64#), relativeTol=4.8558e-14 +- 1.7220e-14 [2.9976e-15 - 5.5067e-14] (8#)}
```



### Performance
Code from [LayerTestBase.java:149](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L149) executed in 0.08 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 3.7551 +- 0.7208 [3.0664 - 6.9649]
    Learning performance: 0.9779 +- 0.1103 [0.8720 - 1.4363]
    
```

