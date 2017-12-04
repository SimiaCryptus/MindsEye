# ImgBandScaleLayer
## ImgBandScaleLayerTest
### Json Serialization
Code from [LayerTestBase.java:83](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L83) executed in 0.00 seconds: 
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
      "class": "com.simiacryptus.mindseye.layers.java.ImgBandScaleLayer",
      "id": "370a9587-74a1-4959-b406-fa4500002bd2",
      "isFrozen": false,
      "name": "ImgBandScaleLayer/370a9587-74a1-4959-b406-fa4500002bd2",
      "bias": [
        0.0,
        0.0,
        0.0
      ]
    }
```



### Example Input/Output Pair
Code from [LayerTestBase.java:120](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L120) executed in 0.00 seconds: 
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
    	[ [ -0.608, -1.628, -0.636 ], [ -0.12, 0.092, -0.996 ] ],
    	[ [ 0.864, 1.84, -0.1 ], [ 1.272, 1.852, -0.892 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ -0.0, -0.0, -0.0 ], [ -0.0, 0.0, -0.0 ] ],
    	[ [ 0.0, 0.0, -0.0 ], [ 0.0, 0.0, -0.0 ] ]
    ]
```



### Batch Execution
Code from [LayerTestBase.java:138](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L138) executed in 0.00 seconds: 
```java
    BatchingTester batchingTester = getBatchingTester();
    return batchingTester==null?null:batchingTester.test(layer, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (240#), relativeTol=0.0000e+00 +- 0.0000e+00 [Infinity - -Infinity] (0#)}
```



### Differential Validation
Code from [LayerTestBase.java:144](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.01 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ -0.608, -1.628, -0.636 ], [ -0.12, 0.092, -0.996 ] ],
    	[ [ 0.864, 1.84, -0.1 ], [ 1.272, 1.852, -0.892 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.21966099826549387, negative=7, min=-0.892, max=-0.892, mean=0.07833333333333332, count=12.0, positive=5, stdDev=1.0935964622392587, zeros=0}
    Output: [
    	[ [ -0.0, -0.0, -0.0 ], [ -0.0, 0.0, -0.0 ] ],
    	[ [ 0.0, 0.0, -0.0 ], [ 0.0, 0.0, -0.0 ] ]
    ]
    Outputs Statistics: {meanExponent=NaN, negative=0, min=-0.0, max=-0.0, mean=0.0, count=12.0, positive=0, stdDev=0.0, zeros=12}
    Feedback for input 0
    Inputs Values: [
    	[ [ -0.608, -1.628, -0.636 ], [ -0.12, 0.092, -0.996 ] ],
    	[ [ 0.864, 1.84, -0.1 ], [ 1.272, 1.852, -0.892 ] ]
    ]
    Value Statistics: {meanExponent=-0.21966099826549387, negative=7, min=-0.892, max=-0.892, mean=0.07833333333333332, count=12.0, positive=5, stdDev=1.0935964622392587, zeros=0}
    Implemented Feedback: [ [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], ... ]
    Implemented Statistics: {meanExponent=NaN, negative=0, min=0.0, max=0.0, mean=0.0, count=144.0, positive=0, stdDev=0.0, zeros=144}
    Measured Feedback: [ [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], ... ]
    Measured Statistics: {meanExponent=NaN, negative=0, min=0.0, max=0.0, mean=0.0, count=144.0, positive=0, stdDev=0.0, zeros=144}
    Feedback Error: [ [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], ... ]
    Error Statistics: {meanExponent=NaN, negative=0, min=0.0, max=0.0, mean=0.0, count=144.0, positive=0, stdDev=0.0, zeros=144}
    Learning Gradient for weight set 0
    Weights: [ 0.0, 0.0, 0.0 ]
    Implemented Gradient: [ [ -0.608, 0.864, -0.12, 1.272, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, -1.628, 1.84, 0.092, 1.852, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ] ]
    Implemented Statistics: {meanExponent=-0.21966099826549387, negative=7, min=-0.892, max=-0.892, mean=0.026111111111111106, count=36.0, positive=5, stdDev=0.63246711718554, zeros=24}
    Measured Gradient: [ [ -0.608, 0.864, -0.12000000000000001, 1.272, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, -1.628, 1.8400000000000003, 0.092, 1.8520000000000003, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ] ]
    Measured Statistics: {meanExponent=-0.21966099826549387, negative=7, min=-0.892, max=-0.892, mean=0.02611111111111113, count=36.0, positive=5, stdDev=0.63246711718554, zeros=24}
    Gradient Error: [ [ 0.0, 0.0, -1.3877787807814457E-17, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 2.220446049250313E-16, 0.0, 2.220446049250313E-16, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ] ]
    Error Statistics: {meanExponent=-16.029847269106998, negative=2, min=0.0, max=0.0, mean=8.866364432770347E-18, count=36.0, positive=2, stdDev=5.484729070944123E-17, zeros=32}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 3.1611e-18 +- 2.4645e-17 [0.0000e+00 - 2.2204e-16] (180#)
    relativeTol: 1.9487e-17 +- 2.7579e-17 [0.0000e+00 - 6.0338e-17] (12#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=3.1611e-18 +- 2.4645e-17 [0.0000e+00 - 2.2204e-16] (180#), relativeTol=1.9487e-17 +- 2.7579e-17 [0.0000e+00 - 6.0338e-17] (12#)}
```



### Performance
Code from [LayerTestBase.java:149](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L149) executed in 0.01 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 0.2293 +- 0.0836 [0.1539 - 0.7096]
    Learning performance: 0.0541 +- 0.0183 [0.0371 - 0.1596]
    
```

