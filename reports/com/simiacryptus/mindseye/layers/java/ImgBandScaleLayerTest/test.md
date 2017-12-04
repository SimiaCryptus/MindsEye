# ImgBandScaleLayer
## ImgBandScaleLayerTest
### Json Serialization
Code from [LayerTestBase.java:83](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L83) executed in 0.00 seconds: 
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
      "id": "a864e734-2f23-44db-97c1-504000002bd2",
      "isFrozen": false,
      "name": "ImgBandScaleLayer/a864e734-2f23-44db-97c1-504000002bd2",
      "bias": [
        0.0,
        0.0,
        0.0
      ]
    }
```



### Example Input/Output Pair
Code from [LayerTestBase.java:120](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L120) executed in 0.00 seconds: 
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
    	[ [ 0.732, 1.312, 0.596 ], [ 0.772, 0.624, 0.272 ] ],
    	[ [ 1.7, 0.292, 0.396 ], [ 0.568, 0.0, -1.764 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ] ],
    	[ [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, -0.0 ] ]
    ]
```



### Batch Execution
Code from [LayerTestBase.java:138](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L138) executed in 0.00 seconds: 
```java
    BatchingTester batchingTester = getBatchingTester();
    return batchingTester==null?null:batchingTester.test(layer, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (240#), relativeTol=0.0000e+00 +- 0.0000e+00 [Infinity - -Infinity] (0#)}
```



### Differential Validation
Code from [LayerTestBase.java:144](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.00 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ 0.732, 1.312, 0.596 ], [ 0.772, 0.624, 0.272 ] ],
    	[ [ 1.7, 0.292, 0.396 ], [ 0.568, 0.0, -1.764 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.16641492316821335, negative=1, min=-1.764, max=-1.764, mean=0.45833333333333326, count=12.0, positive=10, stdDev=0.8019400781161202, zeros=1}
    Output: [
    	[ [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ] ],
    	[ [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, -0.0 ] ]
    ]
    Outputs Statistics: {meanExponent=NaN, negative=0, min=-0.0, max=-0.0, mean=0.0, count=12.0, positive=0, stdDev=0.0, zeros=12}
    Feedback for input 0
    Inputs Values: [
    	[ [ 0.732, 1.312, 0.596 ], [ 0.772, 0.624, 0.272 ] ],
    	[ [ 1.7, 0.292, 0.396 ], [ 0.568, 0.0, -1.764 ] ]
    ]
    Value Statistics: {meanExponent=-0.16641492316821335, negative=1, min=-1.764, max=-1.764, mean=0.45833333333333326, count=12.0, positive=10, stdDev=0.8019400781161202, zeros=1}
    Implemented Feedback: [ [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], ... ]
    Implemented Statistics: {meanExponent=NaN, negative=0, min=0.0, max=0.0, mean=0.0, count=144.0, positive=0, stdDev=0.0, zeros=144}
    Measured Feedback: [ [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], ... ]
    Measured Statistics: {meanExponent=NaN, negative=0, min=0.0, max=0.0, mean=0.0, count=144.0, positive=0, stdDev=0.0, zeros=144}
    Feedback Error: [ [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], ... ]
    Error Statistics: {meanExponent=NaN, negative=0, min=0.0, max=0.0, mean=0.0, count=144.0, positive=0, stdDev=0.0, zeros=144}
    Learning Gradient for weight set 0
    Weights: [ 0.0, 0.0, 0.0 ]
    Implemented Gradient: [ [ 0.732, 1.7, 0.772, 0.568, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 1.312, 0.292, 0.624, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ] ]
    Implemented Statistics: {meanExponent=-0.16641492316821335, negative=1, min=-1.764, max=-1.764, mean=0.15277777777777776, count=36.0, positive=10, stdDev=0.5109318888675167, zeros=25}
    Measured Gradient: [ [ 0.732, 1.7000000000000002, 0.772, 0.568, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 1.3120000000000003, 0.292, 0.624, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ] ]
    Measured Statistics: {meanExponent=-0.16641492316821335, negative=1, min=-1.764, max=-1.764, mean=0.1527777777777778, count=36.0, positive=10, stdDev=0.5109318888675167, zeros=25}
    Gradient Error: [ [ 0.0, 2.220446049250313E-16, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 2.220446049250313E-16, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ] ]
    Error Statistics: {meanExponent=-15.854246438303008, negative=0, min=0.0, max=0.0, mean=1.3877787807814457E-17, count=36.0, positive=3, stdDev=5.1304037436923246E-17, zeros=33}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 2.7756e-18 +- 2.3606e-17 [0.0000e+00 - 2.2204e-16] (180#)
    relativeTol: 2.0002e-17 +- 3.2943e-17 [0.0000e+00 - 8.4621e-17] (11#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=2.7756e-18 +- 2.3606e-17 [0.0000e+00 - 2.2204e-16] (180#), relativeTol=2.0002e-17 +- 3.2943e-17 [0.0000e+00 - 8.4621e-17] (11#)}
```



### Performance
Code from [LayerTestBase.java:149](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L149) executed in 0.01 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 0.1977 +- 0.0318 [0.1653 - 0.3391]
    Learning performance: 0.0521 +- 0.0356 [0.0371 - 0.3733]
    
```

