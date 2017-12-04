# ImgConcatLayer
## ImgConcatLayerTest
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
      "class": "com.simiacryptus.mindseye.layers.java.ImgConcatLayer",
      "id": "a864e734-2f23-44db-97c1-504000002be8",
      "isFrozen": false,
      "name": "ImgConcatLayer/a864e734-2f23-44db-97c1-504000002be8"
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
    	[ [ -1.364 ], [ -0.124 ] ],
    	[ [ -1.696 ], [ 1.132 ] ]
    ],
    [
    	[ [ 1.012 ], [ -0.568 ] ],
    	[ [ -1.284 ], [ -0.332 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ -1.364, 1.012 ], [ -0.124, -0.568 ] ],
    	[ [ -1.696, -1.284 ], [ 1.132, -0.332 ] ]
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
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (160#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (160#)}
```



### Differential Validation
Code from [LayerTestBase.java:144](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.00 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ -1.364 ], [ -0.124 ] ],
    	[ [ -1.696 ], [ 1.132 ] ]
    ],
    [
    	[ [ 1.012 ], [ -0.568 ] ],
    	[ [ -1.284 ], [ -0.332 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.1221229174360893, negative=3, min=1.132, max=1.132, mean=-0.5130000000000001, count=4.0, positive=1, stdDev=1.1159117348607817, zeros=0},
    {meanExponent=-0.15269201108708252, negative=3, min=-0.332, max=-0.332, mean=-0.293, count=4.0, positive=1, stdDev=0.8310012033685631, zeros=0}
    Output: [
    	[ [ -1.364, 1.012 ], [ -0.124, -0.568 ] ],
    	[ [ -1.696, -1.284 ], [ 1.132, -0.332 ] ]
    ]
    Outputs Statistics: {meanExponent=-0.1374074642615859, negative=6, min=-0.332, max=-0.332, mean=-0.4030000000000001, count=8.0, positive=2, stdDev=0.9899550494845712, zeros=0}
    Feedback for input 0
    Inputs Values: [
    	[ [ -1.364 ], [ -0.124 ] ],
    	[ [ -1.696 ], [ 1.132 ] ]
    ]
    Value Statistics: {meanExponent=-0.1221229174360893, negative=3, min=1.132, max=1.132, mean=-0.5130000000000001, count=4.0, positive=1, stdDev=1.1159117348607817, zeros=0}
    Implemented Feedback: [ [ 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0 ] ]
    Implemented Statistics: {meanExponent=0.0, negative=0, min=0.0, max=0.0, mean=0.125, count=32.0, positive=4, stdDev=0.33071891388307384, zeros=28}
    Measured Feedback: [ [ 0.9999999999998899, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.9999999999998899, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 1.0000000000000286, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.9999999999998899, 0.0, 0.0, 0.0, 0.0 ] ]
    Measured Statistics: {meanExponent=-3.276302567614995E-14, negative=0, min=0.0, max=0.0, mean=0.12499999999999056, count=32.0, positive=4, stdDev=0.33071891388304886, zeros=28}
    Feedback Error: [ [ -1.1013412404281553E-13, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, -1.1013412404281553E-13, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 2.864375403532904E-14, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, -1.1013412404281553E-13, 0.0, 0.0, 0.0, 0.0 ] ]
    Error Statistics: {meanExponent=-13.104301089584563, negative=3, min=0.0, max=0.0, mean=-9.429956815409923E-15, count=32.0, positive=1, stdDev=3.2769779210413227E-14, zeros=28}
    Feedback for input 1
    Inputs Values: [
    	[ [ 1.012 ], [ -0.568 ] ],
    	[ [ -1.284 ], [ -0.332 ] ]
    ]
    Value Statistics: {meanExponent=-0.15269201108708252, negative=3, min=-0.332, max=-0.332, mean=-0.293, count=4.0, positive=1, stdDev=0.8310012033685631, zeros=0}
    Implemented Feedback: [ [ 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0 ] ]
    Implemented Statistics: {meanExponent=0.0, negative=0, min=1.0, max=1.0, mean=0.125, count=32.0, positive=4, stdDev=0.33071891388307384, zeros=28}
    Measured Feedback: [ [ 0.0, 0.0, 0.0, 0.0, 0.9999999999998899, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.9999999999998899, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9999999999998899, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9999999999998899 ] ]
    Measured Statistics: {meanExponent=-4.7830642341045674E-14, negative=0, min=0.9999999999998899, max=0.9999999999998899, mean=0.12499999999998623, count=32.0, positive=4, stdDev=0.3307189138830374, zeros=28}
    Feedback Error: [ [ 0.0, 0.0, 0.0, 0.0, -1.1013412404281553E-13, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, -1.1013412404281553E-13, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.1013412404281553E-13, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.1013412404281553E-13 ] ]
    Error Statistics: {meanExponent=-12.958078098036825, negative=4, min=-1.1013412404281553E-13, max=-1.1013412404281553E-13, mean=-1.3766765505351941E-14, count=32.0, positive=0, stdDev=3.6423437884903677E-14, zeros=28}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 1.2493e-14 +- 3.4401e-14 [0.0000e+00 - 1.1013e-13] (64#)
    relativeTol: 4.9974e-14 +- 1.3475e-14 [1.4322e-14 - 5.5067e-14] (8#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=1.2493e-14 +- 3.4401e-14 [0.0000e+00 - 1.1013e-13] (64#), relativeTol=4.9974e-14 +- 1.3475e-14 [1.4322e-14 - 5.5067e-14] (8#)}
```



### Performance
Code from [LayerTestBase.java:149](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L149) executed in 0.00 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 0.1917 +- 0.0737 [0.1482 - 0.6213]
    Learning performance: 0.0370 +- 0.0129 [0.0285 - 0.1283]
    
```

