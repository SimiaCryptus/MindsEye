# ImgBandBiasLayer
## ImgBandBiasLayerTest
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
      "class": "com.simiacryptus.mindseye.layers.cudnn.f64.ImgBandBiasLayer",
      "id": "a864e734-2f23-44db-97c1-504000002b04",
      "isFrozen": false,
      "name": "ImgBandBiasLayer/a864e734-2f23-44db-97c1-504000002b04",
      "bias": [
        0.0,
        0.0
      ]
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
    	[ [ -0.188, 0.736 ], [ 0.632, 1.704 ], [ 1.14, -0.928 ] ],
    	[ [ 1.78, -1.304 ], [ -1.744, 1.524 ], [ 1.968, -0.052 ] ],
    	[ [ 1.304, 1.84 ], [ 0.44, -0.712 ], [ -0.088, 1.304 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ -0.188, 0.736 ], [ 0.632, 1.704 ], [ 1.14, -0.928 ] ],
    	[ [ 1.78, -1.304 ], [ -1.744, 1.524 ], [ 1.968, -0.052 ] ],
    	[ [ 1.304, 1.84 ], [ 0.44, -0.712 ], [ -0.088, 1.304 ] ]
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
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (360#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (360#)}
```



### Differential Validation
Code from [LayerTestBase.java:144](../../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.05 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ -0.188, 0.736 ], [ 0.632, 1.704 ], [ 1.14, -0.928 ] ],
    	[ [ 1.78, -1.304 ], [ -1.744, 1.524 ], [ 1.968, -0.052 ] ],
    	[ [ 1.304, 1.84 ], [ 0.44, -0.712 ], [ -0.088, 1.304 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.11479327850115673, negative=7, min=1.304, max=1.304, mean=0.5197777777777778, count=18.0, positive=11, stdDev=1.1249987434835236, zeros=0}
    Output: [
    	[ [ -0.188, 0.736 ], [ 0.632, 1.704 ], [ 1.14, -0.928 ] ],
    	[ [ 1.78, -1.304 ], [ -1.744, 1.524 ], [ 1.968, -0.052 ] ],
    	[ [ 1.304, 1.84 ], [ 0.44, -0.712 ], [ -0.088, 1.304 ] ]
    ]
    Outputs Statistics: {meanExponent=-0.11479327850115673, negative=7, min=1.304, max=1.304, mean=0.5197777777777778, count=18.0, positive=11, stdDev=1.1249987434835236, zeros=0}
    Feedback for input 0
    Inputs Values: [
    	[ [ -0.188, 0.736 ], [ 0.632, 1.704 ], [ 1.14, -0.928 ] ],
    	[ [ 1.78, -1.304 ], [ -1.744, 1.524 ], [ 1.968, -0.052 ] ],
    	[ [ 1.304, 1.84 ], [ 0.44, -0.712 ], [ -0.088, 1.304 ] ]
    ]
    Value Statistics: {meanExponent=-0.11479327850115673, negative=7, min=1.304, max=1.304, mean=0.5197777777777778, count=18.0, positive=11, stdDev=1.1249987434835236, zeros=0}
    Implemented Feedback: [ [ 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, ... ], ... ]
    Implemented Statistics: {meanExponent=0.0, negative=0, min=1.0, max=1.0, mean=0.05555555555555555, count=324.0, positive=18, stdDev=0.2290614236454256, zeros=306}
    Measured Feedback: [ [ 0.9999999999998899, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.9999999999998899, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.9999999999998899, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.9999999999998899, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.9999999999998899, 0.0, 0.0,
```
...[skipping 616 bytes](etc/1.txt)...
```
     0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, -1.1013412404281553E-13, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, -1.1013412404281553E-13, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, -1.1013412404281553E-13, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.1013412404281553E-13, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.1013412404281553E-13, ... ], ... ]
    Error Statistics: {meanExponent=-13.023066094280262, negative=16, min=-1.1013412404281553E-13, max=-1.1013412404281553E-13, mean=-5.261908878439477E-15, count=324.0, positive=2, stdDev=2.400761991912802E-14, zeros=306}
    Learning Gradient for weight set 0
    Weights: [ 0.0, 0.0 ]
    Implemented Gradient: [ [ 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ] ]
    Implemented Statistics: {meanExponent=0.0, negative=0, min=1.0, max=1.0, mean=0.5, count=36.0, positive=18, stdDev=0.5, zeros=18}
    Measured Gradient: [ [ 0.9999999999998899, 0.9999999999998899, 0.9999999999998899, 0.9999999999998899, 0.9999999999998899, 0.9999999999998899, 0.9999999999998899, 0.9999999999998899, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ] ]
    Measured Statistics: {meanExponent=-4.1133923823314226E-14, negative=0, min=0.9999999999998899, max=0.9999999999998899, mean=0.49999999999995265, count=36.0, positive=18, stdDev=0.4999999999999526, zeros=18}
    Gradient Error: [ [ -1.1013412404281553E-13, -1.1013412404281553E-13, -1.1013412404281553E-13, -1.1013412404281553E-13, -1.1013412404281553E-13, -1.1013412404281553E-13, -1.1013412404281553E-13, -1.1013412404281553E-13, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ] ]
    Error Statistics: {meanExponent=-13.023066094280262, negative=16, min=-1.1013412404281553E-13, max=-1.1013412404281553E-13, mean=-4.735717990595529E-14, count=36.0, positive=2, stdDev=5.651352939245432E-14, zeros=18}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 1.0108e-14 +- 3.1387e-14 [0.0000e+00 - 1.1013e-13] (360#)
    relativeTol: 5.0540e-14 +- 1.2805e-14 [1.4322e-14 - 5.5067e-14] (36#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=1.0108e-14 +- 3.1387e-14 [0.0000e+00 - 1.1013e-13] (360#), relativeTol=5.0540e-14 +- 1.2805e-14 [1.4322e-14 - 5.5067e-14] (36#)}
```



### Performance
Code from [LayerTestBase.java:149](../../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L149) executed in 0.11 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 3.6753 +- 1.2966 [2.6475 - 11.1997]
    Learning performance: 3.7073 +- 0.7295 [2.8156 - 7.5320]
    
```

