# ImgBandBiasLayer
## ImgBandBiasLayerTest
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
      "class": "com.simiacryptus.mindseye.layers.cudnn.f64.ImgBandBiasLayer",
      "id": "370a9587-74a1-4959-b406-fa4500002b04",
      "isFrozen": false,
      "name": "ImgBandBiasLayer/370a9587-74a1-4959-b406-fa4500002b04",
      "bias": [
        0.0,
        0.0
      ]
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
    	[ [ 1.072, 1.748 ], [ -0.356, -0.932 ], [ -0.312, -1.932 ] ],
    	[ [ 0.6, -0.748 ], [ 1.716, -0.084 ], [ -1.748, -0.148 ] ],
    	[ [ -1.04, 1.984 ], [ -1.212, 1.552 ], [ 1.64, 0.156 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 1.072, 1.748 ], [ -0.356, -0.932 ], [ -0.312, -1.932 ] ],
    	[ [ 0.6, -0.748 ], [ 1.716, -0.084 ], [ -1.748, -0.148 ] ],
    	[ [ -1.04, 1.984 ], [ -1.212, 1.552 ], [ 1.64, 0.156 ] ]
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
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (360#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (360#)}
```



### Differential Validation
Code from [LayerTestBase.java:144](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.04 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ 1.072, 1.748 ], [ -0.356, -0.932 ], [ -0.312, -1.932 ] ],
    	[ [ 0.6, -0.748 ], [ 1.716, -0.084 ], [ -1.748, -0.148 ] ],
    	[ [ -1.04, 1.984 ], [ -1.212, 1.552 ], [ 1.64, 0.156 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.1225358782295855, negative=10, min=0.156, max=0.156, mean=0.10866666666666663, count=18.0, positive=8, stdDev=1.2332326084995753, zeros=0}
    Output: [
    	[ [ 1.072, 1.748 ], [ -0.356, -0.932 ], [ -0.312, -1.932 ] ],
    	[ [ 0.6, -0.748 ], [ 1.716, -0.084 ], [ -1.748, -0.148 ] ],
    	[ [ -1.04, 1.984 ], [ -1.212, 1.552 ], [ 1.64, 0.156 ] ]
    ]
    Outputs Statistics: {meanExponent=-0.1225358782295855, negative=10, min=0.156, max=0.156, mean=0.10866666666666663, count=18.0, positive=8, stdDev=1.2332326084995753, zeros=0}
    Feedback for input 0
    Inputs Values: [
    	[ [ 1.072, 1.748 ], [ -0.356, -0.932 ], [ -0.312, -1.932 ] ],
    	[ [ 0.6, -0.748 ], [ 1.716, -0.084 ], [ -1.748, -0.148 ] ],
    	[ [ -1.04, 1.984 ], [ -1.212, 1.552 ], [ 1.64, 0.156 ] ]
    ]
    Value Statistics: {meanExponent=-0.1225358782295855, negative=10, min=0.156, max=0.156, mean=0.10866666666666663, count=18.0, positive=8, stdDev=1.2332326084995753, zeros=0}
    Implemented Feedback: [ [ 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, ... ], ... ]
    Implemented Statistics: {meanExponent=0.0, negative=0, min=1.0, max=1.0, mean=0.05555555555555555, count=324.0, positive=18, stdDev=0.2290614236454256, zeros=306}
    Measured Feedback: [ [ 0.9999999999998899, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.9999999999998899, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.9999999999998899, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.9999999999998899, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.9999999999998899, 
```
...[skipping 626 bytes](etc/1.txt)...
```
    .0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, -1.1013412404281553E-13, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, -1.1013412404281553E-13, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, -1.1013412404281553E-13, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.1013412404281553E-13, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.1013412404281553E-13, ... ], ... ]
    Error Statistics: {meanExponent=-12.990572096158543, negative=17, min=-1.1013412404281553E-13, max=-1.1013412404281553E-13, mean=-5.690235662631281E-15, count=324.0, positive=1, stdDev=2.4628829153208618E-14, zeros=306}
    Learning Gradient for weight set 0
    Weights: [ 0.0, 0.0 ]
    Implemented Gradient: [ [ 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ] ]
    Implemented Statistics: {meanExponent=0.0, negative=0, min=1.0, max=1.0, mean=0.5, count=36.0, positive=18, stdDev=0.5, zeros=18}
    Measured Gradient: [ [ 0.9999999999998899, 0.9999999999998899, 0.9999999999998899, 0.9999999999998899, 0.9999999999998899, 0.9999999999998899, 0.9999999999998899, 0.9999999999998899, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ] ]
    Measured Statistics: {meanExponent=-4.4482283082179934E-14, negative=0, min=0.9999999999998899, max=0.9999999999998899, mean=0.49999999999994876, count=36.0, positive=18, stdDev=0.4999999999999488, zeros=18}
    Gradient Error: [ [ -1.1013412404281553E-13, -1.1013412404281553E-13, -1.1013412404281553E-13, -1.1013412404281553E-13, -1.1013412404281553E-13, -1.1013412404281553E-13, -1.1013412404281553E-13, -1.1013412404281553E-13, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ] ]
    Error Statistics: {meanExponent=-12.990572096158543, negative=17, min=-1.1013412404281553E-13, max=-1.1013412404281553E-13, mean=-5.1212120963681526E-14, count=36.0, positive=1, stdDev=5.592799596435784E-14, zeros=18}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 1.0561e-14 +- 3.2227e-14 [0.0000e+00 - 1.1013e-13] (360#)
    relativeTol: 5.2803e-14 +- 9.3332e-15 [1.4322e-14 - 5.5067e-14] (36#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=1.0561e-14 +- 3.2227e-14 [0.0000e+00 - 1.1013e-13] (360#), relativeTol=5.2803e-14 +- 9.3332e-15 [1.4322e-14 - 5.5067e-14] (36#)}
```



### Performance
Code from [LayerTestBase.java:149](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L149) executed in 0.10 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 2.8752 +- 0.2030 [2.7557 - 4.3858]
    Learning performance: 3.4345 +- 0.6305 [2.6532 - 7.4152]
    
```

