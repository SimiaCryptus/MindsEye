# ImgBandBiasLayer
## ImgBandBiasLayerTest
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
      "class": "com.simiacryptus.mindseye.layers.java.ImgBandBiasLayer",
      "id": "370a9587-74a1-4959-b406-fa4500002bc5",
      "isFrozen": false,
      "name": "ImgBandBiasLayer/370a9587-74a1-4959-b406-fa4500002bc5",
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
    	[ [ -0.504, 0.052, 1.856 ], [ 1.296, -0.052, 1.132 ] ],
    	[ [ 1.268, -1.424, -1.02 ], [ 0.64, -1.144, 0.956 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ -0.504, 0.052, 1.856 ], [ 1.296, -0.052, 1.132 ] ],
    	[ [ 1.268, -1.424, -1.02 ], [ 0.64, -1.144, 0.956 ] ]
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
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (240#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (240#)}
```



### Differential Validation
Code from [LayerTestBase.java:144](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.00 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ -0.504, 0.052, 1.856 ], [ 1.296, -0.052, 1.132 ] ],
    	[ [ 1.268, -1.424, -1.02 ], [ 0.64, -1.144, 0.956 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.19335333922414535, negative=5, min=0.956, max=0.956, mean=0.2546666666666667, count=12.0, positive=7, stdDev=1.0486185621516, zeros=0}
    Output: [
    	[ [ -0.504, 0.052, 1.856 ], [ 1.296, -0.052, 1.132 ] ],
    	[ [ 1.268, -1.424, -1.02 ], [ 0.64, -1.144, 0.956 ] ]
    ]
    Outputs Statistics: {meanExponent=-0.19335333922414535, negative=5, min=0.956, max=0.956, mean=0.2546666666666667, count=12.0, positive=7, stdDev=1.0486185621516, zeros=0}
    Feedback for input 0
    Inputs Values: [
    	[ [ -0.504, 0.052, 1.856 ], [ 1.296, -0.052, 1.132 ] ],
    	[ [ 1.268, -1.424, -1.02 ], [ 0.64, -1.144, 0.956 ] ]
    ]
    Value Statistics: {meanExponent=-0.19335333922414535, negative=5, min=0.956, max=0.956, mean=0.2546666666666667, count=12.0, positive=7, stdDev=1.0486185621516, zeros=0}
    Implemented Feedback: [ [ 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, ... ], ... ]
    Implemented Statistics: {meanExponent=0.0, negative=0, min=1.0, max=1.0, mean=0.08333333333333333, count=144.0, positive=12, stdDev=0.2763853991962833, zeros=132}
    Measured Feedback: [ [ 0.9999999999998899, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.9999999999998899, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.9999999999998899, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.9999999999998899, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 1.0000000000000286, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.9999999999998899, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0000000000000286, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9999999999998899, ... ], ... ]
    M
```
...[skipping 580 bytes](etc/1.txt)...
```
    .0, 0.0, 0.0, 0.0, -1.1013412404281553E-13, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.864375403532904E-14, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.1013412404281553E-13, ... ], ... ]
    Error Statistics: {meanExponent=-13.055560092401983, negative=10, min=-1.1013412404281553E-13, max=-1.1013412404281553E-13, mean=-7.250373141371509E-15, count=144.0, positive=2, stdDev=2.830469176582722E-14, zeros=132}
    Learning Gradient for weight set 0
    Weights: [ 0.0, 0.0, 0.0 ]
    Implemented Gradient: [ [ 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ] ]
    Implemented Statistics: {meanExponent=0.0, negative=0, min=1.0, max=1.0, mean=0.3333333333333333, count=36.0, positive=12, stdDev=0.4714045207910317, zeros=24}
    Measured Gradient: [ [ 0.9999999999998899, 0.9999999999998899, 0.9999999999998899, 0.9999999999998899, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 1.0000000000000286, 0.9999999999998899, 1.0000000000000286, 0.9999999999998899, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ] ]
    Measured Statistics: {meanExponent=-3.7785564564448525E-14, negative=0, min=0.9999999999998899, max=0.9999999999998899, mean=0.33333333333330434, count=36.0, positive=12, stdDev=0.47140452079099066, zeros=24}
    Gradient Error: [ [ -1.1013412404281553E-13, -1.1013412404281553E-13, -1.1013412404281553E-13, -1.1013412404281553E-13, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 2.864375403532904E-14, -1.1013412404281553E-13, 2.864375403532904E-14, -1.1013412404281553E-13, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ] ]
    Error Statistics: {meanExponent=-13.055560092401983, negative=10, min=-1.1013412404281553E-13, max=-1.1013412404281553E-13, mean=-2.9001492565486034E-14, count=36.0, positive=2, stdDev=5.0732705186738195E-14, zeros=24}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 1.2874e-14 +- 3.4644e-14 [0.0000e+00 - 1.1013e-13] (180#)
    relativeTol: 4.8276e-14 +- 1.5185e-14 [1.4322e-14 - 5.5067e-14] (24#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=1.2874e-14 +- 3.4644e-14 [0.0000e+00 - 1.1013e-13] (180#), relativeTol=4.8276e-14 +- 1.5185e-14 [1.4322e-14 - 5.5067e-14] (24#)}
```



### Performance
Code from [LayerTestBase.java:149](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L149) executed in 0.01 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 0.2055 +- 0.0755 [0.1453 - 0.7467]
    Learning performance: 0.0655 +- 0.0317 [0.0427 - 0.2593]
    
```

