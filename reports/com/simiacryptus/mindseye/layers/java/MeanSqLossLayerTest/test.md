# MeanSqLossLayer
## MeanSqLossLayerTest
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
      "class": "com.simiacryptus.mindseye.layers.java.MeanSqLossLayer",
      "id": "370a9587-74a1-4959-b406-fa4500002c2f",
      "isFrozen": false,
      "name": "MeanSqLossLayer/370a9587-74a1-4959-b406-fa4500002c2f"
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
    	[ [ 0.956 ], [ 0.172 ], [ 1.912 ] ],
    	[ [ 0.208 ], [ -0.976 ], [ 0.832 ] ]
    ],
    [
    	[ [ -0.944 ], [ -1.36 ], [ 1.46 ] ],
    	[ [ 0.276 ], [ 0.544 ], [ 1.988 ] ]
    ]]
    --------------------
    Output: 
    [ 1.6354480000000002 ]
```



### Batch Execution
Code from [LayerTestBase.java:138](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L138) executed in 0.00 seconds: 
```java
    BatchingTester batchingTester = getBatchingTester();
    return batchingTester==null?null:batchingTester.test(layer, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (130#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (130#)}
```



### Differential Validation
Code from [LayerTestBase.java:144](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.00 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ 0.956 ], [ 0.172 ], [ 1.912 ] ],
    	[ [ 0.208 ], [ -0.976 ], [ 0.832 ] ]
    ],
    [
    	[ [ -0.944 ], [ -1.36 ], [ 1.46 ] ],
    	[ [ 0.276 ], [ 0.544 ], [ 1.988 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.2124815489926821, negative=1, min=0.832, max=0.832, mean=0.5173333333333333, count=6.0, positive=5, stdDev=0.8833199998993694, zeros=0},
    {meanExponent=-0.04203531328709737, negative=2, min=1.988, max=1.988, mean=0.3273333333333333, count=6.0, positive=4, stdDev=1.193992555346231, zeros=0}
    Output: [ 1.6354480000000002 ]
    Outputs Statistics: {meanExponent=0.21363674004028524, negative=0, min=1.6354480000000002, max=1.6354480000000002, mean=1.6354480000000002, count=1.0, positive=1, stdDev=0.0, zeros=0}
    Feedback for input 0
    Inputs Values: [
    	[ [ 0.956 ], [ 0.172 ], [ 1.912 ] ],
    	[ [ 0.208 ], [ -0.976 ], [ 0.832 ] ]
    ]
    Value Statistics: {meanExponent=-0.2124815489926821, negative=1, min=0.832, max=0.832, mean=0.5173333333333333, count=6.0, positive=5, stdDev=0.8833199998993694, zeros=0}
    Implemented Feedback: [ [ 0.6333333333333333 ], [ -0.022666666666666675 ], [ 0.5106666666666666 ], [ -0.5066666666666666 ], [ 0.15066666666666664 ], [ -0.38533333333333336 ] ]
    Implemented Statistics: {meanExponent=-0.6110443987536099, negative=3, min=-0.38533333333333336, max=-0.38533333333333336, mean=0.06333333333333332, count=6.0, positive=3, stdDev=0.4215511040589662, zeros=0}
    Measured Feedback: [ [ 0.6333500000010872 ], [ -0.022650000000012938 ], [ 0.5106833333345939 ], [ -0.5066499999983876 ], [ 0.15068333333312367 ], [ -0.3853166666689667 ] ]
    Measured Statistics: {meanExponent=-0.6110908789789383, negative=3, min=-0.3853166666689667, max=-0.3853166666689667, mean=0.0633500000002396, count=6.0, positive=3, stdDev=0.421551104059472, zeros=0}
    Feedback Error: [ [ 1.666666775390624E-5 ], [ 1.6666666653737672E-5 ], [ 1.6666667927323076E-5 ], [ 1.666666827904173E-5 ], [ 1.666666645702697E-5 ], [ 1.6666664366671302E-5 ] ]
    Error Statistics: {meanExponent=-4.778151244139763, negative=0, min=1.6666664366671302E-5, max=1.6666664366671302E-5, mean=1.66666669062845E-5, count=6.0, positive=6, stdDev=1.306162322757252E-12, zeros=0}
    Feedback for input 1
    Inputs Values: [
    	[ [ -0.944 ], [ -1.36 ], [ 1.46 ] ],
    	[ [ 0.276 ], [ 0.544 ], [ 1.988 ] ]
    ]
    Value Statistics: {meanExponent=-0.04203531328709737, negative=2, min=1.988, max=1.988, mean=0.3273333333333333, count=6.0, positive=4, stdDev=1.193992555346231, zeros=0}
    Implemented Feedback: [ [ -0.6333333333333333 ], [ 0.022666666666666675 ], [ -0.5106666666666666 ], [ 0.5066666666666666 ], [ -0.15066666666666664 ], [ 0.38533333333333336 ] ]
    Implemented Statistics: {meanExponent=-0.6110443987536099, negative=3, min=0.38533333333333336, max=0.38533333333333336, mean=-0.06333333333333332, count=6.0, positive=3, stdDev=0.4215511040589662, zeros=0}
    Measured Feedback: [ [ -0.6333166666672163 ], [ 0.022683333333883837 ], [ -0.510650000000723 ], [ 0.5066833333344789 ], [ -0.15065000000147322 ], [ 0.3853499999983967 ] ]
    Measured Statistics: {meanExponent=-0.6109979588861254, negative=3, min=0.3853499999983967, max=0.3853499999983967, mean=-0.06331666666710885, count=6.0, positive=3, stdDev=0.4215511040592613, zeros=0}
    Feedback Error: [ [ 1.666666611699341E-5 ], [ 1.666666721716198E-5 ], [ 1.6666665943576575E-5 ], [ 1.666666781230397E-5 ], [ 1.6666665193426633E-5 ], [ 1.666666506333625E-5 ] ]
    Error Statistics: {meanExponent=-4.778151261906351, negative=0, min=1.666666506333625E-5, max=1.666666506333625E-5, mean=1.666666622446647E-5, count=6.0, positive=6, stdDev=1.016845989170083E-12, zeros=0}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 1.6667e-05 +- 1.2454e-12 [1.6667e-05 - 1.6667e-05] (12#)
    relativeTol: 8.1751e-05 +- 1.2865e-04 [1.3158e-05 - 3.6778e-04] (12#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=1.6667e-05 +- 1.2454e-12 [1.6667e-05 - 1.6667e-05] (12#), relativeTol=8.1751e-05 +- 1.2865e-04 [1.3158e-05 - 3.6778e-04] (12#)}
```



### Performance
Code from [LayerTestBase.java:149](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L149) executed in 0.01 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 0.3247 +- 0.0671 [0.1881 - 0.6640]
    Learning performance: 0.0052 +- 0.0023 [0.0029 - 0.0142]
    
```

