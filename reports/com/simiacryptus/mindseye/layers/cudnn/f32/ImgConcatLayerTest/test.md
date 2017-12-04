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
      "class": "com.simiacryptus.mindseye.layers.cudnn.f32.ImgConcatLayer",
      "id": "370a9587-74a1-4959-b406-fa45000003c0",
      "isFrozen": false,
      "name": "ImgConcatLayer/370a9587-74a1-4959-b406-fa45000003c0",
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
    	[ [ 0.128 ], [ -1.18 ] ],
    	[ [ 1.316 ], [ 1.068 ] ]
    ],
    [
    	[ [ 1.128 ], [ -1.068 ] ],
    	[ [ -1.796 ], [ -0.004 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 0.12800000607967377, 1.128000020980835 ], [ -1.1799999475479126, -1.0679999589920044 ] ],
    	[ [ 1.315999984741211, -1.7960000038146973 ], [ 1.0679999589920044, -0.004000000189989805 ] ]
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
Code from [LayerTestBase.java:144](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.03 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ 0.128 ], [ -1.18 ] ],
    	[ [ 1.316 ], [ 1.068 ] ]
    ],
    [
    	[ [ 1.128 ], [ -1.068 ] ],
    	[ [ -1.796 ], [ -0.004 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.16827022026888297, negative=1, min=1.068, max=1.068, mean=0.333, count=4.0, positive=3, stdDev=0.9795034456294679, zeros=0},
    {meanExponent=-0.5156883310002227, negative=3, min=-0.004, max=-0.004, mean=-0.43500000000000005, count=4.0, positive=1, stdDev=1.104732999416601, zeros=0}
    Output: [
    	[ [ 0.12800000607967377, 1.128000020980835 ], [ -1.1799999475479126, -1.0679999589920044 ] ],
    	[ [ 1.315999984741211, -1.7960000038146973 ], [ 1.0679999589920044, -0.004000000189989805 ] ]
    ]
    Outputs Statistics: {meanExponent=-0.34197927656398763, negative=4, min=-0.004000000189989805, max=-0.004000000189989805, mean=-0.05099999246886, count=8.0, positive=4, stdDev=1.1123789669894582, zeros=0}
    Feedback for input 0
    Inputs Values: [
    	[ [ 0.128 ], [ -1.18 ] ],
    	[ [ 1.316 ], [ 1.068 ] ]
    ]
    Value Statistics: {meanExponent=-0.16827022026888297, negative=1, min=1.068, max=1.068, mean=0.333, count=4.0, positive=3, stdDev=0.9795034456294679, zeros=0}
    Implemented Feedback: [ [ 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0 ] ]
    Implemented Statistics: {meanExponent=0.0, negative=0, min=0.0, max=0.0, mean=0.125, count=32.0, positive=4, stdDev=0.33071891388307384, zeros=28}
    Measured Feedback: [ [ 0.9998679161071777, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 1.0001659393310547, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.9989738464355469, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 1.0001659393310547, 0.0, 0.0, 0.0, 0.0 ] ]
    Measured Statistics: {meanExponent=-8.97819053861039E-5, negative=0, min=0.0, max=0.0, mean=0.12497417628765106, count=32.0, positive=4, stdDev=0.3306506358812606, zeros=28}
    Feedback Error: [ [ -1.3208389282226562E-4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 1.659393310546875E-4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, -0.001026153564453125, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 1.659393310546875E-4, 0.0, 0.0, 0.0, 0.0 ] ]
    Error Statistics: {meanExponent=-3.6070097779556143, negative=2, min=0.0, max=0.0, mean=-2.5823712348937988E-5, count=32.0, positive=2, stdDev=1.8575600292632788E-4, zeros=28}
    Feedback for input 1
    Inputs Values: [
    	[ [ 1.128 ], [ -1.068 ] ],
    	[ [ -1.796 ], [ -0.004 ] ]
    ]
    Value Statistics: {meanExponent=-0.5156883310002227, negative=3, min=-0.004, max=-0.004, mean=-0.43500000000000005, count=4.0, positive=1, stdDev=1.104732999416601, zeros=0}
    Implemented Feedback: [ [ 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0 ] ]
    Implemented Statistics: {meanExponent=0.0, negative=0, min=1.0, max=1.0, mean=0.125, count=32.0, positive=4, stdDev=0.33071891388307384, zeros=28}
    Measured Feedback: [ [ 0.0, 0.0, 0.0, 0.0, 1.0001659393310547, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 1.0001659393310547, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0001659393310547, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.000002957880497 ] ]
    Measured Statistics: {meanExponent=5.4366565169123125E-5, negative=0, min=1.000002957880497, max=1.000002957880497, mean=0.1250156492460519, count=32.0, positive=4, stdDev=0.33076031883745116, zeros=28}
    Feedback Error: [ [ 0.0, 0.0, 0.0, 0.0, 1.659393310546875E-4, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 1.659393310546875E-4, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.659393310546875E-4, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.9578804969787598E-6 ] ]
    Error Statistics: {meanExponent=-4.217292842788862, negative=0, min=2.9578804969787598E-6, max=2.9578804969787598E-6, mean=1.564924605190754E-5, count=32.0, positive=4, stdDev=4.8341095665943656E-5, zeros=28}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 3.1108e-05 +- 1.3383e-04 [0.0000e+00 - 1.0262e-03] (64#)
    relativeTol: 1.2446e-04 +- 1.4932e-04 [1.4789e-06 - 5.1334e-04] (8#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=3.1108e-05 +- 1.3383e-04 [0.0000e+00 - 1.0262e-03] (64#), relativeTol=1.2446e-04 +- 1.4932e-04 [1.4789e-06 - 5.1334e-04] (8#)}
```



### Performance
Code from [LayerTestBase.java:149](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L149) executed in 0.09 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 3.5870 +- 0.5077 [3.0806 - 5.7908]
    Learning performance: 1.0366 +- 0.1739 [0.8920 - 1.9606]
    
```

