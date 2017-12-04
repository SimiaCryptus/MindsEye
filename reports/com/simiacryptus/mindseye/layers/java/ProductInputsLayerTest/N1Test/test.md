# ProductInputsLayer
## N1Test
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
      "class": "com.simiacryptus.mindseye.layers.java.ProductInputsLayer",
      "id": "370a9587-74a1-4959-b406-fa4500002c6e",
      "isFrozen": false,
      "name": "ProductInputsLayer/370a9587-74a1-4959-b406-fa4500002c6e"
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
    [[ -1.592, 1.072, -0.06 ],
    [ -1.792 ]]
    --------------------
    Output: 
    [ 2.8528640000000003, -1.921024, 0.10752 ]
```



### Batch Execution
Code from [LayerTestBase.java:138](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L138) executed in 0.00 seconds: 
```java
    BatchingTester batchingTester = getBatchingTester();
    return batchingTester==null?null:batchingTester.test(layer, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (70#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (70#)}
```



### Differential Validation
Code from [LayerTestBase.java:144](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.00 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [ -1.592, 1.072, -0.06 ],
    [ -1.792 ]
    Inputs Statistics: {meanExponent=-0.3299036336193183, negative=2, min=-0.06, max=-0.06, mean=-0.19333333333333336, count=3.0, positive=1, stdDev=1.0916523663185498, zeros=0},
    {meanExponent=0.2533380053261064, negative=1, min=-1.792, max=-1.792, mean=-1.792, count=1.0, positive=0, stdDev=0.0, zeros=0}
    Output: [ 2.8528640000000003, -1.921024, 0.10752 ]
    Outputs Statistics: {meanExponent=-0.07656562829321185, negative=1, min=0.10752, max=0.10752, mean=0.34645333333333345, count=3.0, positive=2, stdDev=1.9562410404428412, zeros=0}
    Feedback for input 0
    Inputs Values: [ -1.592, 1.072, -0.06 ]
    Value Statistics: {meanExponent=-0.3299036336193183, negative=2, min=-0.06, max=-0.06, mean=-0.19333333333333336, count=3.0, positive=1, stdDev=1.0916523663185498, zeros=0}
    Implemented Feedback: [ [ -1.792, 0.0, 0.0 ], [ 0.0, -1.792, 0.0 ], [ 0.0, 0.0, -1.792 ] ]
    Implemented Statistics: {meanExponent=0.2533380053261064, negative=3, min=-1.792, max=-1.792, mean=-0.5973333333333334, count=9.0, positive=0, stdDev=0.8447569012575288, zeros=6}
    Measured Feedback: [ [ -1.7920000000026803, 0.0, 0.0 ], [ 0.0, -1.79200000000046, 0.0 ], [ 0.0, 0.0, -1.7920000000001823 ] ]
    Measured Statistics: {meanExponent=0.2533380053263748, negative=3, min=-1.7920000000001823, max=-1.7920000000001823, mean=-0.5973333333337025, count=9.0, positive=0, stdDev=0.8447569012580508, zeros=6}
    Feedback Error: [ [ -2.680300426050053E-12, 0.0, 0.0 ], [ 0.0, -4.598543767997398E-13, 0.0 ], [ 0.0, 0.0, -1.822986206434507E-13 ] ]
    Error Statistics: {meanExponent=-12.216137605881817, negative=3, min=-1.822986206434507E-13, max=-1.822986206434507E-13, mean=-3.691614914992493E-13, count=9.0, positive=0, stdDev=8.301397036095741E-13, zeros=6}
    Feedback for input 1
    Inputs Values: [ -1.792 ]
    Value Statistics: {meanExponent=0.2533380053261064, negative=1, min=-1.792, max=-1.792, mean=-1.792, count=1.0, positive=0, stdDev=0.0, zeros=0}
    Implemented Feedback: [ [ -1.592, 1.072, -0.06 ] ]
    Implemented Statistics: {meanExponent=-0.3299036336193183, negative=2, min=-0.06, max=-0.06, mean=-0.19333333333333336, count=3.0, positive=1, stdDev=1.0916523663185498, zeros=0}
    Measured Feedback: [ [ -1.59200000000137, 1.0719999999997398, -0.060000000000060005 ] ]
    Measured Statistics: {meanExponent=-0.3299036336190841, negative=2, min=-0.060000000000060005, max=-0.060000000000060005, mean=-0.19333333333389677, count=3.0, positive=1, stdDev=1.0916523663190318, zeros=0}
    Feedback Error: [ [ -1.3700152123874432E-12, -2.602362769721367E-13, -6.000755448098971E-14 ] ]
    Error Statistics: {meanExponent=-12.556566948413469, negative=3, min=-6.000755448098971E-14, max=-6.000755448098971E-14, mean=-5.634196812801898E-13, count=3.0, positive=0, stdDev=5.761771419909293E-13, zeros=0}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 4.1773e-13 +- 7.7905e-13 [0.0000e+00 - 2.6803e-12] (12#)
    relativeTol: 3.2979e-13 +- 2.5023e-13 [5.0865e-14 - 7.4785e-13] (6#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=4.1773e-13 +- 7.7905e-13 [0.0000e+00 - 2.6803e-12] (12#), relativeTol=3.2979e-13 +- 2.5023e-13 [5.0865e-14 - 7.4785e-13] (6#)}
```



### Performance
Code from [LayerTestBase.java:149](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L149) executed in 0.01 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 0.2374 +- 0.0446 [0.1795 - 0.4816]
    Learning performance: 0.0188 +- 0.0113 [0.0114 - 0.1168]
    
```

