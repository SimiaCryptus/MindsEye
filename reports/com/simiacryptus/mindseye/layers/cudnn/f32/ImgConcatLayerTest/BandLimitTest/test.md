# ImgConcatLayer
## BandLimitTest
### Json Serialization
Code from [LayerTestBase.java:83](../../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L83) executed in 0.00 seconds: 
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
      "id": "370a9587-74a1-4959-b406-fa45000003d1",
      "isFrozen": false,
      "name": "ImgConcatLayer/370a9587-74a1-4959-b406-fa45000003d1",
      "maxBands": 3
    }
```



### Example Input/Output Pair
Code from [LayerTestBase.java:120](../../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L120) executed in 0.00 seconds: 
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
    	[ [ -1.516, -0.96 ] ]
    ],
    [
    	[ [ 1.344, -0.928 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ -1.5160000324249268, -0.9599999785423279, 1.343999981880188 ] ]
    ]
```



### Batch Execution
Code from [LayerTestBase.java:138](../../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L138) executed in 0.01 seconds: 
```java
    BatchingTester batchingTester = getBatchingTester();
    return batchingTester==null?null:batchingTester.test(layer, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (70#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (60#)}
```



### Differential Validation
Code from [LayerTestBase.java:144](../../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.01 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ -1.516, -0.96 ] ]
    ],
    [
    	[ [ 1.344, -0.928 ] ]
    ]
    Inputs Statistics: {meanExponent=0.08148521716780156, negative=2, min=-0.96, max=-0.96, mean=-1.238, count=2.0, positive=0, stdDev=0.27800000000000025, zeros=0},
    {meanExponent=0.047973622468334275, negative=1, min=-0.928, max=-0.928, mean=0.20800000000000002, count=2.0, positive=1, stdDev=1.1360000000000001, zeros=0}
    Output: [
    	[ [ -1.5160000324249268, -0.9599999785423279, 1.343999981880188 ] ]
    ]
    Outputs Statistics: {meanExponent=0.09712323225996933, negative=2, min=1.343999981880188, max=1.343999981880188, mean=-0.3773333430290222, count=3.0, positive=1, stdDev=1.2381505903461045, zeros=0}
    Feedback for input 0
    Inputs Values: [
    	[ [ -1.516, -0.96 ] ]
    ]
    Value Statistics: {meanExponent=0.08148521716780156, negative=2, min=-0.96, max=-0.96, mean=-1.238, count=2.0, positive=0, stdDev=0.27800000000000025, zeros=0}
    Implemented Feedback: [ [ 1.0, 0.0, 0.0 ], [ 0.0, 1.0, 0.0 ] ]
    Implemented Statistics: {meanExponent=0.0, negative=0, min=0.0, max=0.0, mean=0.3333333333333333, count=6.0, positive=2, stdDev=0.4714045207910317, zeros=4}
    Measured Feedback: [ [ 1.0001659393310547, 0.0, 0.0 ], [ 0.0, 0.9995698928833008, 0.0 ] ]
    Measured Statistics: {meanExponent=-5.738638616508357E-5, negative=0, min=0.0, max=0.0, mean=0.33328930536905926, count=6.0, positive=2, stdDev=0.47134228725282795, zeros=4}
    Feedback Error: [ [ 1.659393310546875E-4, 0.0, 0.0 ], [ 0.0, -4.3010711669921875E-4, 0.0 ] ]
    Error Statistics: {meanExponent=-3.573237018199344, negative=1, min=0.0, max=0.0, mean=-4.402796427408854E-5, count=6.0, positive=1, stdDev=1.8298325223795302E-4, zeros=4}
    Feedback for input 1
    Inputs Values: [
    	[ [ 1.344, -0.928 ] ]
    ]
    Value Statistics: {meanExponent=0.047973622468334275, negative=1, min=-0.928, max=-0.928, mean=0.20800000000000002, count=2.0, positive=1, stdDev=1.1360000000000001, zeros=0}
    Implemented Feedback: [ [ 0.0, 0.0, 1.0 ], [ 0.0, 0.0, 0.0 ] ]
    Implemented Statistics: {meanExponent=0.0, negative=0, min=0.0, max=0.0, mean=0.16666666666666666, count=6.0, positive=1, stdDev=0.37267799624996495, zeros=5}
    Measured Feedback: [ [ 0.0, 0.0, 1.0001659393310547 ], [ 0.0, 0.0, 0.0 ] ]
    Measured Statistics: {meanExponent=7.206055713278508E-5, negative=0, min=0.0, max=0.0, mean=0.16669432322184244, count=6.0, positive=1, stdDev=0.37273983818736145, zeros=5}
    Feedback Error: [ [ 0.0, 0.0, 1.659393310546875E-4 ], [ 0.0, 0.0, 0.0 ] ]
    Error Statistics: {meanExponent=-3.7800506649970242, negative=0, min=0.0, max=0.0, mean=2.765655517578125E-5, count=6.0, positive=1, stdDev=6.184193739652052E-5, zeros=5}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 6.3499e-05 +- 1.2639e-04 [0.0000e+00 - 4.3011e-04] (12#)
    relativeTol: 1.2701e-04 +- 6.2290e-05 [8.2963e-05 - 2.1510e-04] (3#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=6.3499e-05 +- 1.2639e-04 [0.0000e+00 - 4.3011e-04] (12#), relativeTol=1.2701e-04 +- 6.2290e-05 [8.2963e-05 - 2.1510e-04] (3#)}
```



### Performance
Code from [LayerTestBase.java:149](../../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L149) executed in 0.09 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 3.4425 +- 0.2212 [3.2032 - 4.8532]
    Learning performance: 1.0629 +- 0.3324 [0.8863 - 3.2659]
    
```

