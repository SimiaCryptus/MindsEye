# PoolingLayer
## PoolingLayerTest
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
      "class": "com.simiacryptus.mindseye.layers.cudnn.f32.PoolingLayer",
      "id": "370a9587-74a1-4959-b406-fa45000003d8",
      "isFrozen": false,
      "name": "PoolingLayer/370a9587-74a1-4959-b406-fa45000003d8",
      "mode": 0,
      "windowX": 2,
      "windowY": 2,
      "paddingX": 0,
      "paddingY": 0,
      "strideX": 2,
      "strideY": 2
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
    	[ [ -1.396, 0.628 ], [ 0.416, -1.636 ] ],
    	[ [ 1.116, 0.28 ], [ 1.656, -0.396 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 1.656000018119812, 0.628000020980835 ] ]
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
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (100#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (40#)}
```



### Differential Validation
Code from [LayerTestBase.java:144](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.01 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ -1.396, 0.628 ], [ 0.416, -1.636 ] ],
    	[ [ 1.116, 0.28 ], [ 1.656, -0.396 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.11408757046193274, negative=3, min=-0.396, max=-0.396, mean=0.08350000000000003, count=8.0, positive=5, stdDev=1.0816643425758288, zeros=0}
    Output: [
    	[ [ 1.656000018119812, 0.628000020980835 ] ]
    ]
    Outputs Statistics: {meanExponent=0.008509997723701632, negative=0, min=0.628000020980835, max=0.628000020980835, mean=1.1420000195503235, count=2.0, positive=2, stdDev=0.5139999985694885, zeros=0}
    Feedback for input 0
    Inputs Values: [
    	[ [ -1.396, 0.628 ], [ 0.416, -1.636 ] ],
    	[ [ 1.116, 0.28 ], [ 1.656, -0.396 ] ]
    ]
    Value Statistics: {meanExponent=-0.11408757046193274, negative=3, min=-0.396, max=-0.396, mean=0.08350000000000003, count=8.0, positive=5, stdDev=1.0816643425758288, zeros=0}
    Implemented Feedback: [ [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 1.0, 0.0 ], [ 0.0, 1.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ] ]
    Implemented Statistics: {meanExponent=0.0, negative=0, min=0.0, max=0.0, mean=0.125, count=16.0, positive=2, stdDev=0.33071891388307384, zeros=14}
    Measured Feedback: [ [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 1.0001659393310547, 0.0 ], [ 0.0, 0.9995698928833008 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ] ]
    Measured Statistics: {meanExponent=-5.738638616508357E-5, negative=0, min=0.0, max=0.0, mean=0.12498348951339722, count=16.0, positive=2, stdDev=0.3306752480287095, zeros=14}
    Feedback Error: [ [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 1.659393310546875E-4, 0.0 ], [ 0.0, -4.3010711669921875E-4 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ] ]
    Error Statistics: {meanExponent=-3.573237018199344, negative=1, min=0.0, max=0.0, mean=-1.6510486602783203E-5, count=16.0, positive=1, stdDev=1.140631554064517E-4, zeros=14}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 3.7253e-05 +- 1.0907e-04 [0.0000e+00 - 4.3011e-04] (16#)
    relativeTol: 1.4903e-04 +- 6.6069e-05 [8.2963e-05 - 2.1510e-04] (2#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=3.7253e-05 +- 1.0907e-04 [0.0000e+00 - 4.3011e-04] (16#), relativeTol=1.4903e-04 +- 6.6069e-05 [8.2963e-05 - 2.1510e-04] (2#)}
```



### Performance
Code from [LayerTestBase.java:149](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L149) executed in 0.12 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 5.4384 +- 25.7211 [2.4964 - 261.3028]
    Learning performance: 1.9720 +- 0.6145 [1.0915 - 4.6936]
    
```

