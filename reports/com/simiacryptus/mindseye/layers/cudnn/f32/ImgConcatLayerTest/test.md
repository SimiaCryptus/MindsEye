# ImgConcatLayer
## ImgConcatLayerTest
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
      "class": "com.simiacryptus.mindseye.layers.cudnn.f32.ImgConcatLayer",
      "id": "a864e734-2f23-44db-97c1-5040000003c0",
      "isFrozen": false,
      "name": "ImgConcatLayer/a864e734-2f23-44db-97c1-5040000003c0",
      "maxBands": -1
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
    	[ [ 0.544 ], [ 0.836 ] ],
    	[ [ 1.72 ], [ 0.764 ] ]
    ],
    [
    	[ [ -1.06 ], [ 0.508 ] ],
    	[ [ 1.052 ], [ -0.076 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 0.5440000295639038, -1.059999942779541 ], [ 0.8360000252723694, 0.5080000162124634 ] ],
    	[ [ 1.7200000286102295, 1.0520000457763672 ], [ 0.7639999985694885, -0.07599999755620956 ] ]
    ]
```



### Batch Execution
Code from [LayerTestBase.java:138](../../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L138) executed in 0.02 seconds: 
```java
    BatchingTester batchingTester = getBatchingTester();
    return batchingTester==null?null:batchingTester.test(layer, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (160#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (160#)}
```



### Differential Validation
Code from [LayerTestBase.java:144](../../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.03 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ 0.544 ], [ 0.836 ] ],
    	[ [ 1.72 ], [ 0.764 ] ]
    ],
    [
    	[ [ -1.06 ], [ 0.508 ] ],
    	[ [ 1.052 ], [ -0.076 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.055893254344891216, negative=0, min=0.764, max=0.764, mean=0.966, count=4.0, positive=4, stdDev=0.44841498636865373, zeros=0},
    {meanExponent=-0.3415002725881997, negative=2, min=-0.076, max=-0.076, mean=0.106, count=4.0, positive=2, stdDev=0.7824960063795854, zeros=0}
    Output: [
    	[ [ 0.5440000295639038, -1.059999942779541 ], [ 0.8360000252723694, 0.5080000162124634 ] ],
    	[ [ 1.7200000286102295, 1.0520000457763672 ], [ 0.7639999985694885, -0.07599999755620956 ] ]
    ]
    Outputs Statistics: {meanExponent=-0.19869675865520195, negative=2, min=-0.07599999755620956, max=-0.07599999755620956, mean=0.5360000254586339, count=8.0, positive=6, stdDev=0.7691475755200986, zeros=0}
    Feedback for input 0
    Inputs Values: [
    	[ [ 0.544 ], [ 0.836 ] ],
    	[ [ 1.72 ], [ 0.764 ] ]
    ]
    Value Statistics: {meanExponent=-0.055893254344891216, negative=0, min=0.764, max=0.764, mean=0.966, count=4.0, positive=4, stdDev=0.44841498636865373, zeros=0}
    Implemented Feedback: [ [ 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0 ] ]
    Implemented Statistics: {meanExponent=0.0, negative=0, min=0.0, max=0.0, mean=0.125, count=32.0, positive=4, stdDev=0.33071891388307384, zeros=28}
    Measured Feedback: [ [ 0.9995698928833008, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 1.0001659393310547, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.9995698928833008, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 1.0001659393310547, 0.0, 0.0, 0.0, 0.0 ] ]
    Measured Statistics: {meanExponent=-5.738638616508356E-5, negative=0, min=0.0, max=0.0, mean=0.12498348951339722, count=32.0, positive=4, stdDev=0.3306752480287095, zeros=28}
    Feedback Error: [ [ -4.3010711669921875E-4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 1.659393310546875E-4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, -4.3010711669921875E-4, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 1.659393310546875E-4, 0.0, 0.0, 0.0, 0.0 ] ]
    Error Statistics: {meanExponent=-3.573237018199344, negative=2, min=0.0, max=0.0, mean=-1.6510486602783203E-5, count=32.0, positive=2, stdDev=1.140631554064517E-4, zeros=28}
    Feedback for input 1
    Inputs Values: [
    	[ [ -1.06 ], [ 0.508 ] ],
    	[ [ 1.052 ], [ -0.076 ] ]
    ]
    Value Statistics: {meanExponent=-0.3415002725881997, negative=2, min=-0.076, max=-0.076, mean=0.106, count=4.0, positive=2, stdDev=0.7824960063795854, zeros=0}
    Implemented Feedback: [ [ 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0 ] ]
    Implemented Statistics: {meanExponent=0.0, negative=0, min=1.0, max=1.0, mean=0.125, count=32.0, positive=4, stdDev=0.33071891388307384, zeros=28}
    Measured Feedback: [ [ 0.0, 0.0, 0.0, 0.0, 0.9989738464355469, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.9989738464355469, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9995698928833008, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.999942421913147 ] ]
    Measured Statistics: {meanExponent=-2.759007943414354E-4, negative=0, min=0.999942421913147, max=0.999942421913147, mean=0.12492062523961067, count=32.0, positive=4, stdDev=0.33050894022025706, zeros=28}
    Feedback Error: [ [ 0.0, 0.0, 0.0, 0.0, -0.001026153564453125, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, -0.001026153564453125, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -4.3010711669921875E-4, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -5.7578086853027344E-5 ] ]
    Error Statistics: {meanExponent=-3.3959353562050563, negative=4, min=-5.7578086853027344E-5, max=-5.7578086853027344E-5, mean=-7.9374760389328E-5, count=32.0, positive=0, stdDev=2.557268049611889E-4, zeros=28}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 5.8314e-05 +- 1.9771e-04 [0.0000e+00 - 1.0262e-03] (64#)
    relativeTol: 2.3334e-04 +- 1.7495e-04 [2.8790e-05 - 5.1334e-04] (8#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=5.8314e-05 +- 1.9771e-04 [0.0000e+00 - 1.0262e-03] (64#), relativeTol=2.3334e-04 +- 1.7495e-04 [2.8790e-05 - 5.1334e-04] (8#)}
```



### Performance
Code from [LayerTestBase.java:149](../../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L149) executed in 0.09 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 3.7352 +- 0.8807 [3.1661 - 11.2054]
    Learning performance: 1.1448 +- 0.5680 [0.8977 - 5.7053]
    
```

