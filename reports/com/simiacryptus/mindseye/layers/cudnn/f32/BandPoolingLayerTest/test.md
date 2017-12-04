# BandPoolingLayer
## BandPoolingLayerTest
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
      "class": "com.simiacryptus.mindseye.layers.cudnn.f32.BandPoolingLayer",
      "id": "370a9587-74a1-4959-b406-fa4500000033",
      "isFrozen": false,
      "name": "BandPoolingLayer/370a9587-74a1-4959-b406-fa4500000033",
      "mode": 0
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
    	[ [ -0.812, 0.644 ], [ -0.076, -1.44 ], [ 0.388, 1.032 ] ],
    	[ [ -1.028, 0.936 ], [ -0.984, 1.348 ], [ -1.704, -0.468 ] ],
    	[ [ -1.26, -0.408 ], [ 1.48, -0.46 ], [ 1.8, 0.78 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 1.7999999523162842, 1.3480000495910645 ] ]
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
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (200#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (40#)}
```



### Differential Validation
Code from [LayerTestBase.java:144](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.02 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ -0.812, 0.644 ], [ -0.076, -1.44 ], [ 0.388, 1.032 ] ],
    	[ [ -1.028, 0.936 ], [ -0.984, 1.348 ], [ -1.704, -0.468 ] ],
    	[ [ -1.26, -0.408 ], [ 1.48, -0.46 ], [ 1.8, 0.78 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.10782132419594452, negative=10, min=0.78, max=0.78, mean=-0.012888888888888901, count=18.0, positive=8, stdDev=1.059648206240002, zeros=0}
    Output: [
    	[ [ 1.7999999523162842, 1.3480000495910645 ] ]
    ]
    Outputs Statistics: {meanExponent=0.192481200887414, negative=0, min=1.3480000495910645, max=1.3480000495910645, mean=1.5740000009536743, count=2.0, positive=2, stdDev=0.22599995136260986, zeros=0}
    Feedback for input 0
    Inputs Values: [
    	[ [ -0.812, 0.644 ], [ -0.076, -1.44 ], [ 0.388, 1.032 ] ],
    	[ [ -1.028, 0.936 ], [ -0.984, 1.348 ], [ -1.704, -0.468 ] ],
    	[ [ -1.26, -0.408 ], [ 1.48, -0.46 ], [ 1.8, 0.78 ] ]
    ]
    Value Statistics: {meanExponent=-0.10782132419594452, negative=10, min=0.78, max=0.78, mean=-0.012888888888888901, count=18.0, positive=8, stdDev=1.059648206240002, zeros=0}
    Implemented Feedback: [ [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], ... ]
    Implemented Statistics: {meanExponent=0.0, negative=0, min=0.0, max=0.0, mean=0.05555555555555555, count=36.0, positive=2, stdDev=0.2290614236454256, zeros=34}
    Measured Feedback: [ [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], ... ]
    Measured Statistics: {meanExponent=-1.8691054207909988E-4, negative=0, min=0.0, max=0.0, mean=0.05553166071573893, count=36.0, positive=2, stdDev=0.2289629457984703, zeros=34}
    Feedback Error: [ [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], ... ]
    Error Statistics: {meanExponent=-3.384419153473139, negative=1, min=0.0, max=0.0, mean=-2.3894839816623265E-5, count=36.0, positive=1, stdDev=1.7159159508324927E-4, zeros=34}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 3.3114e-05 +- 1.7005e-04 [0.0000e+00 - 1.0262e-03] (36#)
    relativeTol: 2.9815e-04 +- 2.1519e-04 [8.2963e-05 - 5.1334e-04] (2#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=3.3114e-05 +- 1.7005e-04 [0.0000e+00 - 1.0262e-03] (36#), relativeTol=2.9815e-04 +- 2.1519e-04 [8.2963e-05 - 5.1334e-04] (2#)}
```



### Performance
Code from [LayerTestBase.java:149](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L149) executed in 0.08 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 3.0090 +- 0.4826 [2.5876 - 6.0587]
    Learning performance: 1.3408 +- 0.2307 [1.0943 - 2.4936]
    
```

