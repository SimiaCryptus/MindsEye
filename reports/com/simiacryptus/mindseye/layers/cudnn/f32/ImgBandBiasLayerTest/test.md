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
      "class": "com.simiacryptus.mindseye.layers.cudnn.f32.ImgBandBiasLayer",
      "id": "370a9587-74a1-4959-b406-fa45000003ad",
      "isFrozen": false,
      "name": "ImgBandBiasLayer/370a9587-74a1-4959-b406-fa45000003ad",
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
    	[ [ 1.148, -1.152 ], [ -0.064, -0.228 ], [ 1.272, -1.192 ] ],
    	[ [ -1.712, -1.612 ], [ -1.932, -0.056 ], [ -1.396, -0.2 ] ],
    	[ [ -0.964, -0.18 ], [ 1.924, 0.172 ], [ -0.872, 0.364 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 1.1480000019073486, -1.1519999504089355 ], [ -0.06400000303983688, -0.2280000001192093 ], [ 1.2719999551773071, -1.1920000314712524 ] ],
    	[ [ -1.7120000123977661, -1.6119999885559082 ], [ -1.9320000410079956, -0.0560000017285347 ], [ -1.3960000276565552, -0.20000000298023224 ] ],
    	[ [ -0.9639999866485596, -0.18000000715255737 ], [ 1.9240000247955322, 0.1720000058412552 ], [ -0.871999979019165, 0.36399999260902405 ] ]
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
Code from [LayerTestBase.java:144](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.05 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ 1.148, -1.152 ], [ -0.064, -0.228 ], [ 1.272, -1.192 ] ],
    	[ [ -1.712, -1.612 ], [ -1.932, -0.056 ], [ -1.396, -0.2 ] ],
    	[ [ -0.964, -0.18 ], [ 1.924, 0.172 ], [ -0.872, 0.364 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.2417803978947661, negative=13, min=0.364, max=0.364, mean=-0.37111111111111117, count=18.0, positive=5, stdDev=1.0567485188533579, zeros=0}
    Output: [
    	[ [ 1.1480000019073486, -1.1519999504089355 ], [ -0.06400000303983688, -0.2280000001192093 ], [ 1.2719999551773071, -1.1920000314712524 ] ],
    	[ [ -1.7120000123977661, -1.6119999885559082 ], [ -1.9320000410079956, -0.0560000017285347 ], [ -1.3960000276565552, -0.20000000298023224 ] ],
    	[ [ -0.9639999866485596, -0.18000000715255737 ], [ 1.9240000247955322, 0.1720000058412552 ], [ -0.871999979019165, 0.36399999260902405 ] ]
    ]
    Outputs Statistics: {meanExponent=-0.24178039516560132, negative=13, min=0.36399999260902405, max=0.36399999260902405, mean=-0.37111111399200225, count=18.0, positive=5, stdDev=1.0567485211657808, zeros=0}
    Feedback for input 0
    Inputs Values: [
    	[ [ 1.148, -1.152 ], [ -0.064, -0.228 ], [ 1.272, -1.192 ] ],
    	[ [ -1.712, -1.612 ], [ -1.932, -0.056 ], [ -1.396, -0.2 ] ],
    	[ [ -0.964, -0.18 ], [ 1.924, 0.172 ], [ -0.872, 0.364 ] ]
    ]
    Value Statistics: {meanExponent=-0.2417803978947661, negative=13, min=0.364, max=0.364, mean=-0.37111111111111117, count=18.0, positive=5, stdDev=1.0567485188533579, zeros=0}
    Implemented Feedback: [ [ 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, ... ], ... ]
    Implemented Statistics: {meanExponent=0.0, negative=0, min=1.0, max=1.0, mean=0.05555555555555555, count=324.0, positive=18, stdDev=0.2290614236454256, zeros=306}
    Measured Feedback: [ [ 1.0001659393
```
...[skipping 847 bytes](etc/1.txt)...
```
    0, 0.0, 0.0, ... ], [ 0.0, 0.0, -4.3010711669921875E-4, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 1.6927719116210938E-5, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 1.659393310546875E-4, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 1.659393310546875E-4, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.659393310546875E-4, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.659393310546875E-4, ... ], ... ]
    Error Statistics: {meanExponent=-3.971013793835973, negative=4, min=1.659393310546875E-4, max=1.659393310546875E-4, mean=-1.3591330728413146E-6, count=324.0, positive=14, stdDev=7.220253027067248E-5, zeros=306}
    Learning Gradient for weight set 0
    Weights: [ 0.0, 0.0 ]
    Implemented Gradient: [ [ 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ] ]
    Implemented Statistics: {meanExponent=0.0, negative=0, min=1.0, max=1.0, mean=0.5, count=36.0, positive=18, stdDev=0.5, zeros=18}
    Measured Gradient: [ [ 1.0001659393310547, 1.0001659393310547, 1.0001659393310547, 1.0000169277191162, 1.0001659393310547, 1.0001659393310547, 1.0001659393310547, 1.0001659393310547, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ] ]
    Measured Statistics: {meanExponent=4.330046392377212E-5, negative=0, min=0.9998679161071777, max=0.9998679161071777, mean=0.5000498559739854, count=36.0, positive=18, stdDev=0.5000498599484616, zeros=18}
    Gradient Error: [ [ 1.659393310546875E-4, 1.659393310546875E-4, 1.659393310546875E-4, 1.6927719116210938E-5, 1.659393310546875E-4, 1.659393310546875E-4, 1.659393310546875E-4, 1.659393310546875E-4, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ] ]
    Error Statistics: {meanExponent=-4.116006488343194, negative=1, min=-1.3208389282226562E-4, max=-1.3208389282226562E-4, mean=4.985597398546007E-5, count=36.0, positive=17, stdDev=8.037717772688225E-5, zeros=18}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 1.5710e-05 +- 7.3085e-05 [0.0000e+00 - 1.0262e-03] (360#)
    relativeTol: 7.8555e-05 +- 8.8360e-05 [8.4638e-06 - 5.1334e-04] (36#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=1.5710e-05 +- 7.3085e-05 [0.0000e+00 - 1.0262e-03] (360#), relativeTol=7.8555e-05 +- 8.8360e-05 [8.4638e-06 - 5.1334e-04] (36#)}
```



### Performance
Code from [LayerTestBase.java:149](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L149) executed in 0.11 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 3.6980 +- 0.9392 [3.1120 - 9.9572]
    Learning performance: 3.5894 +- 0.4967 [3.0521 - 7.1159]
    
```

