# ImgBandSelectLayer
## ImgBandSelectLayerTest
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
      "class": "com.simiacryptus.mindseye.layers.java.ImgBandSelectLayer",
      "id": "370a9587-74a1-4959-b406-fa4500002bdf",
      "isFrozen": false,
      "name": "ImgBandSelectLayer/370a9587-74a1-4959-b406-fa4500002bdf",
      "bands": [
        0,
        2
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
    	[ [ 0.448, 1.028, 2.0 ], [ 0.868, -1.852, -1.864 ] ],
    	[ [ 0.72, 0.44, -0.888 ], [ -1.704, 1.416, 1.168 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 0.448, 2.0 ], [ 0.868, -1.864 ] ],
    	[ [ 0.72, -0.888 ], [ -1.704, 1.168 ] ]
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
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (200#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (160#)}
```



### Differential Validation
Code from [LayerTestBase.java:144](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.00 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ 0.448, 1.028, 2.0 ], [ 0.868, -1.852, -1.864 ] ],
    	[ [ 0.72, 0.44, -0.888 ], [ -1.704, 1.416, 1.168 ] ]
    ]
    Inputs Statistics: {meanExponent=0.028340130426551757, negative=4, min=1.168, max=1.168, mean=0.14833333333333332, count=12.0, positive=8, stdDev=1.3046301732249217, zeros=0}
    Output: [
    	[ [ 0.448, 2.0 ], [ 0.868, -1.864 ] ],
    	[ [ 0.72, -0.888 ], [ -1.704, 1.168 ] ]
    ]
    Outputs Statistics: {meanExponent=0.03324144228418888, negative=3, min=1.168, max=1.168, mean=0.09349999999999997, count=8.0, positive=5, stdDev=1.319728665294499, zeros=0}
    Feedback for input 0
    Inputs Values: [
    	[ [ 0.448, 1.028, 2.0 ], [ 0.868, -1.852, -1.864 ] ],
    	[ [ 0.72, 0.44, -0.888 ], [ -1.704, 1.416, 1.168 ] ]
    ]
    Value Statistics: {meanExponent=0.028340130426551757, negative=4, min=1.168, max=1.168, mean=0.14833333333333332, count=12.0, positive=8, stdDev=1.3046301732249217, zeros=0}
    Implemented Feedback: [ [ 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], ... ]
    Implemented Statistics: {meanExponent=0.0, negative=0, min=1.0, max=1.0, mean=0.08333333333333333, count=96.0, positive=8, stdDev=0.2763853991962833, zeros=88}
    Measured Feedback: [ [ 0.9999999999998899, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.9999999999998899, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.9999999999998899, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.9999999999998899, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], ... ]
    Measured Statistics: {meanExponent=7.271029097799467E-14, negative=0, min=0.9999999999998899, max=0.9999999999998899, mean=0.08333333333334729, count=96.0, positive=8, stdDev=0.2763853991963296, zeros=88}
    Feedback Error: [ [ -1.1013412404281553E-13, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, -1.1013412404281553E-13, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, -1.1013412404281553E-13, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, -1.1013412404281553E-13, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], ... ]
    Error Statistics: {meanExponent=-12.797775004143459, negative=7, min=-1.1013412404281553E-13, max=-1.1013412404281553E-13, mean=1.3951802676122801E-14, count=96.0, positive=1, stdDev=2.16978215833027E-13, zeros=88}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 3.0013e-14 +- 2.1534e-13 [0.0000e+00 - 2.1103e-12] (96#)
    relativeTol: 1.8008e-13 +- 3.3075e-13 [5.5067e-14 - 1.0552e-12] (8#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=3.0013e-14 +- 2.1534e-13 [0.0000e+00 - 2.1103e-12] (96#), relativeTol=1.8008e-13 +- 3.3075e-13 [5.5067e-14 - 1.0552e-12] (8#)}
```



### Performance
Code from [LayerTestBase.java:149](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L149) executed in 0.00 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 0.2205 +- 0.0540 [0.1339 - 0.4275]
    Learning performance: 0.0033 +- 0.0023 [0.0000 - 0.0142]
    
```

