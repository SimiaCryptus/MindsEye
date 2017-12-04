# ImgBandSelectLayer
## ImgBandSelectLayerTest
### Json Serialization
Code from [LayerTestBase.java:83](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L83) executed in 0.00 seconds: 
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
      "id": "a864e734-2f23-44db-97c1-504000002bdf",
      "isFrozen": false,
      "name": "ImgBandSelectLayer/a864e734-2f23-44db-97c1-504000002bdf",
      "bands": [
        0,
        2
      ]
    }
```



### Example Input/Output Pair
Code from [LayerTestBase.java:120](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L120) executed in 0.00 seconds: 
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
    	[ [ 0.06, 1.648, -1.176 ], [ 0.076, -1.772, -0.744 ] ],
    	[ [ -1.452, -1.004, 1.564 ], [ -0.864, 1.788, 1.78 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 0.06, -1.176 ], [ 0.076, -0.744 ] ],
    	[ [ -1.452, 1.564 ], [ -0.864, 1.78 ] ]
    ]
```



### Batch Execution
Code from [LayerTestBase.java:138](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L138) executed in 0.00 seconds: 
```java
    BatchingTester batchingTester = getBatchingTester();
    return batchingTester==null?null:batchingTester.test(layer, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (200#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (160#)}
```



### Differential Validation
Code from [LayerTestBase.java:144](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.00 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ 0.06, 1.648, -1.176 ], [ 0.076, -1.772, -0.744 ] ],
    	[ [ -1.452, -1.004, 1.564 ], [ -0.864, 1.788, 1.78 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.09469963649940388, negative=6, min=1.78, max=1.78, mean=-0.00799999999999997, count=12.0, positive=6, stdDev=1.306243468883194, zeros=0}
    Output: [
    	[ [ 0.06, -1.176 ], [ 0.076, -0.744 ] ],
    	[ [ -1.452, 1.564 ], [ -0.864, 1.78 ] ]
    ]
    Outputs Statistics: {meanExponent=-0.23198972377173438, negative=4, min=1.78, max=1.78, mean=-0.09449999999999995, count=8.0, positive=4, stdDev=1.1370873976964129, zeros=0}
    Feedback for input 0
    Inputs Values: [
    	[ [ 0.06, 1.648, -1.176 ], [ 0.076, -1.772, -0.744 ] ],
    	[ [ -1.452, -1.004, 1.564 ], [ -0.864, 1.788, 1.78 ] ]
    ]
    Value Statistics: {meanExponent=-0.09469963649940388, negative=6, min=1.78, max=1.78, mean=-0.00799999999999997, count=12.0, positive=6, stdDev=1.306243468883194, zeros=0}
    Implemented Feedback: [ [ 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], ... ]
    Implemented Statistics: {meanExponent=0.0, negative=0, min=1.0, max=1.0, mean=0.08333333333333333, count=96.0, positive=8, stdDev=0.2763853991962833, zeros=88}
    Measured Feedback: [ [ 1.0000000000000286, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.9999999999998899, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 1.0000000000000286, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.9999999999998899, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], ... ]
    Measured Statistics: {meanExponent=-3.276302567614995E-14, negative=0, min=0.9999999999998899, max=0.9999999999998899, mean=0.08333333333332704, count=96.0, positive=8, stdDev=0.2763853991962625, zeros=88}
    Feedback Error: [ [ 2.864375403532904E-14, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, -1.1013412404281553E-13, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 2.864375403532904E-14, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, -1.1013412404281553E-13, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], ... ]
    Error Statistics: {meanExponent=-13.10430108958456, negative=6, min=-1.1013412404281553E-13, max=-1.1013412404281553E-13, mean=-6.286637876939949E-15, count=96.0, positive=2, stdDev=2.7123173257641554E-14, zeros=88}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 7.4801e-15 +- 2.6819e-14 [0.0000e+00 - 1.1013e-13] (96#)
    relativeTol: 4.4881e-14 +- 1.7643e-14 [1.4322e-14 - 5.5067e-14] (8#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=7.4801e-15 +- 2.6819e-14 [0.0000e+00 - 1.1013e-13] (96#), relativeTol=4.4881e-14 +- 1.7643e-14 [1.4322e-14 - 5.5067e-14] (8#)}
```



### Performance
Code from [LayerTestBase.java:149](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L149) executed in 0.00 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 0.1997 +- 0.0621 [0.1339 - 0.5215]
    Learning performance: 0.0025 +- 0.0024 [0.0000 - 0.0171]
    
```

