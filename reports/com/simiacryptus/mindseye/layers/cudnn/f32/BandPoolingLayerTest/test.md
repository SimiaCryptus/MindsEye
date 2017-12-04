# BandPoolingLayer
## BandPoolingLayerTest
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
      "class": "com.simiacryptus.mindseye.layers.cudnn.f32.BandPoolingLayer",
      "id": "a864e734-2f23-44db-97c1-504000000033",
      "isFrozen": false,
      "name": "BandPoolingLayer/a864e734-2f23-44db-97c1-504000000033",
      "mode": 0
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
    	[ [ -1.872, 0.128 ], [ 1.94, -1.288 ], [ -0.196, 1.304 ] ],
    	[ [ -0.24, 1.02 ], [ -1.58, 0.312 ], [ 0.424, 1.66 ] ],
    	[ [ 0.768, 1.116 ], [ -0.536, -1.988 ], [ -0.44, 1.3 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 1.940000057220459, 1.659999966621399 ] ]
    ]
```



### Batch Execution
Code from [LayerTestBase.java:138](../../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L138) executed in 0.01 seconds: 
```java
    BatchingTester batchingTester = getBatchingTester();
    return batchingTester==null?null:batchingTester.test(layer, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (200#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (40#)}
```



### Differential Validation
Code from [LayerTestBase.java:144](../../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.01 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ -1.872, 0.128 ], [ 1.94, -1.288 ], [ -0.196, 1.304 ] ],
    	[ [ -0.24, 1.02 ], [ -1.58, 0.312 ], [ 0.424, 1.66 ] ],
    	[ [ 0.768, 1.116 ], [ -0.536, -1.988 ], [ -0.44, 1.3 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.12045184877690615, negative=8, min=1.3, max=1.3, mean=0.10177777777777779, count=18.0, positive=10, stdDev=1.1804219375364404, zeros=0}
    Output: [
    	[ [ 1.940000057220459, 1.659999966621399 ] ]
    ]
    Outputs Statistics: {meanExponent=0.25395491102360823, negative=0, min=1.659999966621399, max=1.659999966621399, mean=1.800000011920929, count=2.0, positive=2, stdDev=0.14000004529953003, zeros=0}
    Feedback for input 0
    Inputs Values: [
    	[ [ -1.872, 0.128 ], [ 1.94, -1.288 ], [ -0.196, 1.304 ] ],
    	[ [ -0.24, 1.02 ], [ -1.58, 0.312 ], [ 0.424, 1.66 ] ],
    	[ [ 0.768, 1.116 ], [ -0.536, -1.988 ], [ -0.44, 1.3 ] ]
    ]
    Value Statistics: {meanExponent=-0.12045184877690615, negative=8, min=1.3, max=1.3, mean=0.10177777777777779, count=18.0, positive=10, stdDev=1.1804219375364404, zeros=0}
    Implemented Feedback: [ [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 1.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], ... ]
    Implemented Statistics: {meanExponent=0.0, negative=0, min=0.0, max=0.0, mean=0.05555555555555555, count=36.0, positive=2, stdDev=0.2290614236454256, zeros=34}
    Measured Feedback: [ [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.9989738464355469, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], ... ]
    Measured Statistics: {meanExponent=-1.8691054207909988E-4, negative=0, min=0.0, max=0.0, mean=0.05553166071573893, count=36.0, positive=2, stdDev=0.2289629457984703, zeros=34}
    Feedback Error: [ [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ -0.001026153564453125, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], ... ]
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
Code from [LayerTestBase.java:149](../../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L149) executed in 0.08 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 2.9534 +- 0.5211 [2.6418 - 6.4320]
    Learning performance: 1.5426 +- 0.8575 [1.1057 - 7.4294]
    
```

