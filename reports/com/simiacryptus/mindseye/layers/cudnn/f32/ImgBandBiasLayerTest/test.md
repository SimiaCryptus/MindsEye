# ImgBandBiasLayer
## ImgBandBiasLayerTest
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
      "class": "com.simiacryptus.mindseye.layers.cudnn.f32.ImgBandBiasLayer",
      "id": "a864e734-2f23-44db-97c1-5040000003ad",
      "isFrozen": false,
      "name": "ImgBandBiasLayer/a864e734-2f23-44db-97c1-5040000003ad",
      "bias": [
        0.0,
        0.0
      ]
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
    	[ [ -1.884, 0.7 ], [ -0.764, -0.432 ], [ 1.544, 1.116 ] ],
    	[ [ 1.24, 0.716 ], [ 1.532, 0.188 ], [ -0.224, 0.772 ] ],
    	[ [ 0.036, 1.232 ], [ 0.272, -1.1 ], [ -1.2, -1.896 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ -1.8839999437332153, 0.699999988079071 ], [ -0.7639999985694885, -0.4320000112056732 ], [ 1.5440000295639038, 1.1160000562667847 ] ],
    	[ [ 1.2400000095367432, 0.7160000205039978 ], [ 1.531999945640564, 0.18799999356269836 ], [ -0.2240000069141388, 0.7720000147819519 ] ],
    	[ [ 0.035999998450279236, 1.2319999933242798 ], [ 0.2720000147819519, -1.100000023841858 ], [ -1.2000000476837158, -1.8960000276565552 ] ]
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
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (360#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (360#)}
```



### Differential Validation
Code from [LayerTestBase.java:144](../../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.07 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ -1.884, 0.7 ], [ -0.764, -0.432 ], [ 1.544, 1.116 ] ],
    	[ [ 1.24, 0.716 ], [ 1.532, 0.188 ], [ -0.224, 0.772 ] ],
    	[ [ 0.036, 1.232 ], [ 0.272, -1.1 ], [ -1.2, -1.896 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.16663452488883118, negative=7, min=-1.896, max=-1.896, mean=0.10266666666666671, count=18.0, positive=11, stdDev=1.0843877945130558, zeros=0}
    Output: [
    	[ [ -1.8839999437332153, 0.699999988079071 ], [ -0.7639999985694885, -0.4320000112056732 ], [ 1.5440000295639038, 1.1160000562667847 ] ],
    	[ [ 1.2400000095367432, 0.7160000205039978 ], [ 1.531999945640564, 0.18799999356269836 ], [ -0.2240000069141388, 0.7720000147819519 ] ],
    	[ [ 0.035999998450279236, 1.2319999933242798 ], [ 0.2720000147819519, -1.100000023841858 ], [ -1.2000000476837158, -1.8960000276565552 ] ]
    ]
    Outputs Statistics: {meanExponent=-0.166634521384747, negative=7, min=-1.8960000276565552, max=-1.8960000276565552, mean=0.10266666693819894, count=18.0, positive=11, stdDev=1.0843877988133321, zeros=0}
    Feedback for input 0
    Inputs Values: [
    	[ [ -1.884, 0.7 ], [ -0.764, -0.432 ], [ 1.544, 1.116 ] ],
    	[ [ 1.24, 0.716 ], [ 1.532, 0.188 ], [ -0.224, 0.772 ] ],
    	[ [ 0.036, 1.232 ], [ 0.272, -1.1 ], [ -1.2, -1.896 ] ]
    ]
    Value Statistics: {meanExponent=-0.16663452488883118, negative=7, min=-1.896, max=-1.896, mean=0.10266666666666671, count=18.0, positive=11, stdDev=1.0843877945130558, zeros=0}
    Implemented Feedback: [ [ 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, ... ], ... ]
    Implemented Statistics: {meanExponent=0.0, negative=0, min=1.0, max=1.0, mean=0.05555555555555555, count=324.0, positive=18, stdDev=0.2290614236454256, zeros=306}
    Measured Feedback: [ [ 0.9989738464355469, 0.0, 0.0, 0.0, 0.0, 
```
...[skipping 818 bytes](etc/1.txt)...
```
    0, 0.0, 0.0, ... ], [ 0.0, 0.0, 1.6927719116210938E-5, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 1.659393310546875E-4, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 1.659393310546875E-4, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, -1.3208389282226562E-4, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.659393310546875E-4, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.6927719116210938E-5, ... ], ... ]
    Error Statistics: {meanExponent=-3.816904638012043, negative=5, min=1.659393310546875E-4, max=1.659393310546875E-4, mean=-4.118607367998288E-6, count=324.0, positive=13, stdDev=9.236493249889878E-5, zeros=306}
    Learning Gradient for weight set 0
    Weights: [ 0.0, 0.0 ]
    Implemented Gradient: [ [ 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ] ]
    Implemented Statistics: {meanExponent=0.0, negative=0, min=1.0, max=1.0, mean=0.5, count=36.0, positive=18, stdDev=0.5, zeros=18}
    Measured Gradient: [ [ 1.0001659393310547, 1.0001659393310547, 1.0000169277191162, 1.0001659393310547, 1.0001659393310547, 0.9998679161071777, 1.0001659393310547, 1.0000169277191162, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ] ]
    Measured Statistics: {meanExponent=4.6894872890629774E-5, negative=0, min=1.0001659393310547, max=1.0001659393310547, mean=0.5000539951854281, count=36.0, positive=18, stdDev=0.5000540002905323, zeros=18}
    Gradient Error: [ [ 1.659393310546875E-4, 1.659393310546875E-4, 1.6927719116210938E-5, 1.659393310546875E-4, 1.659393310546875E-4, -1.3208389282226562E-4, 1.659393310546875E-4, 1.6927719116210938E-5, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ] ]
    Error Statistics: {meanExponent=-3.956286866243621, negative=2, min=1.659393310546875E-4, max=1.659393310546875E-4, mean=5.399518542819553E-5, count=36.0, positive=16, stdDev=8.956079118113866E-5, zeros=18}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 2.0075e-05 +- 9.1565e-05 [0.0000e+00 - 1.0262e-03] (360#)
    relativeTol: 1.0039e-04 +- 1.0911e-04 [8.4638e-06 - 5.1334e-04] (36#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=2.0075e-05 +- 9.1565e-05 [0.0000e+00 - 1.0262e-03] (360#), relativeTol=1.0039e-04 +- 1.0911e-04 [8.4638e-06 - 5.1334e-04] (36#)}
```



### Performance
Code from [LayerTestBase.java:149](../../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L149) executed in 0.11 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 3.7332 +- 0.8818 [3.1348 - 9.9201]
    Learning performance: 3.6566 +- 0.3851 [2.8726 - 5.1866]
    
```

