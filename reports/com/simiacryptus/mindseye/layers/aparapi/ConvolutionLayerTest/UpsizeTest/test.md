# ConvolutionLayer
## UpsizeTest
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
      "class": "com.simiacryptus.mindseye.layers.aparapi.ConvolutionLayer",
      "id": "a864e734-2f23-44db-97c1-504000000018",
      "isFrozen": false,
      "name": "ConvolutionLayer/a864e734-2f23-44db-97c1-504000000018",
      "filter": {
        "dimensions": [
          3,
          3,
          6
        ],
        "data": [
          1.744,
          -0.196,
          1.408,
          0.536,
          -1.952,
          -1.872,
          0.368,
          -1.888,
          0.748,
          0.344,
          1.86,
          -1.008,
          -1.396,
          1.98,
          0.444,
          -1.32,
          1.364,
          -0.236,
          -1.656,
          0.376,
          0.752,
          1.02,
          0.268,
          1.176,
          -0.428,
          -0.728,
          0.484,
          -1.884,
          1.86,
          1.228,
          -1.292,
          0.008,
          -0.272,
          0.448,
          -0.528,
          1.504,
          -1.776,
          -1.856,
          -0.456,
          -1.92,
          1.304,
          -1.104,
          1.028,
          1.464,
          -0.044,
          -0.028,
          0.62,
          -1.96,
          -1.98,
          -0.148,
          -0.396,
          -1.288,
          0.828,
          0.828
        ]
      },
      "skip": {
        "dimensions": [
          1,
          1
        ]
      },
      "simple": false
    }
```



### Example Input/Output Pair
Code from [LayerTestBase.java:120](../../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L120) executed in 0.01 seconds: 
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
    	[ [ 0.668, -0.136 ], [ -1.84, -1.108 ], [ 0.712, 1.192 ] ],
    	[ [ -0.94, 0.948 ], [ 1.248, -0.192 ], [ -0.864, 0.52 ] ],
    	[ [ 1.74, 0.68 ], [ -0.288, 0.48 ], [ -1.312, 1.904 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 1.421216, 0.471328, -1.1024 ] ]
    ]
```



### Batch Execution
Code from [LayerTestBase.java:138](../../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L138) executed in 0.09 seconds: 
```java
    BatchingTester batchingTester = getBatchingTester();
    return batchingTester==null?null:batchingTester.test(layer, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (210#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (50#)}
```



### Differential Validation
Code from [LayerTestBase.java:144](../../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.35 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ 0.668, -0.136 ], [ -1.84, -1.108 ], [ 0.712, 1.192 ] ],
    	[ [ -0.94, 0.948 ], [ 1.248, -0.192 ], [ -0.864, 0.52 ] ],
    	[ [ 1.74, 0.68 ], [ -0.288, 0.48 ], [ -1.312, 1.904 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.12279494668878067, negative=8, min=1.904, max=1.904, mean=0.18955555555555553, count=18.0, positive=10, stdDev=1.0515200754792096, zeros=0}
    Output: [
    	[ [ 1.421216, 0.471328, -1.1024 ] ]
    ]
    Outputs Statistics: {meanExponent=-0.043892488943138595, negative=1, min=-1.1024, max=-1.1024, mean=0.2633813333333333, count=3.0, positive=2, stdDev=1.0407019397173984, zeros=0}
    Feedback for input 0
    Inputs Values: [
    	[ [ 0.668, -0.136 ], [ -1.84, -1.108 ], [ 0.712, 1.192 ] ],
    	[ [ -0.94, 0.948 ], [ 1.248, -0.192 ], [ -0.864, 0.52 ] ],
    	[ [ 1.74, 0.68 ], [ -0.288, 0.48 ], [ -1.312, 1.904 ] ]
    ]
    Value Statistics: {meanExponent=-0.12279494668878067, negative=8, min=1.904, max=1.904, mean=0.18955555555555553, count=18.0, positive=10, stdDev=1.0515200754792096, zeros=0}
    Implemented Feedback: [ [ 1.744, 0.344, -1.656 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], ... ]
    Implemented Statistics: {meanExponent=-0.17185880885690005, negative=4, min=0.0, max=0.0, mean=-0.06029629629629629, count=54.0, positive=2, stdDev=0.4793958246094236, zeros=48}
    Measured Feedback: [ [ 1.7439999999990796, 0.34399999999989994, -1.6559999999987696 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], ... ]
    Measured Statistics: {meanExponent=-0.17185880885489765, negative=4, min=0.0, max=0.0, mean=-0.0602962962963237, count=54.0, positive=2, stdDev=0.4793958246093434, zeros=48}
    Feedback Error: [ [ -9.203748874142548E-13, -1.000310945187266E-13, 1.2303491558895985E-12 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], ... ]
    Error Statistics: {meanExponent=-12.79109549546258, negative=4, min=0.0, max=0.0, mean=-2.7408566671415924E-14, count=54.0, positive=2, stdDev=2.6394152126150664E-13, zeros=48}
    Learning Gradient for weight set 0
    Weights: [ 1.744, -0.196, 1.408, 0.536, -1.952, -1.872, 0.368, -1.888, ... ]
    Implemented Gradient: [ [ 0.668, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], ... ]
    Implemented Statistics: {meanExponent=-0.5208423145771184, negative=3, min=0.0, max=0.0, mean=0.009851851851851853, count=162.0, positive=3, stdDev=0.09224351977789601, zeros=156}
    Measured Gradient: [ [ 0.6679999999992248, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], ... ]
    Measured Statistics: {meanExponent=-0.5208423145776208, negative=3, min=0.0, max=0.0, mean=0.00985185185186102, count=162.0, positive=3, stdDev=0.0922435197779057, zeros=156}
    Gradient Error: [ [ -7.752687380957468E-13, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], ... ]
    Error Statistics: {meanExponent=-12.460255615016655, negative=3, min=0.0, max=0.0, mean=9.166878504559317E-15, count=162.0, positive=3, stdDev=1.4244639354609365E-13, zeros=156}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 3.4570e-14 +- 1.7802e-13 [0.0000e+00 - 1.4452e-12] (216#)
    relativeTol: 1.7675e-12 +- 3.8585e-12 [6.2513e-17 - 1.4378e-11] (12#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=3.4570e-14 +- 1.7802e-13 [0.0000e+00 - 1.4452e-12] (216#), relativeTol=1.7675e-12 +- 3.8585e-12 [6.2513e-17 - 1.4378e-11] (12#)}
```



### Performance
Code from [LayerTestBase.java:149](../../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L149) executed in 0.76 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 26.1853 +- 3.5328 [21.8978 - 43.7842]
    Learning performance: 21.7198 +- 3.1111 [18.4438 - 41.4929]
    
```

