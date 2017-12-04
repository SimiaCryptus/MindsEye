# ImgBandBiasLayer
## ImgBandBiasLayerTest
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
      "class": "com.simiacryptus.mindseye.layers.java.ImgBandBiasLayer",
      "id": "a864e734-2f23-44db-97c1-504000002bc5",
      "isFrozen": false,
      "name": "ImgBandBiasLayer/a864e734-2f23-44db-97c1-504000002bc5",
      "bias": [
        0.0,
        0.0,
        0.0
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
    	[ [ -0.084, -1.5, -1.348 ], [ -0.604, -1.58, -1.444 ] ],
    	[ [ -0.272, -0.032, -0.676 ], [ 1.088, 0.608, 1.068 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ -0.084, -1.5, -1.348 ], [ -0.604, -1.58, -1.444 ] ],
    	[ [ -0.272, -0.032, -0.676 ], [ 1.088, 0.608, 1.068 ] ]
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
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (240#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (240#)}
```



### Differential Validation
Code from [LayerTestBase.java:144](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.01 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ -0.084, -1.5, -1.348 ], [ -0.604, -1.58, -1.444 ] ],
    	[ [ -0.272, -0.032, -0.676 ], [ 1.088, 0.608, 1.068 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.2509924198542322, negative=9, min=1.068, max=1.068, mean=-0.39799999999999996, count=12.0, positive=3, stdDev=0.9285149433369396, zeros=0}
    Output: [
    	[ [ -0.084, -1.5, -1.348 ], [ -0.604, -1.58, -1.444 ] ],
    	[ [ -0.272, -0.032, -0.676 ], [ 1.088, 0.608, 1.068 ] ]
    ]
    Outputs Statistics: {meanExponent=-0.2509924198542322, negative=9, min=1.068, max=1.068, mean=-0.39799999999999996, count=12.0, positive=3, stdDev=0.9285149433369396, zeros=0}
    Feedback for input 0
    Inputs Values: [
    	[ [ -0.084, -1.5, -1.348 ], [ -0.604, -1.58, -1.444 ] ],
    	[ [ -0.272, -0.032, -0.676 ], [ 1.088, 0.608, 1.068 ] ]
    ]
    Value Statistics: {meanExponent=-0.2509924198542322, negative=9, min=1.068, max=1.068, mean=-0.39799999999999996, count=12.0, positive=3, stdDev=0.9285149433369396, zeros=0}
    Implemented Feedback: [ [ 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, ... ], ... ]
    Implemented Statistics: {meanExponent=0.0, negative=0, min=1.0, max=1.0, mean=0.08333333333333333, count=144.0, positive=12, stdDev=0.2763853991962833, zeros=132}
    Measured Feedback: [ [ 1.0000000000000286, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.9999999999998899, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.9999999999998899, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.9999999999998899, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.9999999999998899, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 1.0000000000000286, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9999999999998899, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.999999999999
```
...[skipping 601 bytes](etc/1.txt)...
```
    .0, 0.0, 0.0, 0.0, 2.864375403532904E-14, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.1013412404281553E-13, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.1013412404281553E-13, ... ], ... ]
    Error Statistics: {meanExponent=-13.055560092401983, negative=10, min=-1.1013412404281553E-13, max=-1.1013412404281553E-13, mean=-7.250373141371509E-15, count=144.0, positive=2, stdDev=2.830469176582722E-14, zeros=132}
    Learning Gradient for weight set 0
    Weights: [ 0.0, 0.0, 0.0 ]
    Implemented Gradient: [ [ 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ] ]
    Implemented Statistics: {meanExponent=0.0, negative=0, min=1.0, max=1.0, mean=0.3333333333333333, count=36.0, positive=12, stdDev=0.4714045207910317, zeros=24}
    Measured Gradient: [ [ 1.0000000000000286, 0.9999999999998899, 0.9999999999998899, 0.9999999999998899, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.9999999999998899, 1.0000000000000286, 0.9999999999998899, 0.9999999999998899, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ] ]
    Measured Statistics: {meanExponent=-3.7785564564448525E-14, negative=0, min=0.9999999999998899, max=0.9999999999998899, mean=0.33333333333330434, count=36.0, positive=12, stdDev=0.47140452079099066, zeros=24}
    Gradient Error: [ [ 2.864375403532904E-14, -1.1013412404281553E-13, -1.1013412404281553E-13, -1.1013412404281553E-13, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, -1.1013412404281553E-13, 2.864375403532904E-14, -1.1013412404281553E-13, -1.1013412404281553E-13, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ] ]
    Error Statistics: {meanExponent=-13.055560092401983, negative=10, min=-1.1013412404281553E-13, max=-1.1013412404281553E-13, mean=-2.9001492565486034E-14, count=36.0, positive=2, stdDev=5.0732705186738195E-14, zeros=24}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 1.2874e-14 +- 3.4644e-14 [0.0000e+00 - 1.1013e-13] (180#)
    relativeTol: 4.8276e-14 +- 1.5185e-14 [1.4322e-14 - 5.5067e-14] (24#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=1.2874e-14 +- 3.4644e-14 [0.0000e+00 - 1.1013e-13] (180#), relativeTol=4.8276e-14 +- 1.5185e-14 [1.4322e-14 - 5.5067e-14] (24#)}
```



### Performance
Code from [LayerTestBase.java:149](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L149) executed in 0.01 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 0.2117 +- 0.1030 [0.1681 - 0.9689]
    Learning performance: 0.0492 +- 0.0167 [0.0371 - 0.1738]
    
```

