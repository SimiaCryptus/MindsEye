# ReLuActivationLayer
## ReLuActivationLayerTest
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
      "class": "com.simiacryptus.mindseye.layers.java.ReLuActivationLayer",
      "id": "a864e734-2f23-44db-97c1-504000002c7e",
      "isFrozen": true,
      "name": "ReLuActivationLayer/a864e734-2f23-44db-97c1-504000002c7e",
      "weights": {
        "dimensions": [
          1
        ],
        "data": [
          1.0
        ]
      }
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
    	[ [ -1.048 ], [ 0.26 ], [ 0.596 ] ],
    	[ [ -1.416 ], [ 0.94 ], [ 0.924 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 0.0 ], [ 0.26 ], [ 0.596 ] ],
    	[ [ 0.0 ], [ 0.94 ], [ 0.924 ] ]
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
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (120#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (62#)}
```



### Differential Validation
Code from [LayerTestBase.java:144](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.00 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ -1.048 ], [ 0.26 ], [ 0.596 ] ],
    	[ [ -1.416 ], [ 0.94 ], [ 0.924 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.11659267191128038, negative=2, min=0.924, max=0.924, mean=0.04266666666666671, count=6.0, positive=4, stdDev=0.9356485570032278, zeros=0}
    Output: [
    	[ [ 0.0 ], [ 0.26 ], [ 0.596 ] ],
    	[ [ 0.0 ], [ 0.94 ], [ 0.924 ] ]
    ]
    Outputs Statistics: {meanExponent=-0.21774514186728505, negative=0, min=0.924, max=0.924, mean=0.4533333333333333, count=6.0, positive=4, stdDev=0.3930914510503745, zeros=2}
    Feedback for input 0
    Inputs Values: [
    	[ [ -1.048 ], [ 0.26 ], [ 0.596 ] ],
    	[ [ -1.416 ], [ 0.94 ], [ 0.924 ] ]
    ]
    Value Statistics: {meanExponent=-0.11659267191128038, negative=2, min=0.924, max=0.924, mean=0.04266666666666671, count=6.0, positive=4, stdDev=0.9356485570032278, zeros=0}
    Implemented Feedback: [ [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 1.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 1.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 1.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 1.0 ] ]
    Implemented Statistics: {meanExponent=0.0, negative=0, min=1.0, max=1.0, mean=0.1111111111111111, count=36.0, positive=4, stdDev=0.31426968052735443, zeros=32}
    Measured Feedback: [ [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.9999999999998899, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.9999999999998899, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.9999999999998899, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.9999999999998899 ] ]
    Measured Statistics: {meanExponent=-4.7830642341045674E-14, negative=0, min=0.9999999999998899, max=0.9999999999998899, mean=0.11111111111109888, count=36.0, positive=4, stdDev=0.31426968052731985, zeros=32}
    Feedback Error: [ [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, -1.1013412404281553E-13, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, -1.1013412404281553E-13, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, -1.1013412404281553E-13, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, -1.1013412404281553E-13 ] ]
    Error Statistics: {meanExponent=-12.958078098036825, negative=4, min=-1.1013412404281553E-13, max=-1.1013412404281553E-13, mean=-1.223712489364617E-14, count=36.0, positive=0, stdDev=3.461181597809566E-14, zeros=32}
    Learning Gradient for weight set 0
    Weights: [ 1.0 ]
    Implemented Gradient: [ [ 0.0, 0.0, 0.26, 0.94, 0.596, 0.924 ] ]
    Implemented Statistics: {meanExponent=-0.21774514186728505, negative=0, min=0.924, max=0.924, mean=0.4533333333333333, count=6.0, positive=4, stdDev=0.3930914510503745, zeros=2}
    Measured Gradient: [ [ 0.0, 0.0, 0.2599999999997049, 0.940000000000385, 0.596000000000485, 0.9239999999999249 ] ]
    Measured Statistics: {meanExponent=-0.21774514186728428, negative=0, min=0.9239999999999249, max=0.9239999999999249, mean=0.45333333333341663, count=6.0, positive=4, stdDev=0.39309145105049237, zeros=2}
    Gradient Error: [ [ 0.0, 0.0, -2.950972799453666E-13, 3.850253449400043E-13, 4.850564394587309E-13, -7.51620987671231E-14 ] ]
    Error Statistics: {meanExponent=-12.595688575489605, negative=2, min=-7.51620987671231E-14, max=-7.51620987671231E-14, mean=8.330373428104092E-14, count=6.0, positive=2, stdDev=2.691402789617916E-13, zeros=2}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 4.0021e-14 +- 1.0437e-13 [0.0000e+00 - 4.8506e-13] (42#)
    relativeTol: 1.8002e-13 +- 1.8845e-13 [4.0672e-14 - 5.6749e-13] (8#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=4.0021e-14 +- 1.0437e-13 [0.0000e+00 - 4.8506e-13] (42#), relativeTol=1.8002e-13 +- 1.8845e-13 [4.0672e-14 - 5.6749e-13] (8#)}
```



### Performance
Code from [LayerTestBase.java:149](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L149) executed in 0.00 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 0.1963 +- 0.0673 [0.1368 - 0.6326]
    Learning performance: 0.0427 +- 0.0170 [0.0313 - 0.1482]
    
```

### Function Plots
Code from [ActivationLayerTestBase.java:74](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/java/ActivationLayerTestBase.java#L74) executed in 0.00 seconds: 
```java
    return plot("Value Plot", plotData, x -> new double[]{x[0], x[1]});
```

Returns: 

![Result](etc/test.1.png)



Code from [ActivationLayerTestBase.java:78](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/java/ActivationLayerTestBase.java#L78) executed in 0.00 seconds: 
```java
    return plot("Derivative Plot", plotData, x -> new double[]{x[0], x[2]});
```

Returns: 

![Result](etc/test.2.png)



