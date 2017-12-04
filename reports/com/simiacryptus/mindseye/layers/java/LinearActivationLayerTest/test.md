# LinearActivationLayer
## LinearActivationLayerTest
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
      "class": "com.simiacryptus.mindseye.layers.java.LinearActivationLayer",
      "id": "370a9587-74a1-4959-b406-fa4500002c0b",
      "isFrozen": false,
      "name": "LinearActivationLayer/370a9587-74a1-4959-b406-fa4500002c0b",
      "weights": {
        "dimensions": [
          2
        ],
        "data": [
          1.0,
          0.0
        ]
      }
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
    	[ [ 0.58 ], [ -1.248 ], [ 1.976 ] ],
    	[ [ -0.44 ], [ 1.472 ], [ 0.652 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 0.58 ], [ -1.248 ], [ 1.976 ] ],
    	[ [ -0.44 ], [ 1.472 ], [ 0.652 ] ]
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
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (120#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (120#)}
```



### Differential Validation
Code from [LayerTestBase.java:144](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.00 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ 0.58 ], [ -1.248 ], [ 1.976 ] ],
    	[ [ -0.44 ], [ 1.472 ], [ 0.652 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.036493733103243435, negative=2, min=0.652, max=0.652, mean=0.49866666666666665, count=6.0, positive=4, stdDev=1.087194963605373, zeros=0}
    Output: [
    	[ [ 0.58 ], [ -1.248 ], [ 1.976 ] ],
    	[ [ -0.44 ], [ 1.472 ], [ 0.652 ] ]
    ]
    Outputs Statistics: {meanExponent=-0.036493733103243435, negative=2, min=0.652, max=0.652, mean=0.49866666666666665, count=6.0, positive=4, stdDev=1.087194963605373, zeros=0}
    Feedback for input 0
    Inputs Values: [
    	[ [ 0.58 ], [ -1.248 ], [ 1.976 ] ],
    	[ [ -0.44 ], [ 1.472 ], [ 0.652 ] ]
    ]
    Value Statistics: {meanExponent=-0.036493733103243435, negative=2, min=0.652, max=0.652, mean=0.49866666666666665, count=6.0, positive=4, stdDev=1.087194963605373, zeros=0}
    Implemented Feedback: [ [ 1.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 1.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 1.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 1.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 1.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 1.0 ] ]
    Implemented Statistics: {meanExponent=0.0, negative=0, min=1.0, max=1.0, mean=0.16666666666666666, count=36.0, positive=6, stdDev=0.37267799624996495, zeros=30}
    Measured Feedback: [ [ 0.9999999999998899, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.9999999999998899, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.9999999999998899, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.9999999999998899, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.9999999999998899, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.9999999999998899 ] ]
    Measured Statistics: {meanExponent=-4.7830642341045674E-14, negative=0, min=0.9999999999998899, max=0.9999999999998899, mean=0.1666666666666483, count=36.0, positive=6, stdDev=0.3726779962499239, zeros=30}
    Feedback Error: [ [ -1.1013412404281553E-13, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, -1.1013412404281553E-13, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, -1.1013412404281553E-13, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, -1.1013412404281553E-13, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, -1.1013412404281553E-13, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, -1.1013412404281553E-13 ] ]
    Error Statistics: {meanExponent=-12.958078098036824, negative=6, min=-1.1013412404281553E-13, max=-1.1013412404281553E-13, mean=-1.8355687340469256E-14, count=36.0, positive=0, stdDev=4.104456466702158E-14, zeros=30}
    Learning Gradient for weight set 0
    Weights: [ 1.0, 0.0 ]
    Implemented Gradient: [ [ 0.58, -0.44, -1.248, 1.472, 1.976, 0.652 ], [ 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 ] ]
    Implemented Statistics: {meanExponent=-0.018246866551621718, negative=2, min=1.0, max=1.0, mean=0.7493333333333334, count=12.0, positive=10, stdDev=0.8085976887316844, zeros=0}
    Measured Gradient: [ [ 0.5800000000000249, -0.4399999999998849, -1.24800000000036, 1.4720000000001399, 1.9759999999990896, 0.6519999999998749 ], [ 0.9999999999998899, 0.9999999999998899, 0.9999999999998899, 0.9999999999998899, 0.9999999999998899, 0.9999999999998899 ] ]
    Measured Statistics: {meanExponent=-0.018246866551663285, negative=2, min=0.9999999999998899, max=0.9999999999998899, mean=0.7493333333331854, count=12.0, positive=10, stdDev=0.8085976887316236, zeros=0}
    Gradient Error: [ [ 2.4980018054066022E-14, 1.1507461650239748E-13, -3.5993430458347575E-13, 1.3988810110276972E-13, -9.103828801926284E-13, -1.2512213487525514E-13 ], [ -1.1013412404281553E-13, -1.1013412404281553E-13, -1.1013412404281553E-13, -1.1013412404281553E-13, -1.1013412404281553E-13, -1.1013412404281553E-13 ] ]
    Error Statistics: {meanExponent=-12.877611171730486, negative=9, min=-1.1013412404281553E-13, max=-1.1013412404281553E-13, mean=-1.4802511068741828E-13, count=12.0, positive=3, stdDev=2.609725466466176E-13, zeros=0}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 6.2437e-14 +- 1.4185e-13 [0.0000e+00 - 9.1038e-13] (48#)
    relativeTol: 7.3952e-14 +- 4.7631e-14 [2.1534e-14 - 2.3036e-13] (18#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=6.2437e-14 +- 1.4185e-13 [0.0000e+00 - 9.1038e-13] (48#), relativeTol=7.3952e-14 +- 4.7631e-14 [2.1534e-14 - 2.3036e-13] (18#)}
```



### Performance
Code from [LayerTestBase.java:149](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L149) executed in 0.00 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 0.1690 +- 0.0609 [0.1083 - 0.6583]
    Learning performance: 0.0388 +- 0.0166 [0.0285 - 0.1453]
    
```

### Function Plots
Code from [ActivationLayerTestBase.java:74](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/java/ActivationLayerTestBase.java#L74) executed in 0.01 seconds: 
```java
    return plot("Value Plot", plotData, x -> new double[]{x[0], x[1]});
```

Returns: 

![Result](etc/test.1.png)



Code from [ActivationLayerTestBase.java:78](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/java/ActivationLayerTestBase.java#L78) executed in 0.01 seconds: 
```java
    return plot("Derivative Plot", plotData, x -> new double[]{x[0], x[2]});
```

Returns: 

![Result](etc/test.2.png)



