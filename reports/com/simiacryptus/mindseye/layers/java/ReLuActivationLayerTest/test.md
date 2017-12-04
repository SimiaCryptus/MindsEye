# ReLuActivationLayer
## ReLuActivationLayerTest
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
      "class": "com.simiacryptus.mindseye.layers.java.ReLuActivationLayer",
      "id": "370a9587-74a1-4959-b406-fa4500002c7e",
      "isFrozen": true,
      "name": "ReLuActivationLayer/370a9587-74a1-4959-b406-fa4500002c7e",
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
    	[ [ -0.924 ], [ 0.456 ], [ 0.848 ] ],
    	[ [ 1.532 ], [ -1.456 ], [ 1.944 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 0.0 ], [ 0.456 ], [ 0.848 ] ],
    	[ [ 1.532 ], [ 0.0 ], [ 1.944 ] ]
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
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (120#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (70#)}
```



### Differential Validation
Code from [LayerTestBase.java:144](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.00 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ -0.924 ], [ 0.456 ], [ 0.848 ] ],
    	[ [ 1.532 ], [ -1.456 ], [ 1.944 ] ]
    ]
    Inputs Statistics: {meanExponent=0.031691511167519136, negative=2, min=1.944, max=1.944, mean=0.39999999999999997, count=6.0, positive=4, stdDev=1.2292892255283132, zeros=0}
    Output: [
    	[ [ 0.0 ], [ 0.456 ], [ 0.848 ] ],
    	[ [ 1.532 ], [ 0.0 ], [ 1.944 ] ]
    ]
    Outputs Statistics: {meanExponent=0.01532893020199743, negative=0, min=1.944, max=1.944, mean=0.7966666666666665, count=6.0, positive=4, stdDev=0.7354288242621143, zeros=2}
    Feedback for input 0
    Inputs Values: [
    	[ [ -0.924 ], [ 0.456 ], [ 0.848 ] ],
    	[ [ 1.532 ], [ -1.456 ], [ 1.944 ] ]
    ]
    Value Statistics: {meanExponent=0.031691511167519136, negative=2, min=1.944, max=1.944, mean=0.39999999999999997, count=6.0, positive=4, stdDev=1.2292892255283132, zeros=0}
    Implemented Feedback: [ [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 1.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 1.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 1.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 1.0 ] ]
    Implemented Statistics: {meanExponent=0.0, negative=0, min=1.0, max=1.0, mean=0.1111111111111111, count=36.0, positive=4, stdDev=0.31426968052735443, zeros=32}
    Measured Feedback: [ [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.9999999999998899, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.9999999999998899, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.9999999999998899, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.9999999999998899 ] ]
    Measured Statistics: {meanExponent=-4.7830642341045674E-14, negative=0, min=0.9999999999998899, max=0.9999999999998899, mean=0.11111111111109888, count=36.0, positive=4, stdDev=0.31426968052731985, zeros=32}
    Feedback Error: [ [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, -1.1013412404281553E-13, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, -1.1013412404281553E-13, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, -1.1013412404281553E-13, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, -1.1013412404281553E-13 ] ]
    Error Statistics: {meanExponent=-12.958078098036825, negative=4, min=-1.1013412404281553E-13, max=-1.1013412404281553E-13, mean=-1.223712489364617E-14, count=36.0, positive=0, stdDev=3.461181597809566E-14, zeros=32}
    Learning Gradient for weight set 0
    Weights: [ 1.0 ]
    Implemented Gradient: [ [ 0.0, 1.532, 0.456, 0.0, 0.848, 1.944 ] ]
    Implemented Statistics: {meanExponent=0.01532893020199743, negative=0, min=1.944, max=1.944, mean=0.7966666666666665, count=6.0, positive=4, stdDev=0.7354288242621143, zeros=2}
    Measured Gradient: [ [ 0.0, 1.5319999999996448, 0.4559999999997899, 0.0, 0.8479999999999599, 1.9440000000003899 ] ]
    Measured Statistics: {meanExponent=0.015328930201938859, negative=0, min=1.9440000000003899, max=1.9440000000003899, mean=0.7966666666666308, count=6.0, positive=4, stdDev=0.735428824262172, zeros=2}
    Gradient Error: [ [ 0.0, -3.552713678800501E-13, -2.1010970741031088E-13, 0.0, -4.007905118896815E-14, 3.89910326248355E-13 ] ]
    Error Statistics: {meanExponent=-12.73327787628307, negative=3, min=3.89910326248355E-13, max=3.89910326248355E-13, mean=-3.5924966705162355E-14, count=6.0, positive=1, stdDev=2.295853535710676E-13, zeros=2}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 3.4188e-14 +- 8.7753e-14 [0.0000e+00 - 3.8991e-13] (42#)
    relativeTol: 8.6315e-14 +- 6.0893e-14 [2.3632e-14 - 2.3038e-13] (8#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=3.4188e-14 +- 8.7753e-14 [0.0000e+00 - 3.8991e-13] (42#), relativeTol=8.6315e-14 +- 6.0893e-14 [2.3632e-14 - 2.3038e-13] (8#)}
```



### Performance
Code from [LayerTestBase.java:149](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L149) executed in 0.00 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 0.1696 +- 0.0233 [0.1396 - 0.3220]
    Learning performance: 0.0479 +- 0.0247 [0.0314 - 0.2166]
    
```

### Function Plots
Code from [ActivationLayerTestBase.java:74](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/java/ActivationLayerTestBase.java#L74) executed in 0.00 seconds: 
```java
    return plot("Value Plot", plotData, x -> new double[]{x[0], x[1]});
```

Returns: 

![Result](etc/test.1.png)



Code from [ActivationLayerTestBase.java:78](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/java/ActivationLayerTestBase.java#L78) executed in 0.00 seconds: 
```java
    return plot("Derivative Plot", plotData, x -> new double[]{x[0], x[2]});
```

Returns: 

![Result](etc/test.2.png)



