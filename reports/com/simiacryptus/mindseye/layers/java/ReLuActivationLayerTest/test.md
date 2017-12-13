# ReLuActivationLayer
## ReLuActivationLayerTest
### Json Serialization
Code from [StandardLayerTests.java:69](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L69) executed in 0.00 seconds: 
```java
    JsonObject json = layer.getJson();
    NNLayer echo = NNLayer.fromJson(json);
    if ((echo == null)) throw new AssertionError("Failed to deserialize");
    if ((layer == echo)) throw new AssertionError("Serialization did not copy");
    if ((!layer.equals(echo))) throw new AssertionError("Serialization not equal");
    return new GsonBuilder().setPrettyPrinting().create().toJson(json);
```

Returns: 

```
    {
      "class": "com.simiacryptus.mindseye.layers.java.ReLuActivationLayer",
      "id": "58fc1c1d-c23d-4040-9415-dc4182d5a752",
      "isFrozen": true,
      "name": "ReLuActivationLayer/58fc1c1d-c23d-4040-9415-dc4182d5a752",
      "weights": [
        1.0
      ]
    }
```



### Example Input/Output Pair
Code from [StandardLayerTests.java:153](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L153) executed in 0.00 seconds: 
```java
    SimpleEval eval = SimpleEval.run(layer, inputPrototype);
    return String.format("--------------------\nInput: \n[%s]\n--------------------\nOutput: \n%s\n--------------------\nDerivative: \n%s",
      Arrays.stream(inputPrototype).map(t -> t.prettyPrint()).reduce((a, b) -> a + ",\n" + b).get(),
      eval.getOutput().prettyPrint(),
      Arrays.stream(eval.getDerivative()).map(t -> t.prettyPrint()).reduce((a, b) -> a + ",\n" + b).get());
```

Returns: 

```
    --------------------
    Input: 
    [[
    	[ [ 0.176 ], [ -1.04 ], [ 0.52 ] ],
    	[ [ -1.644 ], [ 0.112 ], [ -1.536 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 0.176 ], [ 0.0 ], [ 0.52 ] ],
    	[ [ 0.0 ], [ 0.112 ], [ 0.0 ] ]
    ]
    --------------------
    Derivative: 
    [
    	[ [ 1.0 ], [ 0.0 ], [ 1.0 ] ],
    	[ [ 0.0 ], [ 1.0 ], [ 0.0 ] ]
    ]
```



### Batch Execution
Code from [StandardLayerTests.java:102](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L102) executed in 0.00 seconds: 
```java
    return getBatchingTester().test(layer, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (120#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (46#)}
```



### Differential Validation
Code from [StandardLayerTests.java:110](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L110) executed in 0.00 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ -1.368 ], [ 0.22 ], [ 1.824 ] ],
    	[ [ -1.816 ], [ 0.68 ], [ 1.684 ] ]
    ]
    Inputs Statistics: {meanExponent=0.009583409375605747, negative=2, min=1.684, max=1.684, mean=0.20400000000000004, count=6.0, positive=4, stdDev=1.390036929965052, zeros=0}
    Output: [
    	[ [ 0.0 ], [ 0.22 ], [ 1.824 ] ],
    	[ [ 0.0 ], [ 0.68 ], [ 1.684 ] ]
    ]
    Outputs Statistics: {meanExponent=-0.08442537132888234, negative=0, min=1.684, max=1.684, mean=0.7346666666666667, count=6.0, positive=4, stdDev=0.7566616739923391, zeros=2}
    Feedback for input 0
    Inputs Values: [
    	[ [ -1.368 ], [ 0.22 ], [ 1.824 ] ],
    	[ [ -1.816 ], [ 0.68 ], [ 1.684 ] ]
    ]
    Value Statistics: {meanExponent=0.009583409375605747, negative=2, min=1.684, max=1.684, mean=0.20400000000000004, count=6.0, positive=4, stdDev=1.390036929965052, zeros=0}
    Implemented Feedback: [ [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 1.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 1.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 1.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0,
```
...[skipping 1315 bytes](etc/99.txt)...
```
    plemented Statistics: {meanExponent=-0.08442537132888234, negative=0, min=1.684, max=1.684, mean=0.7346666666666667, count=6.0, positive=4, stdDev=0.7566616739923391, zeros=2}
    Measured Gradient: [ [ 0.0, 0.0, 0.21999999999994246, 0.6799999999995698, 1.8239999999991596, 1.6839999999995747 ] ]
    Measured Statistics: {meanExponent=-0.08442537132905684, negative=0, min=1.6839999999995747, max=1.6839999999995747, mean=0.7346666666663744, count=6.0, positive=4, stdDev=0.7566616739920602, zeros=2}
    Gradient Error: [ [ 0.0, 0.0, -5.753730825119874E-14, -4.3021142204224816E-13, -8.404388296412435E-13, -4.2521541843143495E-13 ] ]
    Error Statistics: {meanExponent=-12.513313352421772, negative=4, min=-4.2521541843143495E-13, max=-4.2521541843143495E-13, mean=-2.922338297276876E-13, count=6.0, positive=0, stdDev=3.06358809381283E-13, zeros=2}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 5.2237e-14 +- 1.5503e-13 [0.0000e+00 - 8.4044e-13] (42#)
    relativeTol: 1.2800e-13 +- 9.1678e-14 [5.5067e-14 - 3.1633e-13] (8#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=5.2237e-14 +- 1.5503e-13 [0.0000e+00 - 8.4044e-13] (42#), relativeTol=1.2800e-13 +- 9.1678e-14 [5.5067e-14 - 3.1633e-13] (8#)}
```



### Performance
Code from [StandardLayerTests.java:120](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L120) executed in 0.00 seconds: 
```java
    getPerformanceTester().test(layer, permPrototype);
```
Logging: 
```
    Evaluation performance: 0.000213s +- 0.000050s [0.000162s - 0.000293s]
    Learning performance: 0.000203s +- 0.000090s [0.000133s - 0.000370s]
    
```

### Function Plots
Code from [ActivationLayerTestBase.java:103](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/java/ActivationLayerTestBase.java#L103) executed in 0.00 seconds: 
```java
    return plot("Value Plot", plotData, x -> new double[]{x[0], x[1]});
```

Returns: 

![Result](etc/test.42.png)



Code from [ActivationLayerTestBase.java:107](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/java/ActivationLayerTestBase.java#L107) executed in 0.00 seconds: 
```java
    return plot("Derivative Plot", plotData, x -> new double[]{x[0], x[2]});
```

Returns: 

![Result](etc/test.43.png)



