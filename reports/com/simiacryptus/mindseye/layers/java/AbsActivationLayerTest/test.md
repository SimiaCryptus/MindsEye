# AbsActivationLayer
## AbsActivationLayerTest
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
      "class": "com.simiacryptus.mindseye.layers.java.AbsActivationLayer",
      "id": "2cb04f03-7316-4f28-9901-18a85b514d06",
      "isFrozen": true,
      "name": "AbsActivationLayer/2cb04f03-7316-4f28-9901-18a85b514d06"
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
    	[ [ -1.428 ], [ -1.416 ], [ 1.152 ] ],
    	[ [ -0.484 ], [ 1.772 ], [ -0.444 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 1.428 ], [ 1.416 ], [ 1.152 ] ],
    	[ [ 0.484 ], [ 1.772 ], [ 0.444 ] ]
    ]
    --------------------
    Derivative: 
    [
    	[ [ -1.0 ], [ -1.0 ], [ 1.0 ] ],
    	[ [ -1.0 ], [ 1.0 ], [ -1.0 ] ]
    ]
```



### Batch Execution
Code from [StandardLayerTests.java:102](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L102) executed in 0.00 seconds: 
```java
    return getBatchingTester().test(layer, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (120#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (120#)}
```



### Differential Validation
Code from [StandardLayerTests.java:110](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L110) executed in 0.00 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ -1.384 ], [ -0.4 ], [ 1.016 ] ],
    	[ [ 0.44 ], [ -0.504 ], [ 0.388 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.21919921201291306, negative=3, min=0.388, max=0.388, mean=-0.07399999999999997, count=6.0, positive=3, stdDev=0.7824125084208031, zeros=0}
    Output: [
    	[ [ 1.384 ], [ 0.4 ], [ 1.016 ] ],
    	[ [ 0.44 ], [ 0.504 ], [ 0.388 ] ]
    ]
    Outputs Statistics: {meanExponent=-0.21919921201291306, negative=0, min=0.388, max=0.388, mean=0.6886666666666666, count=6.0, positive=6, stdDev=0.3786602112125799, zeros=0}
    Feedback for input 0
    Inputs Values: [
    	[ [ -1.384 ], [ -0.4 ], [ 1.016 ] ],
    	[ [ 0.44 ], [ -0.504 ], [ 0.388 ] ]
    ]
    Value Statistics: {meanExponent=-0.21919921201291306, negative=3, min=0.388, max=0.388, mean=-0.07399999999999997, count=6.0, positive=3, stdDev=0.7824125084208031, zeros=0}
    Implemented Feedback: [ [ -1.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 1.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, -1.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, -1.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 1.0, 0.0 ], [ 0.0, 0.0,
```
...[skipping 378 bytes](etc/53.txt)...
```
    .0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.9999999999998899, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.9999999999998899 ] ]
    Measured Statistics: {meanExponent=-4.7830642341045674E-14, negative=3, min=0.9999999999998899, max=0.9999999999998899, mean=0.0, count=36.0, positive=3, stdDev=0.40824829046381805, zeros=30}
    Feedback Error: [ [ 1.1013412404281553E-13, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, -1.1013412404281553E-13, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 1.1013412404281553E-13, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 1.1013412404281553E-13, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, -1.1013412404281553E-13, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, -1.1013412404281553E-13 ] ]
    Error Statistics: {meanExponent=-12.958078098036824, negative=3, min=-1.1013412404281553E-13, max=-1.1013412404281553E-13, mean=0.0, count=36.0, positive=3, stdDev=4.496206786221447E-14, zeros=30}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 1.8356e-14 +- 4.1045e-14 [0.0000e+00 - 1.1013e-13] (36#)
    relativeTol: 5.5067e-14 +- 0.0000e+00 [5.5067e-14 - 5.5067e-14] (6#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=1.8356e-14 +- 4.1045e-14 [0.0000e+00 - 1.1013e-13] (36#), relativeTol=5.5067e-14 +- 0.0000e+00 [5.5067e-14 - 5.5067e-14] (6#)}
```



### Performance
Code from [StandardLayerTests.java:120](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L120) executed in 0.00 seconds: 
```java
    getPerformanceTester().test(layer, permPrototype);
```
Logging: 
```
    Evaluation performance: 0.000283s +- 0.000036s [0.000233s - 0.000337s]
    Learning performance: 0.000041s +- 0.000003s [0.000037s - 0.000044s]
    
```

### Function Plots
Code from [ActivationLayerTestBase.java:103](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/java/ActivationLayerTestBase.java#L103) executed in 0.01 seconds: 
```java
    return plot("Value Plot", plotData, x -> new double[]{x[0], x[1]});
```

Returns: 

![Result](etc/test.9.png)



Code from [ActivationLayerTestBase.java:107](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/java/ActivationLayerTestBase.java#L107) executed in 0.01 seconds: 
```java
    return plot("Derivative Plot", plotData, x -> new double[]{x[0], x[2]});
```

Returns: 

![Result](etc/test.10.png)



