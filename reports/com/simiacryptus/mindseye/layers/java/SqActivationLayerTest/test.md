# SqActivationLayer
## SqActivationLayerTest
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
      "class": "com.simiacryptus.mindseye.layers.java.SqActivationLayer",
      "id": "2a3bbbfa-393b-438c-913d-97b16f007075",
      "isFrozen": true,
      "name": "SqActivationLayer/2a3bbbfa-393b-438c-913d-97b16f007075"
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
    	[ [ -0.648 ], [ -1.604 ], [ 0.256 ] ],
    	[ [ -0.22 ], [ -1.344 ], [ -0.948 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 0.419904 ], [ 2.5728160000000004 ], [ 0.065536 ] ],
    	[ [ 0.0484 ], [ 1.8063360000000002 ], [ 0.898704 ] ]
    ]
    --------------------
    Derivative: 
    [
    	[ [ -1.296 ], [ -3.208 ], [ 0.512 ] ],
    	[ [ -0.44 ], [ -2.688 ], [ -1.896 ] ]
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
    	[ [ -0.08 ], [ 0.54 ], [ -1.036 ] ],
    	[ [ 0.152 ], [ -1.248 ], [ 1.016 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.3440341027561325, negative=3, min=1.016, max=1.016, mean=-0.10933333333333335, count=6.0, positive=3, stdDev=0.8075103026518541, zeros=0}
    Output: [
    	[ [ 0.0064 ], [ 0.2916 ], [ 1.073296 ] ],
    	[ [ 0.023104 ], [ 1.557504 ], [ 1.032256 ] ]
    ]
    Outputs Statistics: {meanExponent=-0.688068205512265, negative=0, min=1.032256, max=1.032256, mean=0.6640266666666667, count=6.0, positive=6, stdDev=0.5892583323859089, zeros=0}
    Feedback for input 0
    Inputs Values: [
    	[ [ -0.08 ], [ 0.54 ], [ -1.036 ] ],
    	[ [ 0.152 ], [ -1.248 ], [ 1.016 ] ]
    ]
    Value Statistics: {meanExponent=-0.3440341027561325, negative=3, min=1.016, max=1.016, mean=-0.10933333333333335, count=6.0, positive=3, stdDev=0.8075103026518541, zeros=0}
    Implemented Feedback: [ [ -0.16, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.304, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 1.08, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, -2.496, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.
```
...[skipping 455 bytes](etc/106.txt)...
```
     ], [ 0.0, 0.0, 0.0, 0.0, -2.0719000000002374, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 2.032100000000092 ] ]
    Measured Statistics: {meanExponent=-0.04302168378905492, negative=3, min=2.032100000000092, max=2.032100000000092, mean=-0.036427777777760934, count=36.0, positive=3, stdDev=0.6643419106366122, zeros=30}
    Feedback Error: [ [ 9.999999999552034E-5, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 9.999999994608766E-5, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 9.999999991716635E-5, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 1.0000000089283034E-4, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 9.99999997626233E-5, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 1.000000000921375E-4 ] ]
    Error Statistics: {meanExponent=-3.9999999995610978, negative=0, min=1.000000000921375E-4, max=1.000000000921375E-4, mean=1.666666668351015E-5, count=36.0, positive=6, stdDev=3.726779966265968E-5, zeros=30}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 1.6667e-05 +- 3.7268e-05 [0.0000e+00 - 1.0000e-04] (36#)
    relativeTol: 9.8685e-05 +- 1.0806e-04 [2.0032e-05 - 3.1260e-04] (6#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=1.6667e-05 +- 3.7268e-05 [0.0000e+00 - 1.0000e-04] (36#), relativeTol=9.8685e-05 +- 1.0806e-04 [2.0032e-05 - 3.1260e-04] (6#)}
```



### Performance
Code from [StandardLayerTests.java:120](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L120) executed in 0.00 seconds: 
```java
    getPerformanceTester().test(layer, permPrototype);
```
Logging: 
```
    Evaluation performance: 0.000167s +- 0.000012s [0.000154s - 0.000186s]
    Learning performance: 0.000043s +- 0.000004s [0.000039s - 0.000048s]
    
```

### Function Plots
Code from [ActivationLayerTestBase.java:103](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/java/ActivationLayerTestBase.java#L103) executed in 0.00 seconds: 
```java
    return plot("Value Plot", plotData, x -> new double[]{x[0], x[1]});
```

Returns: 

![Result](etc/test.49.png)



Code from [ActivationLayerTestBase.java:107](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/java/ActivationLayerTestBase.java#L107) executed in 0.00 seconds: 
```java
    return plot("Derivative Plot", plotData, x -> new double[]{x[0], x[2]});
```

Returns: 

![Result](etc/test.50.png)



