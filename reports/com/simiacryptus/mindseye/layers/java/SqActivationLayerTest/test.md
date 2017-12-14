# SqActivationLayer
## SqActivationLayerTest
### Json Serialization
Code from [StandardLayerTests.java:68](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L68) executed in 0.00 seconds: 
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
      "id": "dbc8bcde-1a06-4e15-aa39-c3162c482b9a",
      "isFrozen": true,
      "name": "SqActivationLayer/dbc8bcde-1a06-4e15-aa39-c3162c482b9a"
    }
```



### Example Input/Output Pair
Code from [StandardLayerTests.java:152](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L152) executed in 0.00 seconds: 
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
    	[ [ 0.796 ], [ -0.132 ], [ 0.188 ] ],
    	[ [ -1.512 ], [ -1.604 ], [ 0.324 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 0.6336160000000001 ], [ 0.017424000000000002 ], [ 0.035344 ] ],
    	[ [ 2.286144 ], [ 2.5728160000000004 ], [ 0.104976 ] ]
    ]
    --------------------
    Derivative: 
    [
    	[ [ 1.592 ], [ -0.264 ], [ 0.376 ] ],
    	[ [ -3.024 ], [ -3.208 ], [ 0.648 ] ]
    ]
```



### Batch Execution
Code from [StandardLayerTests.java:101](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L101) executed in 0.00 seconds: 
```java
    return getBatchingTester().test(layer, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (120#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (120#)}
```



### Differential Validation
Code from [StandardLayerTests.java:109](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L109) executed in 0.00 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ 0.616 ], [ -1.448 ], [ 1.28 ] ],
    	[ [ 0.868 ], [ -0.456 ], [ -0.368 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.12985139496868892, negative=3, min=-0.368, max=-0.368, mean=0.08200000000000002, count=6.0, positive=3, stdDev=0.9283497903987125, zeros=0}
    Output: [
    	[ [ 0.379456 ], [ 2.096704 ], [ 1.6384 ] ],
    	[ [ 0.753424 ], [ 0.207936 ], [ 0.135424 ] ]
    ]
    Outputs Statistics: {meanExponent=-0.25970278993737783, negative=0, min=0.135424, max=0.135424, mean=0.8685573333333334, count=6.0, positive=6, stdDev=0.7446904049645209, zeros=0}
    Feedback for input 0
    Inputs Values: [
    	[ [ 0.616 ], [ -1.448 ], [ 1.28 ] ],
    	[ [ 0.868 ], [ -0.456 ], [ -0.368 ] ]
    ]
    Value Statistics: {meanExponent=-0.12985139496868892, negative=3, min=-0.368, max=-0.368, mean=0.08200000000000002, count=6.0, positive=3, stdDev=0.9283497903987125, zeros=0}
    Implemented Feedback: [ [ 1.232, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 1.736, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, -2.896, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, -0.912, 0.0, 0.0 ], [ 0.0, 
```
...[skipping 461 bytes](etc/147.txt)...
```
    ], [ 0.0, 0.0, 0.0, 0.0, 2.560099999999732, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, -0.735899999999845 ] ]
    Measured Statistics: {meanExponent=0.17117120063484792, negative=3, min=-0.735899999999845, max=-0.735899999999845, mean=0.02734999999999081, count=36.0, positive=3, stdDev=0.7604575386274731, zeros=30}
    Feedback Error: [ [ 9.99999998472223E-5, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 9.999999935206283E-5, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 1.0000000049270596E-4, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 1.0000000009002807E-4, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 9.999999973198115E-5, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 1.0000000015497612E-4 ] ]
    Error Statistics: {meanExponent=-4.000000000239602, negative=0, min=1.0000000015497612E-4, max=1.0000000015497612E-4, mean=1.666666665747157E-5, count=36.0, positive=6, stdDev=3.726779960443563E-5, zeros=30}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 1.6667e-05 +- 3.7268e-05 [0.0000e+00 - 1.0000e-04] (36#)
    relativeTol: 3.8158e-05 +- 1.8447e-05 [1.7265e-05 - 6.7939e-05] (6#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=1.6667e-05 +- 3.7268e-05 [0.0000e+00 - 1.0000e-04] (36#), relativeTol=3.8158e-05 +- 1.8447e-05 [1.7265e-05 - 6.7939e-05] (6#)}
```



### Performance
Code from [StandardLayerTests.java:119](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L119) executed in 0.09 seconds: 
```java
    getPerformanceTester().test(layer, permPrototype);
```
Logging: 
```
    100 batches
    Input Dimensions:
    	[100, 100, 1]
    Performance:
    	Evaluation performance: 0.003445s +- 0.000227s [0.003142s - 0.003698s]
    	Learning performance: 0.010506s +- 0.000285s [0.010061s - 0.010771s]
    
```

### Function Plots
Code from [ActivationLayerTestBase.java:110](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/java/ActivationLayerTestBase.java#L110) executed in 0.00 seconds: 
```java
    return plot("Value Plot", plotData, x -> new double[]{x[0], x[1]});
```

Returns: 

![Result](etc/test.682.png)



Code from [ActivationLayerTestBase.java:114](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/java/ActivationLayerTestBase.java#L114) executed in 0.00 seconds: 
```java
    return plot("Derivative Plot", plotData, x -> new double[]{x[0], x[2]});
```

Returns: 

![Result](etc/test.683.png)



