# GaussianActivationLayer
## GaussianActivationLayerTest
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
      "class": "com.simiacryptus.mindseye.layers.java.GaussianActivationLayer",
      "id": "afe3e22a-7438-4688-9d68-347c364818ab",
      "isFrozen": true,
      "name": "GaussianActivationLayer/afe3e22a-7438-4688-9d68-347c364818ab",
      "mean": 0.0,
      "stddev": 1.0
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
    	[ [ 1.804 ], [ 1.592 ], [ 0.612 ] ],
    	[ [ -0.008 ], [ 1.94 ], [ -0.952 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 0.07838313157916622 ], [ 0.11234615175777428 ], [ 0.33081018305004833 ] ],
    	[ [ 0.3989295144527161 ], [ 0.06076516895456478 ], [ 0.25357629539510257 ] ]
    ]
    --------------------
    Derivative: 
    [
    	[ [ -0.1414031693688159 ], [ -0.17885507359837668 ], [ -0.2024558320266296 ] ],
    	[ [ 0.003191436115621729 ], [ -0.11788442777185568 ], [ 0.24140463321613764 ] ]
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
    	[ [ 1.404 ], [ -0.24 ], [ 0.72 ] ],
    	[ [ 1.58 ], [ 1.056 ], [ -1.828 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.021798659585551727, negative=2, min=-1.828, max=-1.828, mean=0.44866666666666655, count=6.0, positive=4, stdDev=1.1756221993291704, zeros=0}
    Output: [
    	[ [ 0.1488901440526344 ], [ 0.38761661512501416 ], [ 0.30785126046985295 ] ],
    	[ [ 0.11450480025929236 ], [ 0.22843432378901604 ], [ 0.0750402582190496 ] ]
    ]
    Outputs Statistics: {meanExponent=-0.7429185592646251, negative=0, min=0.0750402582190496, max=0.0750402582190496, mean=0.21038956698580993, count=6.0, positive=6, stdDev=0.1099464448659379, zeros=0}
    Feedback for input 0
    Inputs Values: [
    	[ [ 1.404 ], [ -0.24 ], [ 0.72 ] ],
    	[ [ 1.58 ], [ 1.056 ], [ -1.828 ] ]
    ]
    Value Statistics: {meanExponent=-0.021798659585551727, negative=2, min=-1.828, max=-1.828, mean=0.44866666666666655, count=6.0, positive=4, stdDev=1.1756221993291704, zeros=0}
    Implemented Feedback: [ [ -0.20904176224989868, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, -0.1809175844
```
...[skipping 673 bytes](etc/68.txt)...
```
    , 0.0, 0.0, 0.0, -0.22166031967973954, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.13718237775572906 ] ]
    Measured Statistics: {meanExponent=-0.7647307016953535, negative=4, min=0.13718237775572906, max=0.13718237775572906, mean=-0.017295474945370575, count=36.0, positive=2, stdDev=0.07461893452421096, zeros=30}
    Feedback Error: [ [ 7.230583024653292E-6, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 8.56740116936261E-6, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, -1.8264951331489754E-5, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 1.3158085854780843E-6, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, -7.412141445423126E-6, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 8.785731306387712E-6 ] ]
    Error Statistics: {meanExponent=-5.168907454820406, negative=2, min=8.785731306387712E-6, max=8.785731306387712E-6, mean=6.178647471356053E-9, count=36.0, positive=4, stdDev=4.059108530066338E-6, zeros=30}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 1.4327e-06 +- 3.7979e-06 [0.0000e+00 - 1.8265e-05] (36#)
    relativeTol: 3.1770e-05 +- 3.0971e-05 [2.7273e-06 - 9.8179e-05] (6#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=1.4327e-06 +- 3.7979e-06 [0.0000e+00 - 1.8265e-05] (36#), relativeTol=3.1770e-05 +- 3.0971e-05 [2.7273e-06 - 9.8179e-05] (6#)}
```



### Performance
Code from [StandardLayerTests.java:120](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L120) executed in 0.00 seconds: 
```java
    getPerformanceTester().test(layer, permPrototype);
```
Logging: 
```
    Evaluation performance: 0.000194s +- 0.000010s [0.000183s - 0.000213s]
    Learning performance: 0.000047s +- 0.000005s [0.000041s - 0.000056s]
    
```

### Function Plots
Code from [ActivationLayerTestBase.java:103](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/java/ActivationLayerTestBase.java#L103) executed in 0.01 seconds: 
```java
    return plot("Value Plot", plotData, x -> new double[]{x[0], x[1]});
```

Returns: 

![Result](etc/test.15.png)



Code from [ActivationLayerTestBase.java:107](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/java/ActivationLayerTestBase.java#L107) executed in 0.01 seconds: 
```java
    return plot("Derivative Plot", plotData, x -> new double[]{x[0], x[2]});
```

Returns: 

![Result](etc/test.16.png)



