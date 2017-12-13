# SumInputsLayer
## NNTest
### Json Serialization
Code from [StandardLayerTests.java:69](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L69) executed in 0.00 seconds: 
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
      "class": "com.simiacryptus.mindseye.layers.java.SumInputsLayer",
      "id": "61713fea-cb25-46aa-8098-08fb2ebdd450",
      "isFrozen": false,
      "name": "SumInputsLayer/61713fea-cb25-46aa-8098-08fb2ebdd450"
    }
```



### Example Input/Output Pair
Code from [StandardLayerTests.java:153](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L153) executed in 0.00 seconds: 
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
    [[ -0.8, 0.64, 0.916 ],
    [ 1.168, 1.132, 0.188 ]]
    --------------------
    Output: 
    [ 0.3679999999999999, 1.7719999999999998, 1.104 ]
    --------------------
    Derivative: 
    [ 1.0, 1.0, 1.0 ],
    [ 1.0, 1.0, 1.0 ]
```



### Batch Execution
Code from [StandardLayerTests.java:102](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L102) executed in 0.00 seconds: 
```java
    return getBatchingTester().test(layer, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (90#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (90#)}
```



### Differential Validation
Code from [StandardLayerTests.java:110](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L110) executed in 0.00 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [ -0.644, 0.152, -0.236 ],
    [ -1.508, 1.704, 1.26 ]
    Inputs Statistics: {meanExponent=-0.5454528472417696, negative=2, min=-0.236, max=-0.236, mean=-0.24266666666666667, count=3.0, positive=1, stdDev=0.32499982905978414, zeros=0},
    {meanExponent=0.1700804923606665, negative=1, min=1.26, max=1.26, mean=0.48533333333333334, count=3.0, positive=2, stdDev=1.421106923805837, zeros=0}
    Output: [ -2.152, 1.8559999999999999, 1.024 ]
    Outputs Statistics: {meanExponent=0.20390673183900224, negative=1, min=1.024, max=1.024, mean=0.2426666666666666, count=3.0, positive=2, stdDev=1.727016180841653, zeros=0}
    Feedback for input 0
    Inputs Values: [ -0.644, 0.152, -0.236 ]
    Value Statistics: {meanExponent=-0.5454528472417696, negative=2, min=-0.236, max=-0.236, mean=-0.24266666666666667, count=3.0, positive=1, stdDev=0.32499982905978414, zeros=0}
    Implemented Feedback: [ [ 1.0, 0.0, 0.0 ], [ 0.0, 1.0, 0.0 ], [ 0.0, 0.0, 1.0 ] ]
    Implemented Statistics: {meanExponent=0.0, negative=0, min=1.0, max=1.0, mean=0.33333333333
```
...[skipping 1056 bytes](etc/109.txt)...
```
    ] ]
    Implemented Statistics: {meanExponent=0.0, negative=0, min=1.0, max=1.0, mean=0.3333333333333333, count=9.0, positive=3, stdDev=0.4714045207910317, zeros=6}
    Measured Feedback: [ [ 1.0000000000021103, 0.0, 0.0 ], [ 0.0, 0.9999999999998899, 0.0 ], [ 0.0, 0.0, 0.9999999999998899 ] ]
    Measured Statistics: {meanExponent=2.7361184650972856E-13, negative=0, min=0.9999999999998899, max=0.9999999999998899, mean=0.3333333333335433, count=9.0, positive=3, stdDev=0.4714045207913287, zeros=6}
    Feedback Error: [ [ 2.1103119252074976E-12, 0.0, 0.0 ], [ 0.0, -1.1013412404281553E-13, 0.0 ], [ 0.0, 0.0, -1.1013412404281553E-13 ] ]
    Error Statistics: {meanExponent=-12.530603180987853, negative=2, min=-1.1013412404281553E-13, max=-1.1013412404281553E-13, mean=2.1000485301354073E-13, count=9.0, positive=1, stdDev=6.733627986644662E-13, zeros=6}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 2.5895e-13 +- 6.5610e-13 [0.0000e+00 - 2.1103e-12] (18#)
    relativeTol: 3.8843e-13 +- 4.7145e-13 [5.5067e-14 - 1.0552e-12] (6#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=2.5895e-13 +- 6.5610e-13 [0.0000e+00 - 2.1103e-12] (18#), relativeTol=3.8843e-13 +- 4.7145e-13 [5.5067e-14 - 1.0552e-12] (6#)}
```



### Performance
Code from [StandardLayerTests.java:120](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L120) executed in 0.00 seconds: 
```java
    getPerformanceTester().test(layer, permPrototype);
```
Logging: 
```
    Evaluation performance: 0.000156s +- 0.000013s [0.000137s - 0.000172s]
    Learning performance: 0.000046s +- 0.000001s [0.000045s - 0.000048s]
    
```

