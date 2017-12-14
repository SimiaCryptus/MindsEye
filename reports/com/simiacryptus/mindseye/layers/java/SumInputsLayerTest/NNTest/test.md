# SumInputsLayer
## NNTest
### Json Serialization
Code from [StandardLayerTests.java:68](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L68) executed in 0.00 seconds: 
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
      "id": "2b6c5db9-d633-4518-bea0-8a0b4d3c8381",
      "isFrozen": false,
      "name": "SumInputsLayer/2b6c5db9-d633-4518-bea0-8a0b4d3c8381"
    }
```



### Example Input/Output Pair
Code from [StandardLayerTests.java:152](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L152) executed in 0.00 seconds: 
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
    [[ -1.68, 1.616, -1.112 ],
    [ -1.672, 1.704, 0.904 ]]
    --------------------
    Output: 
    [ -3.352, 3.3200000000000003, -0.20800000000000007 ]
    --------------------
    Derivative: 
    [ 1.0, 1.0, 1.0 ],
    [ 1.0, 1.0, 1.0 ]
```



### Batch Execution
Code from [StandardLayerTests.java:101](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L101) executed in 0.00 seconds: 
```java
    return getBatchingTester().test(layer, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (90#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (90#)}
```



### Differential Validation
Code from [StandardLayerTests.java:109](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L109) executed in 0.00 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [ 0.94, 1.188, 0.996 ],
    [ -0.66, -1.964, 1.88 ]
    Inputs Statistics: {meanExponent=0.01540121088952403, negative=0, min=0.996, max=0.996, mean=1.0413333333333334, count=3.0, positive=3, stdDev=0.10619897467594533, zeros=0},
    {meanExponent=0.1289477560854931, negative=2, min=1.88, max=1.88, mean=-0.24800000000000008, count=3.0, positive=1, stdDev=1.5961186254995794, zeros=0}
    Output: [ 0.2799999999999999, -0.776, 2.876 ]
    Outputs Statistics: {meanExponent=-0.06806378856291584, negative=1, min=2.876, max=2.876, mean=0.7933333333333333, count=3.0, positive=2, stdDev=1.5344726202256664, zeros=0}
    Feedback for input 0
    Inputs Values: [ 0.94, 1.188, 0.996 ]
    Value Statistics: {meanExponent=0.01540121088952403, negative=0, min=0.996, max=0.996, mean=1.0413333333333334, count=3.0, positive=3, stdDev=0.10619897467594533, zeros=0}
    Implemented Feedback: [ [ 1.0, 0.0, 0.0 ], [ 0.0, 1.0, 0.0 ], [ 0.0, 0.0, 1.0 ] ]
    Implemented Statistics: {meanExponent=0.0, negative=0, min=1.0, max=1.0, mean=0.3333333333333333, cou
```
...[skipping 1044 bytes](etc/150.txt)...
```
    0 ] ]
    Implemented Statistics: {meanExponent=0.0, negative=0, min=1.0, max=1.0, mean=0.3333333333333333, count=9.0, positive=3, stdDev=0.4714045207910317, zeros=6}
    Measured Feedback: [ [ 0.9999999999998899, 0.0, 0.0 ], [ 0.0, 0.9999999999998899, 0.0 ], [ 0.0, 0.0, 1.0000000000021103 ] ]
    Measured Statistics: {meanExponent=2.7361184650972856E-13, negative=0, min=1.0000000000021103, max=1.0000000000021103, mean=0.3333333333335433, count=9.0, positive=3, stdDev=0.4714045207913287, zeros=6}
    Feedback Error: [ [ -1.1013412404281553E-13, 0.0, 0.0 ], [ 0.0, -1.1013412404281553E-13, 0.0 ], [ 0.0, 0.0, 2.1103119252074976E-12 ] ]
    Error Statistics: {meanExponent=-12.530603180987852, negative=2, min=2.1103119252074976E-12, max=2.1103119252074976E-12, mean=2.1000485301354073E-13, count=9.0, positive=1, stdDev=6.733627986644662E-13, zeros=6}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 2.5895e-13 +- 6.5610e-13 [0.0000e+00 - 2.1103e-12] (18#)
    relativeTol: 3.8843e-13 +- 4.7145e-13 [5.5067e-14 - 1.0552e-12] (6#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=2.5895e-13 +- 6.5610e-13 [0.0000e+00 - 2.1103e-12] (18#), relativeTol=3.8843e-13 +- 4.7145e-13 [5.5067e-14 - 1.0552e-12] (6#)}
```



### Performance
Code from [StandardLayerTests.java:119](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L119) executed in 0.00 seconds: 
```java
    getPerformanceTester().test(layer, permPrototype);
```
Logging: 
```
    100 batches
    Input Dimensions:
    	[100]
    	[100]
    Performance:
    	Evaluation performance: 0.000473s +- 0.000425s [0.000122s - 0.001099s]
    	Learning performance: 0.000217s +- 0.000005s [0.000213s - 0.000224s]
    
```

