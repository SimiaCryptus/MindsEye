# SumInputsLayer
## N1Test
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
      "id": "bdf188fb-4816-4190-8326-78d6cec0c248",
      "isFrozen": false,
      "name": "SumInputsLayer/bdf188fb-4816-4190-8326-78d6cec0c248"
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
    [[ -0.72, 1.18, -1.116 ],
    [ 0.604 ]]
    --------------------
    Output: 
    [ -0.11599999999999999, 1.7839999999999998, -0.5120000000000001 ]
    --------------------
    Derivative: 
    [ 1.0, 1.0, 1.0 ],
    [ 3.0 ]
```



### Batch Execution
Code from [StandardLayerTests.java:101](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L101) executed in 0.00 seconds: 
```java
    return getBatchingTester().test(layer, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (70#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (70#)}
```



### Differential Validation
Code from [StandardLayerTests.java:109](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L109) executed in 0.00 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [ 1.136, -1.976, 1.928 ],
    [ -1.66 ]
    Inputs Statistics: {meanExponent=0.21209076706447375, negative=1, min=1.928, max=1.928, mean=0.36266666666666664, count=3.0, positive=2, stdDev=1.6849999670293434, zeros=0},
    {meanExponent=0.22010808804005508, negative=1, min=-1.66, max=-1.66, mean=-1.66, count=1.0, positive=0, stdDev=0.0, zeros=0}
    Output: [ -0.524, -3.636, 0.268 ]
    Outputs Statistics: {meanExponent=-0.09730334814585152, negative=2, min=0.268, max=0.268, mean=-1.2973333333333334, count=3.0, positive=1, stdDev=1.6849999670293436, zeros=0}
    Feedback for input 0
    Inputs Values: [ 1.136, -1.976, 1.928 ]
    Value Statistics: {meanExponent=0.21209076706447375, negative=1, min=1.928, max=1.928, mean=0.36266666666666664, count=3.0, positive=2, stdDev=1.6849999670293434, zeros=0}
    Implemented Feedback: [ [ 1.0, 0.0, 0.0 ], [ 0.0, 1.0, 0.0 ], [ 0.0, 0.0, 1.0 ] ]
    Implemented Statistics: {meanExponent=0.0, negative=0, min=1.0, max=1.0, mean=0.3333333333333333, count=9.0, positive=3, stdDev=0.4714045207910317, z
```
...[skipping 816 bytes](etc/149.txt)...
```
    =-1.66, mean=-1.66, count=1.0, positive=0, stdDev=0.0, zeros=0}
    Implemented Feedback: [ [ 1.0, 1.0, 1.0 ] ]
    Implemented Statistics: {meanExponent=0.0, negative=0, min=1.0, max=1.0, mean=1.0, count=3.0, positive=3, stdDev=0.0, zeros=0}
    Measured Feedback: [ [ 0.9999999999998899, 1.0000000000021103, 0.9999999999998899 ] ]
    Measured Statistics: {meanExponent=2.7361184650972856E-13, negative=0, min=0.9999999999998899, max=0.9999999999998899, mean=1.00000000000063, count=3.0, positive=3, stdDev=1.4901161193847656E-8, zeros=0}
    Feedback Error: [ [ -1.1013412404281553E-13, 2.1103119252074976E-12, -1.1013412404281553E-13 ] ]
    Error Statistics: {meanExponent=-12.530603180987853, negative=2, min=-1.1013412404281553E-13, max=-1.1013412404281553E-13, mean=6.300145590406222E-13, count=3.0, positive=1, stdDev=1.0467283057891834E-12, zeros=0}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 3.8843e-13 +- 7.7162e-13 [0.0000e+00 - 2.1103e-12] (12#)
    relativeTol: 3.8843e-13 +- 4.7145e-13 [5.5067e-14 - 1.0552e-12] (6#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=3.8843e-13 +- 7.7162e-13 [0.0000e+00 - 2.1103e-12] (12#), relativeTol=3.8843e-13 +- 4.7145e-13 [5.5067e-14 - 1.0552e-12] (6#)}
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
    	[1]
    Performance:
    	Evaluation performance: 0.000115s +- 0.000011s [0.000098s - 0.000128s]
    	Learning performance: 0.000217s +- 0.000011s [0.000208s - 0.000238s]
    
```

