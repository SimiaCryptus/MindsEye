# L1NormalizationLayer
## L1NormalizationLayerTest
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
      "class": "com.simiacryptus.mindseye.layers.java.L1NormalizationLayer",
      "id": "2dfbdc5c-bdbd-4527-8369-44cfba627c06",
      "isFrozen": false,
      "name": "L1NormalizationLayer/2dfbdc5c-bdbd-4527-8369-44cfba627c06"
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
    [[ -174.0, 27.200000000000003, 41.6, 158.4 ]]
    --------------------
    Output: 
    [ -3.2706766917293235, 0.5112781954887219, 0.7819548872180453, 2.977443609022557 ]
    --------------------
    Derivative: 
    [ 0.0, 0.0, 0.0, 0.0 ]
```



### Batch Execution
Code from [StandardLayerTests.java:101](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L101) executed in 0.00 seconds: 
```java
    return getBatchingTester().test(layer, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (80#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (40#)}
```



### Differential Validation
Code from [StandardLayerTests.java:109](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L109) executed in 0.00 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [ 118.8, -89.60000000000001, -140.79999999999998, -130.0 ]
    Inputs Statistics: {meanExponent=2.0724176143550572, negative=3, min=-130.0, max=-130.0, mean=-60.4, count=4.0, positive=1, stdDev=105.20646367975685, zeros=0}
    Output: [ -0.4917218543046358, 0.37086092715231794, 0.5827814569536424, 0.5380794701986755 ]
    Outputs Statistics: {meanExponent=-0.31067931559403666, negative=1, min=0.5380794701986755, max=0.5380794701986755, mean=0.25, count=4.0, positive=3, stdDev=0.4354572172175367, zeros=0}
    Feedback for input 0
    Inputs Values: [ 118.8, -89.60000000000001, -140.79999999999998, -130.0 ]
    Value Statistics: {meanExponent=2.0724176143550572, negative=3, min=-130.0, max=-130.0, mean=-60.4, count=4.0, positive=1, stdDev=105.20646367975685, zeros=0}
    Implemented Feedback: [ [ -0.00617434542344634, 0.0015350203938423756, 0.002412174904609447, 0.002227150124994518 ], [ -0.002035272575764221, -0.0026040524538397435, 0.002412174904609447, 0.002227150124994518 ], [ -0.002035272575764221, 0.001535020393842375
```
...[skipping 771 bytes](etc/118.txt)...
```
    74 ] ]
    Measured Statistics: {meanExponent=-2.6625209008574346, negative=7, min=-0.0019119235150633074, max=-0.0019119235150633074, mean=-2.42861286636753E-13, count=16.0, positive=9, stdDev=0.0025418197677279483, zeros=0}
    Feedback Error: [ [ -2.555558062042329E-9, 6.347778669889281E-10, 9.97745982188325E-10, 9.219239894067699E-10 ], [ -8.418134778663544E-10, -1.078411605674734E-9, 9.97745982188325E-10, 9.219239894067699E-10 ], [ -8.418134778663544E-10, 6.347778669889281E-10, -7.154434904753371E-10, 9.219239894067699E-10 ], [ -8.418134778663544E-10, 6.347778669889281E-10, 9.97745982188325E-10, -7.923757060646769E-10 ] ]
    Error Statistics: {meanExponent=-9.04572280902276, negative=7, min=-7.923757060646769E-10, max=-7.923757060646769E-10, mean=-2.428613815044431E-13, count=16.0, positive=9, stdDev=1.0519228697099609E-9, zeros=0}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 9.5816e-10 +- 4.3413e-10 [6.3478e-10 - 2.5556e-09] (16#)
    relativeTol: 2.0690e-07 +- 1.3953e-10 [2.0677e-07 - 2.0722e-07] (16#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=9.5816e-10 +- 4.3413e-10 [6.3478e-10 - 2.5556e-09] (16#), relativeTol=2.0690e-07 +- 1.3953e-10 [2.0677e-07 - 2.0722e-07] (16#)}
```



### Performance
Code from [StandardLayerTests.java:119](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L119) executed in 0.00 seconds: 
```java
    getPerformanceTester().test(layer, permPrototype);
```
Logging: 
```
    100 batches
    Input Dimensions:
    	[4]
    Performance:
    	Evaluation performance: 0.000130s +- 0.000048s [0.000101s - 0.000227s]
    	Learning performance: 0.000039s +- 0.000010s [0.000026s - 0.000049s]
    
```

