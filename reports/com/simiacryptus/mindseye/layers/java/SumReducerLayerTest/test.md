# SumReducerLayer
## SumReducerLayerTest
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
      "class": "com.simiacryptus.mindseye.layers.java.SumReducerLayer",
      "id": "100f2be0-c5ac-4097-9633-51682cb053a4",
      "isFrozen": false,
      "name": "SumReducerLayer/100f2be0-c5ac-4097-9633-51682cb053a4"
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
    [[ -0.272, -1.196, -0.904 ]]
    --------------------
    Output: 
    [ -2.372 ]
    --------------------
    Derivative: 
    [ 1.0, 1.0, 1.0 ]
```



### Batch Execution
Code from [StandardLayerTests.java:101](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L101) executed in 0.00 seconds: 
```java
    return getBatchingTester().test(layer, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (40#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (40#)}
```



### Differential Validation
Code from [StandardLayerTests.java:109](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L109) executed in 0.00 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [ -1.652, -1.572, -0.44 ]
    Inputs Statistics: {meanExponent=0.01930508705797997, negative=3, min=-0.44, max=-0.44, mean=-1.2213333333333334, count=3.0, positive=0, stdDev=0.5534505899857326, zeros=0}
    Output: [ -3.664 ]
    Outputs Statistics: {meanExponent=0.5639554649958128, negative=1, min=-3.664, max=-3.664, mean=-3.664, count=1.0, positive=0, stdDev=0.0, zeros=0}
    Feedback for input 0
    Inputs Values: [ -1.652, -1.572, -0.44 ]
    Value Statistics: {meanExponent=0.01930508705797997, negative=3, min=-0.44, max=-0.44, mean=-1.2213333333333334, count=3.0, positive=0, stdDev=0.5534505899857326, zeros=0}
    Implemented Feedback: [ [ 1.0 ], [ 1.0 ], [ 1.0 ] ]
    Implemented Statistics: {meanExponent=0.0, negative=0, min=1.0, max=1.0, mean=1.0, count=3.0, positive=3, stdDev=0.0, zeros=0}
    Measured Feedback: [ [ 1.0000000000021103 ], [ 1.0000000000021103 ], [ 0.9999999999976694 ] ]
    Measured Statistics: {meanExponent=2.736118465090148E-13, negative=0, min=0.9999999999976694, max=0.9999999999976694, mean=1.00000000000063, count=3.0, positive=3, stdDev=1.4901161193847656E-8, zeros=0}
    Feedback Error: [ [ 2.1103119252074976E-12 ], [ 2.1103119252074976E-12 ], [ -2.3305801732931286E-12 ] ]
    Error Statistics: {meanExponent=-11.66128088209175, negative=1, min=-2.3305801732931286E-12, max=-2.3305801732931286E-12, mean=6.300145590406222E-13, count=3.0, positive=2, stdDev=2.093456611578367E-12, zeros=0}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 2.1837e-12 +- 1.0384e-13 [2.1103e-12 - 2.3306e-12] (3#)
    relativeTol: 1.0919e-12 +- 5.1918e-14 [1.0552e-12 - 1.1653e-12] (3#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=2.1837e-12 +- 1.0384e-13 [2.1103e-12 - 2.3306e-12] (3#), relativeTol=1.0919e-12 +- 5.1918e-14 [1.0552e-12 - 1.1653e-12] (3#)}
```



### Performance
Code from [StandardLayerTests.java:119](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L119) executed in 0.03 seconds: 
```java
    getPerformanceTester().test(layer, permPrototype);
```
Logging: 
```
    100 batches
    Input Dimensions:
    	[100, 100, 1]
    Performance:
    	Evaluation performance: 0.006151s +- 0.011896s [0.000175s - 0.029942s]
    	Learning performance: 0.000040s +- 0.000012s [0.000030s - 0.000063s]
    
```

