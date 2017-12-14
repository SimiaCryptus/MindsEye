# CrossDifferenceLayer
## CrossDifferenceLayerTest
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
      "class": "com.simiacryptus.mindseye.layers.java.CrossDifferenceLayer",
      "id": "87abf51b-cffd-4979-8fa4-57eb7025dacf",
      "isFrozen": false,
      "name": "CrossDifferenceLayer/87abf51b-cffd-4979-8fa4-57eb7025dacf"
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
    [[ -1.084, -1.288, 0.152, 0.792 ]]
    --------------------
    Output: 
    [ 0.20399999999999996, -1.236, -1.8760000000000001, -1.44, -2.08, -0.64 ]
    --------------------
    Derivative: 
    [ 3.0, 1.0, -1.0, -3.0 ]
```



### Batch Execution
Code from [StandardLayerTests.java:101](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L101) executed in 0.00 seconds: 
```java
    return getBatchingTester().test(layer, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (100#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (100#)}
```



### Differential Validation
Code from [StandardLayerTests.java:109](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L109) executed in 0.00 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [ -1.712, -1.932, -1.412, 0.956 ]
    Inputs Statistics: {meanExponent=0.16245086785312346, negative=3, min=0.956, max=0.956, mean=-1.025, count=4.0, positive=1, stdDev=1.1585279452822879, zeros=0}
    Output: [ 0.21999999999999997, -0.30000000000000004, -2.668, -0.52, -2.888, -2.368 ]
    Outputs Statistics: {meanExponent=-0.03388133477172282, negative=5, min=-2.368, max=-2.368, mean=-1.4206666666666667, count=6.0, positive=1, stdDev=1.2493484524165208, zeros=0}
    Feedback for input 0
    Inputs Values: [ -1.712, -1.932, -1.412, 0.956 ]
    Value Statistics: {meanExponent=0.16245086785312346, negative=3, min=0.956, max=0.956, mean=-1.025, count=4.0, positive=1, stdDev=1.1585279452822879, zeros=0}
    Implemented Feedback: [ [ 1.0, 1.0, 1.0, 0.0, 0.0, 0.0 ], [ -1.0, 0.0, 0.0, 1.0, 1.0, 0.0 ], [ 0.0, -1.0, 0.0, -1.0, 0.0, 1.0 ], [ 0.0, 0.0, -1.0, 0.0, -1.0, -1.0 ] ]
    Implemented Statistics: {meanExponent=0.0, negative=6, min=-1.0, max=-1.0, mean=0.0, count=24.0, positive=6, stdDev=0.7071067811865476, zeros=12}
    Measured 
```
...[skipping 314 bytes](etc/103.txt)...
```
    0000000000021103 ] ]
    Measured Statistics: {meanExponent=-4.7830642341580995E-14, negative=6, min=-1.0000000000021103, max=-1.0000000000021103, mean=1.8503717077085943E-13, count=24.0, positive=6, stdDev=0.7071067811864696, zeros=12}
    Feedback Error: [ [ -1.1013412404281553E-13, -1.1013412404281553E-13, 2.1103119252074976E-12, 0.0, 0.0, 0.0 ], [ 1.1013412404281553E-13, 0.0, 0.0, -1.1013412404281553E-13, -2.3305801732931286E-12, 0.0 ], [ 0.0, 1.1013412404281553E-13, 0.0, 1.1013412404281553E-13, 0.0, 2.1103119252074976E-12 ], [ 0.0, 0.0, 2.3305801732931286E-12, 0.0, 2.3305801732931286E-12, -2.1103119252074976E-12 ] ]
    Error Statistics: {meanExponent=-12.306086373864746, negative=5, min=-2.1103119252074976E-12, max=-2.1103119252074976E-12, mean=1.8503717077085943E-13, count=24.0, positive=7, stdDev=1.0974612396254757E-12, zeros=12}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 5.8265e-13 +- 9.4825e-13 [0.0000e+00 - 2.3306e-12] (24#)
    relativeTol: 5.8265e-13 +- 5.2901e-13 [5.5067e-14 - 1.1653e-12] (12#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=5.8265e-13 +- 9.4825e-13 [0.0000e+00 - 2.3306e-12] (24#), relativeTol=5.8265e-13 +- 5.2901e-13 [5.5067e-14 - 1.1653e-12] (12#)}
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
    	Evaluation performance: 0.000284s +- 0.000067s [0.000229s - 0.000395s]
    	Learning performance: 0.000055s +- 0.000012s [0.000042s - 0.000073s]
    
```

