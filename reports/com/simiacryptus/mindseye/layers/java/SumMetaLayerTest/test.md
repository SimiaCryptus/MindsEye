# SumMetaLayer
## SumMetaLayerTest
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
      "class": "com.simiacryptus.mindseye.layers.java.SumMetaLayer",
      "id": "48ff1d13-14e9-4ed6-bce4-a11d2a27cf57",
      "isFrozen": false,
      "name": "SumMetaLayer/48ff1d13-14e9-4ed6-bce4-a11d2a27cf57",
      "minBatches": 0
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
    [[ 0.016, 0.872, -1.116 ]]
    --------------------
    Output: 
    [ 0.016, 0.872, -1.116 ]
    --------------------
    Derivative: 
    [ 1.0, 1.0, 1.0 ]
```



### Differential Validation
Code from [StandardLayerTests.java:109](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L109) executed in 0.00 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [ 0.992, 1.476, -1.256 ],
    [ 0.792, -1.656, -0.956 ],
    [ -1.928, 0.892, -0.08 ]
    Inputs Statistics: {meanExponent=0.08819588968079291, negative=1, min=-1.256, max=-1.256, mean=0.40399999999999997, count=3.0, positive=2, stdDev=1.1903120039160602, zeros=0},
    {meanExponent=0.03274780210481829, negative=2, min=-0.956, max=-0.956, mean=-0.6066666666666666, count=3.0, positive=1, stdDev=1.0294663126537404, zeros=0},
    {meanExponent=-0.28714604302170715, negative=2, min=-0.08, max=-0.08, mean=-0.37200000000000005, count=3.0, positive=1, stdDev=1.1696290010084394, zeros=0}
    Output: [ 0.992, 1.476, -1.256 ]
    Outputs Statistics: {meanExponent=0.08819588968079291, negative=1, min=-1.256, max=-1.256, mean=0.40399999999999997, count=3.0, positive=2, stdDev=1.1903120039160602, zeros=0}
    Feedback for input 0
    Inputs Values: [ 0.992, 1.476, -1.256 ]
    Value Statistics: {meanExponent=0.08819588968079291, negative=1, min=-1.256, max=-1.256, mean=0.40399999999999997, count=3.0, positive=2, stdDev=1.1903120039160602, zeros=0}
```
...[skipping 2408 bytes](etc/151.txt)...
```
    
    Implemented Statistics: {meanExponent=0.0, negative=0, min=1.0, max=1.0, mean=0.3333333333333333, count=9.0, positive=3, stdDev=0.4714045207910317, zeros=6}
    Measured Feedback: [ [ 0.9999999999998899, 0.0, 0.0 ], [ 0.0, 0.9999999999998899, 0.0 ], [ 0.0, 0.0, 0.9999999999976694 ] ]
    Measured Statistics: {meanExponent=-3.692731311925336E-13, negative=0, min=0.9999999999976694, max=0.9999999999976694, mean=0.33333333333304993, count=9.0, positive=3, stdDev=0.47140452079063083, zeros=6}
    Feedback Error: [ [ -1.1013412404281553E-13, 0.0, 0.0 ], [ 0.0, -1.1013412404281553E-13, 0.0 ], [ 0.0, 0.0, -2.3305801732931286E-12 ] ]
    Error Statistics: {meanExponent=-12.516230716189696, negative=3, min=-2.3305801732931286E-12, max=-2.3305801732931286E-12, mean=-2.834276023754177E-13, count=9.0, positive=0, stdDev=7.251729404930389E-13, zeros=6}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 2.8343e-13 +- 7.2517e-13 [0.0000e+00 - 2.3306e-12] (27#)
    relativeTol: 4.2514e-13 +- 5.2336e-13 [5.5067e-14 - 1.1653e-12] (9#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=2.8343e-13 +- 7.2517e-13 [0.0000e+00 - 2.3306e-12] (27#), relativeTol=4.2514e-13 +- 5.2336e-13 [5.5067e-14 - 1.1653e-12] (9#)}
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
    	[100]
    Performance:
    	Evaluation performance: 0.000431s +- 0.000096s [0.000320s - 0.000591s]
    	Learning performance: 0.000005s +- 0.000003s [0.000003s - 0.000011s]
    
```

