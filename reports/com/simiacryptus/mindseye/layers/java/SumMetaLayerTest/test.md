# SumMetaLayer
## SumMetaLayerTest
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
      "class": "com.simiacryptus.mindseye.layers.java.SumMetaLayer",
      "id": "46999917-0d55-4088-a110-1e2a73fc959f",
      "isFrozen": false,
      "name": "SumMetaLayer/46999917-0d55-4088-a110-1e2a73fc959f",
      "minBatches": 0
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
    [[ -0.276, 1.264, 1.064 ]]
    --------------------
    Output: 
    [ -0.276, 1.264, 1.064 ]
    --------------------
    Derivative: 
    [ 1.0, 1.0, 1.0 ]
```



### Differential Validation
Code from [StandardLayerTests.java:110](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L110) executed in 0.00 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [ 1.776, -0.664, 1.22 ],
    [ 0.188, -1.316, 0.448 ],
    [ -0.972, 0.616, 1.42 ]
    Inputs Statistics: {meanExponent=0.05265695716178265, negative=1, min=1.22, max=1.22, mean=0.7773333333333333, count=3.0, positive=2, stdDev=1.0441472863325154, zeros=0},
    {meanExponent=-0.3184360824867465, negative=1, min=0.448, max=0.448, mean=-0.2266666666666667, count=3.0, positive=2, stdDev=0.7775539995538716, zeros=0},
    {meanExponent=-0.0234882261754145, negative=1, min=1.42, max=1.42, mean=0.3546666666666667, count=3.0, positive=2, stdDev=0.9938602629254387, zeros=0}
    Output: [ 1.776, -0.664, 1.22 ]
    Outputs Statistics: {meanExponent=0.05265695716178265, negative=1, min=1.22, max=1.22, mean=0.7773333333333333, count=3.0, positive=2, stdDev=1.0441472863325154, zeros=0}
    Feedback for input 0
    Inputs Values: [ 1.776, -0.664, 1.22 ]
    Value Statistics: {meanExponent=0.05265695716178265, negative=1, min=1.22, max=1.22, mean=0.7773333333333333, count=3.0, positive=2, stdDev=1.0441472863325154, zeros=0}
    Implemented Feedback: [ [
```
...[skipping 2369 bytes](etc/110.txt)...
```
    ]
    Implemented Statistics: {meanExponent=0.0, negative=0, min=1.0, max=1.0, mean=0.3333333333333333, count=9.0, positive=3, stdDev=0.4714045207910317, zeros=6}
    Measured Feedback: [ [ 0.9999999999998899, 0.0, 0.0 ], [ 0.0, 1.0000000000021103, 0.0 ], [ 0.0, 0.0, 0.9999999999976694 ] ]
    Measured Statistics: {meanExponent=-4.7830642341759385E-14, negative=0, min=0.9999999999976694, max=0.9999999999976694, mean=0.3333333333332966, count=9.0, positive=3, stdDev=0.4714045207909798, zeros=6}
    Feedback Error: [ [ -1.1013412404281553E-13, 0.0, 0.0 ], [ 0.0, 2.1103119252074976E-12, 0.0 ], [ 0.0, 0.0, -2.3305801732931286E-12 ] ]
    Error Statistics: {meanExponent=-12.088755799140722, negative=2, min=-2.3305801732931286E-12, max=-2.3305801732931286E-12, mean=-3.671137468093851E-14, count=9.0, positive=1, stdDev=1.0480150744155462E-12, zeros=6}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 5.0567e-13 +- 9.1868e-13 [0.0000e+00 - 2.3306e-12] (27#)
    relativeTol: 7.5850e-13 +- 4.9943e-13 [5.5067e-14 - 1.1653e-12] (9#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=5.0567e-13 +- 9.1868e-13 [0.0000e+00 - 2.3306e-12] (27#), relativeTol=7.5850e-13 +- 4.9943e-13 [5.5067e-14 - 1.1653e-12] (9#)}
```



### Performance
Code from [StandardLayerTests.java:120](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L120) executed in 0.00 seconds: 
```java
    getPerformanceTester().test(layer, permPrototype);
```
Logging: 
```
    Evaluation performance: 0.000106s +- 0.000059s [0.000045s - 0.000188s]
    Learning performance: 0.000002s +- 0.000001s [0.000001s - 0.000005s]
    
```

