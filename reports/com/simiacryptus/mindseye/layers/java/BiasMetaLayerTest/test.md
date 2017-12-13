# BiasMetaLayer
## BiasMetaLayerTest
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
      "class": "com.simiacryptus.mindseye.layers.java.BiasMetaLayer",
      "id": "31ca3925-e1d1-45c5-baa2-5c948662e5fb",
      "isFrozen": false,
      "name": "BiasMetaLayer/31ca3925-e1d1-45c5-baa2-5c948662e5fb"
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
    [[ -1.044, 1.36, 0.944 ],
    [ 0.292, -1.844, -1.964 ]]
    --------------------
    Output: 
    [ -0.752, -0.484, -1.02 ]
    --------------------
    Derivative: 
    [ 1.0, 1.0, 1.0 ],
    [ 1.0, 1.0, 1.0 ]
```



### Differential Validation
Code from [StandardLayerTests.java:110](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L110) executed in 0.00 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [ -1.276, -1.508, 0.708 ],
    [ -1.552, 1.084, 1.548 ]
    Inputs Statistics: {meanExponent=0.04476175786955592, negative=2, min=0.708, max=0.708, mean=-0.6919999999999998, count=3.0, positive=1, stdDev=0.9944700431217961, zeros=0},
    {meanExponent=0.1385639851571372, negative=1, min=1.548, max=1.548, mean=0.36000000000000004, count=3.0, positive=2, stdDev=1.3651940033074665, zeros=0}
    Output: [ -2.8280000000000003, -0.42399999999999993, 2.2560000000000002 ]
    Outputs Statistics: {meanExponent=0.14406145234296638, negative=2, min=2.2560000000000002, max=2.2560000000000002, mean=-0.332, count=3.0, positive=1, stdDev=2.076553554971956, zeros=0}
    Feedback for input 0
    Inputs Values: [ -1.276, -1.508, 0.708 ]
    Value Statistics: {meanExponent=0.04476175786955592, negative=2, min=0.708, max=0.708, mean=-0.6919999999999998, count=3.0, positive=1, stdDev=0.9944700431217961, zeros=0}
    Implemented Feedback: [ [ 1.0, 0.0, 0.0 ], [ 0.0, 1.0, 0.0 ], [ 0.0, 0.0, 1.0 ] ]
    Implemented Statistics: {meanExponent=0.0, negative=0
```
...[skipping 1102 bytes](etc/59.txt)...
```
    ]
    Implemented Statistics: {meanExponent=0.0, negative=0, min=1.0, max=1.0, mean=0.3333333333333333, count=9.0, positive=3, stdDev=0.4714045207910317, zeros=6}
    Measured Feedback: [ [ 1.0000000000021103, 0.0, 0.0 ], [ 0.0, 0.9999999999998899, 0.0 ], [ 0.0, 0.0, 0.9999999999976694 ] ]
    Measured Statistics: {meanExponent=-4.7830642341759385E-14, negative=0, min=0.9999999999976694, max=0.9999999999976694, mean=0.3333333333332966, count=9.0, positive=3, stdDev=0.4714045207909798, zeros=6}
    Feedback Error: [ [ 2.1103119252074976E-12, 0.0, 0.0 ], [ 0.0, -1.1013412404281553E-13, 0.0 ], [ 0.0, 0.0, -2.3305801732931286E-12 ] ]
    Error Statistics: {meanExponent=-12.088755799140722, negative=2, min=-2.3305801732931286E-12, max=-2.3305801732931286E-12, mean=-3.671137468093851E-14, count=9.0, positive=1, stdDev=1.0480150744155462E-12, zeros=6}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 5.0567e-13 +- 9.1868e-13 [0.0000e+00 - 2.3306e-12] (18#)
    relativeTol: 7.5850e-13 +- 4.9943e-13 [5.5067e-14 - 1.1653e-12] (6#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=5.0567e-13 +- 9.1868e-13 [0.0000e+00 - 2.3306e-12] (18#), relativeTol=7.5850e-13 +- 4.9943e-13 [5.5067e-14 - 1.1653e-12] (6#)}
```



### Performance
Code from [StandardLayerTests.java:120](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L120) executed in 0.00 seconds: 
```java
    getPerformanceTester().test(layer, permPrototype);
```
Logging: 
```
    Evaluation performance: 0.000188s +- 0.000029s [0.000158s - 0.000238s]
    Learning performance: 0.000037s +- 0.000002s [0.000035s - 0.000040s]
    
```

