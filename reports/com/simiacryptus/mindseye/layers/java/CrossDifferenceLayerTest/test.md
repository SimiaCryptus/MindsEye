# CrossDifferenceLayer
## CrossDifferenceLayerTest
### Json Serialization
Code from [LayerTestBase.java:121](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L121) executed in 0.00 seconds: 
```java
    JsonObject json = layer.getJson();
    NNLayer echo = NNLayer.fromJson(json);
    assert (echo != null) : "Failed to deserialize";
    assert (layer != echo) : "Serialization did not copy";
    Assert.assertEquals("Serialization not equal", layer, echo);
    return new GsonBuilder().setPrettyPrinting().create().toJson(json);
```

Returns: 

```
    {
      "class": "com.simiacryptus.mindseye.layers.java.CrossDifferenceLayer",
      "id": "e2a3bda5-e7e7-4c05-aeb3-4ede00001e2b",
      "isFrozen": false,
      "name": "CrossDifferenceLayer/e2a3bda5-e7e7-4c05-aeb3-4ede00001e2b"
    }
```



### Example Input/Output Pair
Code from [LayerTestBase.java:159](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L159) executed in 0.00 seconds: 
```java
    SimpleEval eval = SimpleEval.run(layer, inputPrototype);
    return String.format("--------------------\nInput: \n[%s]\n--------------------\nOutput: \n%s",
      Arrays.stream(inputPrototype).map(t -> t.prettyPrint()).reduce((a, b) -> a + ",\n" + b).get(),
      eval.getOutput().prettyPrint());
```

Returns: 

```
    --------------------
    Input: 
    [[ 1.268, -1.216, 1.348, -0.368 ]]
    --------------------
    Output: 
    [ 2.484, -0.08000000000000007, 1.6360000000000001, -2.564, -0.848, 1.7160000000000002 ]
```



### Batch Execution
Code from [LayerTestBase.java:178](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L178) executed in 0.00 seconds: 
```java
    BatchingTester batchingTester = getBatchingTester();
    return batchingTester == null ? null : batchingTester.test(layer, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (100#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (100#)}
```



### Differential Validation
Code from [LayerTestBase.java:186](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L186) executed in 0.00 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [ 1.268, -1.216, 1.348, -0.368 ]
    Inputs Statistics: {meanExponent=-0.029102365161187832, negative=2, min=-0.368, max=-0.368, mean=0.258, count=4.0, positive=2, stdDev=1.0923314515292508, zeros=0}
    Output: [ 2.484, -0.08000000000000007, 1.6360000000000001, -2.564, -0.848, 1.7160000000000002 ]
    Outputs Statistics: {meanExponent=0.013976005741328526, negative=3, min=1.7160000000000002, max=1.7160000000000002, mean=0.3906666666666667, count=6.0, positive=3, stdDev=1.7404637951483573, zeros=0}
    Feedback for input 0
    Inputs Values: [ 1.268, -1.216, 1.348, -0.368 ]
    Value Statistics: {meanExponent=-0.029102365161187832, negative=2, min=-0.368, max=-0.368, mean=0.258, count=4.0, positive=2, stdDev=1.0923314515292508, zeros=0}
    Implemented Feedback: [ [ 1.0, 1.0, 1.0, 0.0, 0.0, 0.0 ], [ -1.0, 0.0, 0.0, 1.0, 1.0, 0.0 ], [ 0.0, -1.0, 0.0, -1.0, 0.0, 1.0 ], [ 0.0, 0.0, -1.0, 0.0, -1.0, -1.0 ] ]
    Implemented Statistics: {meanExponent=0.0, negative=6, min=-1.0, max=-1.0, mean=0.0, count=24.0, positive=6, stdDev=0.
```
...[skipping 357 bytes](etc/50.txt)...
```
    00000021103 ] ]
    Measured Statistics: {meanExponent=-3.6927313119289045E-13, negative=6, min=-1.0000000000021103, max=-1.0000000000021103, mean=-3.7007434154171886E-13, count=24.0, positive=6, stdDev=0.7071067811859463, zeros=12}
    Feedback Error: [ [ -2.3305801732931286E-12, -1.1013412404281553E-13, -2.3305801732931286E-12, 0.0, 0.0, 0.0 ], [ 2.3305801732931286E-12, 0.0, 0.0, -2.3305801732931286E-12, -1.1013412404281553E-13, 0.0 ], [ 0.0, 1.1013412404281553E-13, 0.0, 2.3305801732931286E-12, 0.0, -2.3305801732931286E-12 ], [ 0.0, 0.0, -2.1103119252074976E-12, 0.0, 1.1013412404281553E-13, -2.1103119252074976E-12 ] ]
    Error Statistics: {meanExponent=-12.081569566741642, negative=8, min=-2.1103119252074976E-12, max=-2.1103119252074976E-12, mean=-3.7007434154171886E-13, count=24.0, positive=4, stdDev=1.2625710239250232E-12, zeros=12}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 7.7686e-13 +- 1.0619e-12 [0.0000e+00 - 2.3306e-12] (24#)
    relativeTol: 7.7686e-13 +- 5.1187e-13 [5.5067e-14 - 1.1653e-12] (12#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=7.7686e-13 +- 1.0619e-12 [0.0000e+00 - 2.3306e-12] (24#), relativeTol=7.7686e-13 +- 5.1187e-13 [5.5067e-14 - 1.1653e-12] (12#)}
```



### Performance
Code from [LayerTestBase.java:192](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L192) executed in 0.00 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 0.1914 +- 0.1139 [0.1169 - 0.8407]
    Learning performance: 0.0026 +- 0.0033 [0.0000 - 0.0257]
    
```

