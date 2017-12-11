# ProductInputsLayer
## NNTest
### Json Serialization
Code from [LayerTestBase.java:121](../../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L121) executed in 0.00 seconds: 
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
      "class": "com.simiacryptus.mindseye.layers.java.ProductInputsLayer",
      "id": "e2a3bda5-e7e7-4c05-aeb3-4ede00001f36",
      "isFrozen": false,
      "name": "ProductInputsLayer/e2a3bda5-e7e7-4c05-aeb3-4ede00001f36"
    }
```



### Example Input/Output Pair
Code from [LayerTestBase.java:159](../../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L159) executed in 0.00 seconds: 
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
    [[ -1.68, -1.428, 0.56 ],
    [ -0.04, -0.42, 0.6 ]]
    --------------------
    Output: 
    [ 0.0672, 0.59976, 0.336 ]
```



### Batch Execution
Code from [LayerTestBase.java:178](../../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L178) executed in 0.00 seconds: 
```java
    BatchingTester batchingTester = getBatchingTester();
    return batchingTester == null ? null : batchingTester.test(layer, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (90#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (90#)}
```



### Differential Validation
Code from [LayerTestBase.java:186](../../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L186) executed in 0.00 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [ -1.68, -1.428, 0.56 ],
    [ -0.04, -0.42, 0.6 ]
    Inputs Statistics: {meanExponent=0.042741838724072966, negative=2, min=0.56, max=0.56, mean=-0.8493333333333332, count=3.0, positive=1, stdDev=1.001845408345131, zeros=0},
    {meanExponent=-0.6655131559634978, negative=2, min=0.6, max=0.6, mean=0.04666666666666667, count=3.0, positive=1, stdDev=0.42089850980438925, zeros=0}
    Output: [ 0.0672, 0.59976, 0.336 ]
    Outputs Statistics: {meanExponent=-0.6227713172394249, negative=0, min=0.336, max=0.336, mean=0.33432, count=3.0, positive=3, stdDev=0.21741995492594507, zeros=0}
    Feedback for input 0
    Inputs Values: [ -1.68, -1.428, 0.56 ]
    Value Statistics: {meanExponent=0.042741838724072966, negative=2, min=0.56, max=0.56, mean=-0.8493333333333332, count=3.0, positive=1, stdDev=1.001845408345131, zeros=0}
    Implemented Feedback: [ [ -0.04, 0.0, 0.0 ], [ 0.0, -0.42, 0.0 ], [ 0.0, 0.0, 0.6 ] ]
    Implemented Statistics: {meanExponent=-0.6655131559634978, negative=2, min=0.6, max=0.6, mean=0.015555555555555557, count=9.
```
...[skipping 1052 bytes](etc/80.txt)...
```
    ed Statistics: {meanExponent=0.042741838724072966, negative=2, min=0.56, max=0.56, mean=-0.2831111111111111, count=9.0, positive=1, stdDev=0.7034689354974223, zeros=6}
    Measured Feedback: [ [ -1.6800000000000148, 0.0, 0.0 ], [ 0.0, -1.4279999999999848, 0.0 ], [ 0.0, 0.0, 0.5600000000000049 ] ]
    Measured Statistics: {meanExponent=0.042741838724073965, negative=2, min=0.5600000000000049, max=0.5600000000000049, mean=-0.28311111111111054, count=9.0, positive=1, stdDev=0.7034689354974234, zeros=6}
    Feedback Error: [ [ -1.4876988529977098E-14, 0.0, 0.0 ], [ 0.0, 1.509903313490213E-14, 0.0 ], [ 0.0, 0.0, 4.884981308350689E-15 ] ]
    Error Statistics: {meanExponent=-13.986557642450599, negative=1, min=4.884981308350689E-15, max=4.884981308350689E-15, mean=5.674473236973022E-16, count=9.0, positive=2, stdDev=7.228574959656349E-15, zeros=6}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 1.7425e-14 +- 3.7288e-14 [0.0000e+00 - 1.3506e-13] (18#)
    relativeTol: 2.4117e-13 +- 4.4774e-13 [4.3616e-15 - 1.2347e-12] (6#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=1.7425e-14 +- 3.7288e-14 [0.0000e+00 - 1.3506e-13] (18#), relativeTol=2.4117e-13 +- 4.4774e-13 [4.3616e-15 - 1.2347e-12] (6#)}
```



### Performance
Code from [LayerTestBase.java:192](../../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L192) executed in 0.00 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 0.1616 +- 0.0639 [0.1140 - 0.6612]
    Learning performance: 0.0109 +- 0.0190 [0.0057 - 0.1767]
    
```

