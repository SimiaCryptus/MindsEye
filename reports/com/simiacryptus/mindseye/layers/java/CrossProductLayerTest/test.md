# CrossProductLayer
## CrossProductLayerTest
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
      "class": "com.simiacryptus.mindseye.layers.java.CrossProductLayer",
      "id": "e2a3bda5-e7e7-4c05-aeb3-4ede00001e3c",
      "isFrozen": false,
      "name": "CrossProductLayer/e2a3bda5-e7e7-4c05-aeb3-4ede00001e3c"
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
    [[ -1.756, 1.3, -0.948, -0.7 ]]
    --------------------
    Output: 
    [ -2.2828, 1.664688, 1.2291999999999998, -1.2324, -0.9099999999999999, 0.6636 ]
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
    Inputs: [ -1.756, 1.3, -0.948, -0.7 ]
    Inputs Statistics: {meanExponent=0.04509356030731089, negative=3, min=-0.7, max=-0.7, mean=-0.526, count=4.0, positive=1, stdDev=1.1242259559359054, zeros=0}
    Output: [ -2.2828, 1.664688, 1.2291999999999998, -1.2324, -0.9099999999999999, 0.6636 ]
    Outputs Statistics: {meanExponent=0.09018712061462177, negative=3, min=0.6636, max=0.6636, mean=-0.14461866666666667, count=6.0, positive=3, stdDev=1.4233225205397249, zeros=0}
    Feedback for input 0
    Inputs Values: [ -1.756, 1.3, -0.948, -0.7 ]
    Value Statistics: {meanExponent=0.04509356030731089, negative=3, min=-0.7, max=-0.7, mean=-0.526, count=4.0, positive=1, stdDev=1.1242259559359054, zeros=0}
    Implemented Feedback: [ [ 1.3, -0.948, -0.7, 0.0, 0.0, 0.0 ], [ -1.756, 0.0, 0.0, -0.948, -0.7, 0.0 ], [ 0.0, -1.756, 0.0, 1.3, 0.0, -0.7 ], [ 0.0, 0.0, -1.756, 0.0, 1.3, -0.948 ] ]
    Implemented Statistics: {meanExponent=0.04509356030731091, negative=9, min=-0.948, max=-0.948, mean=-0.26300000000000007, count=24.0, positive=3, stdD
```
...[skipping 339 bytes](etc/52.txt)...
```
    99999996348, -0.948000000000615 ] ]
    Measured Statistics: {meanExponent=0.04509356030733775, negative=9, min=-0.948000000000615, max=-0.948000000000615, mean=-0.26300000000006873, count=24.0, positive=3, stdDev=0.8373237127896019, zeros=12}
    Feedback Error: [ [ -3.652633751016765E-13, -6.150635556423367E-13, -1.4499512701604544E-13, 0.0, 0.0, 0.0 ], [ 5.753175713607561E-13, 0.0, 0.0, -6.150635556423367E-13, -1.4499512701604544E-13, 0.0 ], [ 0.0, 5.753175713607561E-13, 0.0, -3.652633751016765E-13, 0.0, -1.4499512701604544E-13 ], [ 0.0, 0.0, 5.753175713607561E-13, 0.0, -3.652633751016765E-13, -6.150635556423367E-13 ] ]
    Error Statistics: {meanExponent=-12.431803208124188, negative=9, min=-6.150635556423367E-13, max=-6.150635556423367E-13, mean=-6.875056079991282E-14, count=24.0, positive=3, stdDev=3.2131006997962257E-13, zeros=12}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 2.1258e-13 +- 2.5055e-13 [0.0000e+00 - 6.1506e-13] (24#)
    relativeTol: 1.8307e-13 +- 8.4379e-14 [1.0357e-13 - 3.2440e-13] (12#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=2.1258e-13 +- 2.5055e-13 [0.0000e+00 - 6.1506e-13] (24#), relativeTol=1.8307e-13 +- 8.4379e-14 [1.0357e-13 - 3.2440e-13] (12#)}
```



### Performance
Code from [LayerTestBase.java:192](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L192) executed in 0.00 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 0.2147 +- 0.1310 [0.1482 - 1.1684]
    Learning performance: 0.0028 +- 0.0019 [0.0000 - 0.0171]
    
```

