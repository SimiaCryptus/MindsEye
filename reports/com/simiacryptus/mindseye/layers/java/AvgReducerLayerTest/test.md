# AvgReducerLayer
## AvgReducerLayerTest
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
      "class": "com.simiacryptus.mindseye.layers.java.AvgReducerLayer",
      "id": "e2a3bda5-e7e7-4c05-aeb3-4ede00001e0b",
      "isFrozen": false,
      "name": "AvgReducerLayer/e2a3bda5-e7e7-4c05-aeb3-4ede00001e0b"
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
    [[ 1.876, -0.068, -1.152 ]]
    --------------------
    Output: 
    [ 0.21866666666666673 ]
```



### Batch Execution
Code from [LayerTestBase.java:178](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L178) executed in 0.00 seconds: 
```java
    BatchingTester batchingTester = getBatchingTester();
    return batchingTester == null ? null : batchingTester.test(layer, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (40#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (40#)}
```



### Differential Validation
Code from [LayerTestBase.java:186](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L186) executed in 0.00 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [ 1.876, -0.068, -1.152 ]
    Inputs Statistics: {meanExponent=-0.2776019247211749, negative=2, min=-1.152, max=-1.152, mean=0.21866666666666665, count=3.0, positive=1, stdDev=1.2526849386639705, zeros=0}
    Output: [ 0.21866666666666673 ]
    Outputs Statistics: {meanExponent=-0.660217415344002, negative=0, min=0.21866666666666673, max=0.21866666666666673, mean=0.21866666666666673, count=1.0, positive=1, stdDev=0.0, zeros=0}
    Feedback for input 0
    Inputs Values: [ 1.876, -0.068, -1.152 ]
    Value Statistics: {meanExponent=-0.2776019247211749, negative=2, min=-1.152, max=-1.152, mean=0.21866666666666665, count=3.0, positive=1, stdDev=1.2526849386639705, zeros=0}
    Implemented Feedback: [ [ 0.3333333333333333 ], [ 0.3333333333333333 ], [ 0.3333333333333333 ] ]
    Implemented Statistics: {meanExponent=-0.47712125471966244, negative=0, min=0.3333333333333333, max=0.3333333333333333, mean=0.3333333333333333, count=3.0, positive=3, stdDev=0.0, zeros=0}
    Measured Feedback: [ [ 0.3333333333332966 ], [ 0.3333333333332966 ], [ 0.3333333333332966 ] ]
    Measured Statistics: {meanExponent=-0.4771212547197103, negative=0, min=0.3333333333332966, max=0.3333333333332966, mean=0.3333333333332966, count=3.0, positive=3, stdDev=0.0, zeros=0}
    Feedback Error: [ [ -3.6692870963861424E-14 ], [ -3.6692870963861424E-14 ], [ -3.6692870963861424E-14 ] ]
    Error Statistics: {meanExponent=-13.435418306369344, negative=3, min=-3.6692870963861424E-14, max=-3.6692870963861424E-14, mean=-3.6692870963861424E-14, count=3.0, positive=0, stdDev=0.0, zeros=0}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 3.6693e-14 +- 0.0000e+00 [3.6693e-14 - 3.6693e-14] (3#)
    relativeTol: 5.5039e-14 +- 0.0000e+00 [5.5039e-14 - 5.5039e-14] (3#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=3.6693e-14 +- 0.0000e+00 [3.6693e-14 - 3.6693e-14] (3#), relativeTol=5.5039e-14 +- 0.0000e+00 [5.5039e-14 - 5.5039e-14] (3#)}
```



### Performance
Code from [LayerTestBase.java:192](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L192) executed in 0.00 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 0.1753 +- 0.0823 [0.1396 - 0.7866]
    Learning performance: 0.0035 +- 0.0035 [0.0000 - 0.0314]
    
```

