# AvgReducerLayer
## AvgReducerLayerTest
### Json Serialization
Code from [LayerTestBase.java:83](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L83) executed in 0.00 seconds: 
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
      "id": "a864e734-2f23-44db-97c1-504000002b80",
      "isFrozen": false,
      "name": "AvgReducerLayer/a864e734-2f23-44db-97c1-504000002b80"
    }
```



### Example Input/Output Pair
Code from [LayerTestBase.java:120](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L120) executed in 0.00 seconds: 
```java
    SimpleEval eval = SimpleEval.run(layer, inputPrototype);
    return String.format("--------------------\nInput: \n[%s]\n--------------------\nOutput: \n%s",
      Arrays.stream(inputPrototype).map(t->t.prettyPrint()).reduce((a,b)->a+",\n"+b).get(),
      eval.getOutput().prettyPrint());
```

Returns: 

```
    --------------------
    Input: 
    [[ -0.868, -1.728, 0.24 ]]
    --------------------
    Output: 
    [ -0.7853333333333333 ]
```



### Batch Execution
Code from [LayerTestBase.java:138](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L138) executed in 0.00 seconds: 
```java
    BatchingTester batchingTester = getBatchingTester();
    return batchingTester==null?null:batchingTester.test(layer, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (40#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (40#)}
```



### Differential Validation
Code from [LayerTestBase.java:144](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.00 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [ -0.868, -1.728, 0.24 ]
    Inputs Statistics: {meanExponent=-0.14790843165634251, negative=2, min=0.24, max=0.24, mean=-0.7853333333333333, count=3.0, positive=1, stdDev=0.8055562605360899, zeros=0}
    Output: [ -0.7853333333333333 ]
    Outputs Statistics: {meanExponent=-0.10494596860459841, negative=1, min=-0.7853333333333333, max=-0.7853333333333333, mean=-0.7853333333333333, count=1.0, positive=0, stdDev=0.0, zeros=0}
    Feedback for input 0
    Inputs Values: [ -0.868, -1.728, 0.24 ]
    Value Statistics: {meanExponent=-0.14790843165634251, negative=2, min=0.24, max=0.24, mean=-0.7853333333333333, count=3.0, positive=1, stdDev=0.8055562605360899, zeros=0}
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
Code from [LayerTestBase.java:149](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L149) executed in 0.00 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 0.2156 +- 0.1163 [0.1368 - 0.7894]
    Learning performance: 0.0045 +- 0.0039 [0.0028 - 0.0342]
    
```

