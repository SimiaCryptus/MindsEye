# FullyConnectedLayer
## FullyConnectedLayerTest
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
      "class": "com.simiacryptus.mindseye.layers.java.FullyConnectedLayer",
      "id": "e2a3bda5-e7e7-4c05-aeb3-4ede00001e51",
      "isFrozen": false,
      "name": "FullyConnectedLayer/e2a3bda5-e7e7-4c05-aeb3-4ede00001e51",
      "outputDims": [
        3
      ],
      "inputDims": [
        3
      ],
      "weights": [
        [
          0.9257460951780645,
          -0.9089981104890295,
          -0.2044950637092602
        ],
        [
          0.723563711644605,
          0.6860201274226323,
          0.5764737043355546
        ],
        [
          0.8770101857754737,
          0.14353079014225628,
          -0.30125657282121515
        ]
      ]
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
    [[ 1.652, 1.232, 0.084 ]]
    --------------------
    Output: 
    [ 2.4944318975854554, -0.6444314951712441, 0.34708420637672327 ]
```



### Batch Execution
Code from [LayerTestBase.java:178](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L178) executed in 0.01 seconds: 
```java
    BatchingTester batchingTester = getBatchingTester();
    return batchingTester == null ? null : batchingTester.test(layer, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (60#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (60#)}
```



### Differential Validation
Code from [LayerTestBase.java:186](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L186) executed in 0.01 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [ 1.652, 1.232, 0.084 ]
    Inputs Statistics: {meanExponent=-0.2556999877084494, negative=0, min=0.084, max=0.084, mean=0.9893333333333333, count=3.0, positive=3, stdDev=0.6627323911068648, zeros=0}
    Output: [ 2.4944318975854554, -0.6444314951712441, 0.34708420637672327 ]
    Outputs Statistics: {meanExponent=-0.08447224633796703, negative=1, min=0.34708420637672327, max=0.34708420637672327, mean=0.732361536263645, count=3.0, positive=2, stdDev=1.3100750144656146, zeros=0}
    Feedback for input 0
    Inputs Values: [ 1.652, 1.232, 0.084 ]
    Value Statistics: {meanExponent=-0.2556999877084494, negative=0, min=0.084, max=0.084, mean=0.9893333333333333, count=3.0, positive=3, stdDev=0.6627323911068648, zeros=0}
    Implemented Feedback: [ [ 0.9257460951780645, -0.9089981104890295, -0.2044950637092602 ], [ 0.723563711644605, 0.6860201274226323, 0.5764737043355546 ], [ 0.8770101857754737, 0.14353079014225628, -0.30125657282121515 ] ]
    Implemented Statistics: {meanExponent=-0.3031981022034548, negative=3, min=-0.30125657
```
...[skipping 1903 bytes](etc/55.txt)...
```
    0000000019503, 0.0 ], [ 0.0, 0.0, 0.08400000000019503 ] ]
    Measured Statistics: {meanExponent=-0.2556999877081295, negative=0, min=0.08400000000019503, max=0.08400000000019503, mean=0.3297777777777828, count=27.0, positive=9, stdDev=0.6032507821568025, zeros=18}
    Gradient Error: [ [ 8.750777880095484E-13, 0.0, 0.0 ], [ 0.0, -1.3453682612407647E-12, 0.0 ], [ 0.0, 0.0, 3.199662756969701E-13 ], [ -1.0014211682118912E-13, 0.0, 0.0 ], [ 0.0, -1.0014211682118912E-13, 0.0 ], [ 0.0, 0.0, -1.0014211682118912E-13 ], [ 1.9502455206321656E-13, 0.0, 0.0 ], [ 0.0, 1.9502455206321656E-13, 0.0 ], [ 0.0, 0.0, 1.9502455206321656E-13 ] ]
    Error Statistics: {meanExponent=-12.61687664290228, negative=4, min=1.9502455206321656E-13, max=1.9502455206321656E-13, mean=4.9749299330309675E-15, count=27.0, positive=5, stdDev=3.232746866263066E-13, zeros=18}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 1.5964e-13 +- 2.8457e-13 [0.0000e+00 - 1.3454e-12] (36#)
    relativeTol: 3.6149e-13 +- 3.9031e-13 [4.0642e-14 - 1.1609e-12] (18#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=1.5964e-13 +- 2.8457e-13 [0.0000e+00 - 1.3454e-12] (36#), relativeTol=3.6149e-13 +- 3.9031e-13 [4.0642e-14 - 1.1609e-12] (18#)}
```



### Performance
Code from [LayerTestBase.java:192](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L192) executed in 0.01 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 0.2203 +- 0.0860 [0.1396 - 0.6355]
    Learning performance: 0.5751 +- 0.3137 [0.3163 - 2.0775]
    
```

