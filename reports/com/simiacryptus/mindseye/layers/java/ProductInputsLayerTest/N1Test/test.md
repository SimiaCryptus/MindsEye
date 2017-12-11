# ProductInputsLayer
## N1Test
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
      "id": "e2a3bda5-e7e7-4c05-aeb3-4ede00001f25",
      "isFrozen": false,
      "name": "ProductInputsLayer/e2a3bda5-e7e7-4c05-aeb3-4ede00001f25"
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
    [[ -1.132, -0.616, 1.152 ],
    [ -0.852 ]]
    --------------------
    Output: 
    [ 0.9644639999999999, 0.524832, -0.9815039999999999 ]
```



### Batch Execution
Code from [LayerTestBase.java:178](../../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L178) executed in 0.00 seconds: 
```java
    BatchingTester batchingTester = getBatchingTester();
    return batchingTester == null ? null : batchingTester.test(layer, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (70#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (68#)}
```



### Differential Validation
Code from [LayerTestBase.java:186](../../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L186) executed in 0.00 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [ -1.132, -0.616, 1.152 ],
    [ -0.852 ]
    Inputs Statistics: {meanExponent=-0.031706793965376255, negative=2, min=1.152, max=1.152, mean=-0.19866666666666663, count=3.0, positive=1, stdDev=0.9780215857649677, zeros=0},
    {meanExponent=-0.0695604052332999, negative=1, min=-0.852, max=-0.852, mean=-0.852, count=1.0, positive=0, stdDev=0.0, zeros=0}
    Output: [ 0.9644639999999999, 0.524832, -0.9815039999999999 ]
    Outputs Statistics: {meanExponent=-0.10126719919867615, negative=1, min=-0.9815039999999999, max=-0.9815039999999999, mean=0.169264, count=3.0, positive=2, stdDev=0.8332743910717524, zeros=0}
    Feedback for input 0
    Inputs Values: [ -1.132, -0.616, 1.152 ]
    Value Statistics: {meanExponent=-0.031706793965376255, negative=2, min=1.152, max=1.152, mean=-0.19866666666666663, count=3.0, positive=1, stdDev=0.9780215857649677, zeros=0}
    Implemented Feedback: [ [ -0.852, 0.0, 0.0 ], [ 0.0, -0.852, 0.0 ], [ 0.0, 0.0, -0.852 ] ]
    Implemented Statistics: {meanExponent=-0.0695604052332999, negative=3, min=-0.852, 
```
...[skipping 966 bytes](etc/78.txt)...
```
    eros=0}
    Implemented Feedback: [ [ -1.132, -0.616, 1.152 ] ]
    Implemented Statistics: {meanExponent=-0.031706793965376255, negative=2, min=1.152, max=1.152, mean=-0.19866666666666663, count=3.0, positive=1, stdDev=0.9780215857649677, zeros=0}
    Measured Feedback: [ [ -1.1319999999992447, -0.6159999999999499, 1.1519999999998198 ] ]
    Measured Statistics: {meanExponent=-0.03170679396550722, negative=2, min=1.1519999999998198, max=1.1519999999998198, mean=-0.19866666666645827, count=3.0, positive=1, stdDev=0.9780215857646375, zeros=0}
    Feedback Error: [ [ 7.551737013500315E-13, 5.007105841059456E-14, -1.800781745942004E-13 ] ]
    Error Statistics: {meanExponent=-12.72230176374107, negative=1, min=-1.800781745942004E-13, max=-1.800781745942004E-13, mean=2.0838886172214188E-13, count=3.0, positive=2, stdDev=3.978881023590649E-13, zeros=0}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 1.0085e-13 +- 2.0413e-13 [0.0000e+00 - 7.5517e-13] (12#)
    relativeTol: 9.7383e-14 +- 1.0640e-13 [4.0642e-14 - 3.3356e-13] (6#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=1.0085e-13 +- 2.0413e-13 [0.0000e+00 - 7.5517e-13] (12#), relativeTol=9.7383e-14 +- 1.0640e-13 [4.0642e-14 - 3.3356e-13] (6#)}
```



### Performance
Code from [LayerTestBase.java:192](../../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L192) executed in 0.00 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 0.2147 +- 0.0649 [0.1653 - 0.5814]
    Learning performance: 0.0193 +- 0.0159 [0.0114 - 0.1197]
    
```

