# SumInputsLayer
## N1Test
### Json Serialization
Code from [StandardLayerTests.java:69](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L69) executed in 0.00 seconds: 
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
      "class": "com.simiacryptus.mindseye.layers.java.SumInputsLayer",
      "id": "19231668-fc83-4d5a-81dc-71ff73b579c9",
      "isFrozen": false,
      "name": "SumInputsLayer/19231668-fc83-4d5a-81dc-71ff73b579c9"
    }
```



### Example Input/Output Pair
Code from [StandardLayerTests.java:153](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L153) executed in 0.00 seconds: 
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
    [[ -1.64, 1.56, 0.316 ],
    [ 1.112 ]]
    --------------------
    Output: 
    [ -0.5279999999999998, 2.672, 1.4280000000000002 ]
    --------------------
    Derivative: 
    [ 1.0, 1.0, 1.0 ],
    [ 3.0 ]
```



### Batch Execution
Code from [StandardLayerTests.java:102](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L102) executed in 0.00 seconds: 
```java
    return getBatchingTester().test(layer, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (70#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (70#)}
```



### Differential Validation
Code from [StandardLayerTests.java:110](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L110) executed in 0.00 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [ 1.96, -1.02, -1.352 ],
    [ 2.0 ]
    Inputs Statistics: {meanExponent=0.14394431157467025, negative=2, min=-1.352, max=-1.352, mean=-0.1373333333333334, count=3.0, positive=1, stdDev=1.4892193331481507, zeros=0},
    {meanExponent=0.3010299956639812, negative=0, min=2.0, max=2.0, mean=2.0, count=1.0, positive=1, stdDev=0.0, zeros=0}
    Output: [ 3.96, 0.98, 0.6479999999999999 ]
    Outputs Statistics: {meanExponent=0.1334987558295335, negative=0, min=0.6479999999999999, max=0.6479999999999999, mean=1.8626666666666665, count=3.0, positive=3, stdDev=1.4892193331481507, zeros=0}
    Feedback for input 0
    Inputs Values: [ 1.96, -1.02, -1.352 ]
    Value Statistics: {meanExponent=0.14394431157467025, negative=2, min=-1.352, max=-1.352, mean=-0.1373333333333334, count=3.0, positive=1, stdDev=1.4892193331481507, zeros=0}
    Implemented Feedback: [ [ 1.0, 0.0, 0.0 ], [ 0.0, 1.0, 0.0 ], [ 0.0, 0.0, 1.0 ] ]
    Implemented Statistics: {meanExponent=0.0, negative=0, min=1.0, max=1.0, mean=0.3333333333333333, count=9.0, positive=3, std
```
...[skipping 795 bytes](etc/108.txt)...
```
    0.3010299956639812, negative=0, min=2.0, max=2.0, mean=2.0, count=1.0, positive=1, stdDev=0.0, zeros=0}
    Implemented Feedback: [ [ 1.0, 1.0, 1.0 ] ]
    Implemented Statistics: {meanExponent=0.0, negative=0, min=1.0, max=1.0, mean=1.0, count=3.0, positive=3, stdDev=0.0, zeros=0}
    Measured Feedback: [ [ 1.0000000000021103, 1.0000000000021103, 1.0000000000021103 ] ]
    Measured Statistics: {meanExponent=9.16496824211277E-13, negative=0, min=1.0000000000021103, max=1.0000000000021103, mean=1.0000000000021103, count=3.0, positive=3, stdDev=0.0, zeros=0}
    Feedback Error: [ [ 2.1103119252074976E-12, 2.1103119252074976E-12, 2.1103119252074976E-12 ] ]
    Error Statistics: {meanExponent=-11.675653346889904, negative=0, min=2.1103119252074976E-12, max=2.1103119252074976E-12, mean=2.1103119252074976E-12, count=3.0, positive=3, stdDev=0.0, zeros=0}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 7.4015e-13 +- 1.0100e-12 [0.0000e+00 - 2.3306e-12] (12#)
    relativeTol: 7.4015e-13 +- 4.8599e-13 [5.5067e-14 - 1.1653e-12] (6#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=7.4015e-13 +- 1.0100e-12 [0.0000e+00 - 2.3306e-12] (12#), relativeTol=7.4015e-13 +- 4.8599e-13 [5.5067e-14 - 1.1653e-12] (6#)}
```



### Performance
Code from [StandardLayerTests.java:120](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L120) executed in 0.00 seconds: 
```java
    getPerformanceTester().test(layer, permPrototype);
```
Logging: 
```
    Evaluation performance: 0.000134s +- 0.000022s [0.000103s - 0.000169s]
    Learning performance: 0.000048s +- 0.000002s [0.000045s - 0.000049s]
    
```

