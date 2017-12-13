# ProductInputsLayer
## NNTest
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
      "class": "com.simiacryptus.mindseye.layers.java.ProductInputsLayer",
      "id": "e659f1ae-7a1f-4e14-aece-fc8f5b89f5f0",
      "isFrozen": false,
      "name": "ProductInputsLayer/e659f1ae-7a1f-4e14-aece-fc8f5b89f5f0"
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
    [[ -0.58, -1.456, 1.28 ],
    [ 1.56, -0.628, 1.748 ]]
    --------------------
    Output: 
    [ -0.9047999999999999, 0.914368, 2.23744 ]
    --------------------
    Derivative: 
    [ 1.56, -0.628, 1.748 ],
    [ -0.58, -1.456, 1.28 ]
```



### Batch Execution
Code from [StandardLayerTests.java:102](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L102) executed in 0.00 seconds: 
```java
    return getBatchingTester().test(layer, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (90#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (90#)}
```



### Differential Validation
Code from [StandardLayerTests.java:110](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L110) executed in 0.00 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [ 0.712, -1.528, 0.668 ],
    [ -1.072, -1.676, -1.564 ]
    Inputs Statistics: {meanExponent=-0.046206729882642274, negative=1, min=0.668, max=0.668, mean=-0.04933333333333334, count=3.0, positive=2, stdDev=1.0457295167595786, zeros=0},
    {meanExponent=0.14956851612494604, negative=3, min=-1.564, max=-1.564, mean=-1.4373333333333334, count=3.0, positive=0, stdDev=0.2623449806817141, zeros=0}
    Output: [ -0.763264, 2.560928, -1.0447520000000001 ]
    Outputs Statistics: {meanExponent=0.1033617862423038, negative=2, min=-1.0447520000000001, max=-1.0447520000000001, mean=0.2509706666666667, count=3.0, positive=1, stdDev=1.6374240002518863, zeros=0}
    Feedback for input 0
    Inputs Values: [ 0.712, -1.528, 0.668 ]
    Value Statistics: {meanExponent=-0.046206729882642274, negative=1, min=0.668, max=0.668, mean=-0.04933333333333334, count=3.0, positive=2, stdDev=1.0457295167595786, zeros=0}
    Implemented Feedback: [ [ -1.072, 0.0, 0.0 ], [ 0.0, -1.676, 0.0 ], [ 0.0, 0.0, -1.564 ] ]
    Implemented Statistics: {meanExponent=0.14
```
...[skipping 1161 bytes](etc/96.txt)...
```
    tistics: {meanExponent=-0.046206729882642274, negative=1, min=0.668, max=0.668, mean=-0.016444444444444446, count=9.0, positive=2, stdDev=0.604199950551896, zeros=6}
    Measured Feedback: [ [ 0.71200000000049, 0.0, 0.0 ], [ 0.0, -1.5279999999995297, 0.0 ], [ 0.0, 0.0, 0.6679999999992248 ] ]
    Measured Statistics: {meanExponent=-0.046206729882755204, negative=1, min=0.6679999999992248, max=0.6679999999992248, mean=-0.016444444444423882, count=9.0, positive=2, stdDev=0.6041999505517334, zeros=6}
    Feedback Error: [ [ 4.900524430695441E-13, 0.0, 0.0 ], [ 0.0, 4.702904732312163E-13, 0.0 ], [ 0.0, 0.0, -7.752687380957468E-13 ] ]
    Error Statistics: {meanExponent=-12.249312996090303, negative=1, min=-7.752687380957468E-13, max=-7.752687380957468E-13, mean=2.0563797578334845E-14, count=9.0, positive=2, stdDev=3.4295453448290215E-13, zeros=6}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 2.2922e-13 +- 4.0506e-13 [0.0000e+00 - 1.5652e-12] (18#)
    relativeTol: 3.0787e-13 +- 1.7104e-13 [1.2138e-13 - 5.8029e-13] (6#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=2.2922e-13 +- 4.0506e-13 [0.0000e+00 - 1.5652e-12] (18#), relativeTol=3.0787e-13 +- 1.7104e-13 [1.2138e-13 - 5.8029e-13] (6#)}
```



### Performance
Code from [StandardLayerTests.java:120](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L120) executed in 0.00 seconds: 
```java
    getPerformanceTester().test(layer, permPrototype);
```
Logging: 
```
    Evaluation performance: 0.000143s +- 0.000016s [0.000123s - 0.000165s]
    Learning performance: 0.000046s +- 0.000006s [0.000041s - 0.000057s]
    
```

