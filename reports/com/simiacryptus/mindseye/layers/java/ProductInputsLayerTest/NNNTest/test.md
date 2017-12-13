# ProductInputsLayer
## NNNTest
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
      "id": "fce7289a-a828-415d-adf8-e9b78b23af63",
      "isFrozen": false,
      "name": "ProductInputsLayer/fce7289a-a828-415d-adf8-e9b78b23af63"
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
    [[ 1.048, -1.96, -1.548 ],
    [ 1.304, -1.612, -0.728 ],
    [ 1.028, -0.356, -1.128 ]]
    --------------------
    Output: 
    [ 1.4048565760000002, -1.12478912, -1.2711928319999999 ]
    --------------------
    Derivative: 
    [ 1.3405120000000001, 0.573872, 0.8211839999999999 ],
    [ 1.077344, 0.6977599999999999, 1.746144 ],
    [ 1.366592, 3.15952, 1.126944 ]
```



### Batch Execution
Code from [StandardLayerTests.java:102](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L102) executed in 0.00 seconds: 
```java
    return getBatchingTester().test(layer, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (120#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (120#)}
```



### Differential Validation
Code from [StandardLayerTests.java:110](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L110) executed in 0.00 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [ 1.172, 0.428, -0.436 ],
    [ 0.176, -1.796, -1.268 ],
    [ -1.26, -1.444, 0.792 ]
    Inputs Statistics: {meanExponent=-0.22004737667872334, negative=1, min=-0.436, max=-0.436, mean=0.38799999999999996, count=3.0, positive=2, stdDev=0.6570722943481943, zeros=0},
    {meanExponent=-0.13235391543628355, negative=2, min=-1.268, max=-1.268, mean=-0.9626666666666667, count=3.0, positive=1, stdDev=0.8335135805065739, zeros=0},
    {meanExponent=0.05288763998022556, negative=2, min=0.792, max=0.792, mean=-0.6373333333333332, count=3.0, positive=1, stdDev=1.013478936907697, zeros=0}
    Output: [ -0.25990272, 1.109985472, 0.437855616 ]
    Outputs Statistics: {meanExponent=-0.2995136521347814, negative=1, min=0.437855616, max=0.437855616, mean=0.42931278933333333, count=3.0, positive=2, stdDev=0.5592871352956664, zeros=0}
    Feedback for input 0
    Inputs Values: [ 1.172, 0.428, -0.436 ]
    Value Statistics: {meanExponent=-0.22004737667872334, negative=1, min=-0.436, max=-0.436, mean=0.38799999999999996, count=3.0, positive=2, stdDev=0
```
...[skipping 2593 bytes](etc/95.txt)...
```
     {meanExponent=-0.352401292115007, negative=1, min=0.552848, max=0.552848, mean=-0.0010631111111111134, count=9.0, positive=2, stdDev=0.32301704781032653, zeros=6}
    Measured Feedback: [ [ 0.20627200000011836, 0.0, 0.0 ], [ 0.0, -0.7686879999990737, 0.0 ], [ 0.0, 0.0, 0.5528480000005498 ] ]
    Measured Statistics: {meanExponent=-0.35240129211495436, negative=1, min=0.5528480000005498, max=0.5528480000005498, mean=-0.0010631111109339465, count=9.0, positive=2, stdDev=0.32301704781019513, zeros=6}
    Feedback Error: [ [ 1.1837753000065732E-13, 0.0, 0.0 ], [ 0.0, 9.263700917472306E-13, 0.0 ], [ 0.0, 0.0, 5.497824417943775E-13 ] ]
    Error Statistics: {meanExponent=-12.406585111655632, negative=0, min=5.497824417943775E-13, max=5.497824417943775E-13, mean=1.7717000706025173E-13, count=9.0, positive=3, stdDev=3.1480728093772795E-13, zeros=6}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 2.0314e-13 +- 4.1718e-13 [0.0000e+00 - 1.5450e-12] (27#)
    relativeTol: 4.2464e-13 +- 3.2781e-13 [4.2027e-14 - 1.2499e-12] (9#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=2.0314e-13 +- 4.1718e-13 [0.0000e+00 - 1.5450e-12] (27#), relativeTol=4.2464e-13 +- 3.2781e-13 [4.2027e-14 - 1.2499e-12] (9#)}
```



### Performance
Code from [StandardLayerTests.java:120](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L120) executed in 0.00 seconds: 
```java
    getPerformanceTester().test(layer, permPrototype);
```
Logging: 
```
    Evaluation performance: 0.000236s +- 0.000013s [0.000223s - 0.000260s]
    Learning performance: 0.000045s +- 0.000004s [0.000042s - 0.000053s]
    
```

