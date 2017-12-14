# ProductInputsLayer
## NNNTest
### Json Serialization
Code from [StandardLayerTests.java:68](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L68) executed in 0.00 seconds: 
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
      "id": "89bf1226-7d79-46a7-8450-293003088649",
      "isFrozen": false,
      "name": "ProductInputsLayer/89bf1226-7d79-46a7-8450-293003088649"
    }
```



### Example Input/Output Pair
Code from [StandardLayerTests.java:152](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L152) executed in 0.00 seconds: 
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
    [[ -0.624, -1.588, 0.36 ],
    [ 1.972, -1.484, -1.652 ],
    [ 0.12, -0.06, -1.724 ]]
    --------------------
    Output: 
    [ -0.14766336, -0.14139552, 1.02529728 ]
    --------------------
    Derivative: 
    [ 0.23664, 0.08904, 2.848048 ],
    [ -0.07488, 0.09528, -0.62064 ],
    [ -1.230528, 2.356592, -0.5947199999999999 ]
```



### Batch Execution
Code from [StandardLayerTests.java:101](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L101) executed in 0.00 seconds: 
```java
    return getBatchingTester().test(layer, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (120#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (120#)}
```



### Differential Validation
Code from [StandardLayerTests.java:109](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L109) executed in 0.00 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [ -0.62, -0.108, 0.42 ],
    [ -1.28, -0.096, 1.924 ],
    [ -0.46, -1.76, -0.372 ]
    Inputs Statistics: {meanExponent=-0.5169784215389653, negative=2, min=0.42, max=0.42, mean=-0.10266666666666667, count=3.0, positive=1, stdDev=0.42459497039989635, zeros=0},
    {meanExponent=-0.20877124320358967, negative=2, min=1.924, max=1.924, mean=0.18266666666666662, count=3.0, positive=1, stdDev=1.322786284913612, zeros=0},
    {meanExponent=-0.17372885354079284, negative=3, min=-0.372, max=-0.372, mean=-0.864, count=3.0, positive=0, stdDev=0.6345854289744342, zeros=0}
    Output: [ -0.365056, -0.01824768, -0.30060575999999994 ]
    Outputs Statistics: {meanExponent=-0.8994785182833479, negative=3, min=-0.30060575999999994, max=-0.30060575999999994, mean=-0.2279698133333333, count=3.0, positive=0, stdDev=0.15061205853987422, zeros=0}
    Feedback for input 0
    Inputs Values: [ -0.62, -0.108, 0.42 ]
    Value Statistics: {meanExponent=-0.5169784215389653, negative=2, min=0.42, max=0.42, mean=-0.10266666666666667, count=3.0, positive=1, stdD
```
...[skipping 2602 bytes](etc/137.txt)...
```
    0.7257496647425549, negative=0, min=0.8080799999999999, max=0.8080799999999999, mean=0.17911644444444444, count=9.0, positive=3, stdDev=0.33235814614663, zeros=6}
    Measured Feedback: [ [ 0.7935999999991727, 0.0, 0.0 ], [ 0.0, 0.010367999999996713, 0.0 ], [ 0.0, 0.0, 0.8080799999998778 ] ]
    Measured Statistics: {meanExponent=-0.7257496647427737, negative=0, min=0.8080799999998778, max=0.8080799999998778, mean=0.17911644444433858, count=9.0, positive=3, stdDev=0.3323581461464346, zeros=6}
    Feedback Error: [ [ -8.272271756482041E-13, 0.0, 0.0 ], [ 0.0, -3.2873009869760494E-15, 0.0 ], [ 0.0, 0.0, -1.2212453270876722E-13 ] ]
    Error Statistics: {meanExponent=-13.15957760723233, negative=3, min=-1.2212453270876722E-13, max=-1.2212453270876722E-13, mean=-1.0584877881599415E-13, count=9.0, positive=0, stdDev=2.57853173590554E-13, zeros=6}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 7.9763e-14 +- 1.8542e-13 [0.0000e+00 - 8.2723e-13] (27#)
    relativeTol: 3.4410e-13 +- 4.4404e-13 [2.4020e-14 - 1.5162e-12] (9#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=7.9763e-14 +- 1.8542e-13 [0.0000e+00 - 8.2723e-13] (27#), relativeTol=3.4410e-13 +- 4.4404e-13 [2.4020e-14 - 1.5162e-12] (9#)}
```



### Performance
Code from [StandardLayerTests.java:119](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L119) executed in 0.00 seconds: 
```java
    getPerformanceTester().test(layer, permPrototype);
```
Logging: 
```
    100 batches
    Input Dimensions:
    	[3]
    	[3]
    	[3]
    Performance:
    	Evaluation performance: 0.000227s +- 0.000013s [0.000209s - 0.000249s]
    	Learning performance: 0.000050s +- 0.000006s [0.000044s - 0.000058s]
    
```

