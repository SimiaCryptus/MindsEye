# ProductInputsLayer
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
      "class": "com.simiacryptus.mindseye.layers.java.ProductInputsLayer",
      "id": "8c78bdcb-d6f1-491e-a684-cc5a36cbb609",
      "isFrozen": false,
      "name": "ProductInputsLayer/8c78bdcb-d6f1-491e-a684-cc5a36cbb609"
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
    [[ 1.368, 0.056, 0.008 ],
    [ -1.092 ]]
    --------------------
    Output: 
    [ -1.4938560000000003, -0.061152000000000005, -0.008736 ]
    --------------------
    Derivative: 
    [ -1.092, -1.092, -1.092 ],
    [ 1.4320000000000002 ]
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
    Inputs: [ 0.124, -1.232, -0.192 ],
    [ -1.572 ]
    Inputs Statistics: {meanExponent=-0.5108887927686029, negative=2, min=-0.192, max=-0.192, mean=-0.43333333333333335, count=3.0, positive=1, stdDev=0.5792899868709013, zeros=0},
    {meanExponent=0.1964525417033891, negative=1, min=-1.572, max=-1.572, mean=-1.572, count=1.0, positive=0, stdDev=0.0, zeros=0}
    Output: [ -0.19492800000000002, 1.936704, 0.30182400000000004 ]
    Outputs Statistics: {meanExponent=-0.3144362510652137, negative=1, min=0.30182400000000004, max=0.30182400000000004, mean=0.6812, count=3.0, positive=2, stdDev=0.9106438593610566, zeros=0}
    Feedback for input 0
    Inputs Values: [ 0.124, -1.232, -0.192 ]
    Value Statistics: {meanExponent=-0.5108887927686029, negative=2, min=-0.192, max=-0.192, mean=-0.43333333333333335, count=3.0, positive=1, stdDev=0.5792899868709013, zeros=0}
    Implemented Feedback: [ [ -1.572, 0.0, 0.0 ], [ 0.0, -1.572, 0.0 ], [ 0.0, 0.0, -1.572 ] ]
    Implemented Statistics: {meanExponent=0.1964525417033891, negative=3, min=-1.572, max
```
...[skipping 951 bytes](etc/94.txt)...
```
    }
    Implemented Feedback: [ [ 0.124, -1.232, -0.192 ] ]
    Implemented Statistics: {meanExponent=-0.5108887927686029, negative=2, min=-0.192, max=-0.192, mean=-0.43333333333333335, count=3.0, positive=1, stdDev=0.5792899868709013, zeros=0}
    Measured Feedback: [ [ 0.12400000000023503, -1.2319999999998998, -0.19199999999996997 ] ]
    Measured Statistics: {meanExponent=-0.5108887927683629, negative=2, min=-0.19199999999996997, max=-0.19199999999996997, mean=-0.4333333333332116, count=3.0, positive=1, stdDev=0.5792899868709346, zeros=0}
    Feedback Error: [ [ 2.3503421431314564E-13, 1.0014211682118912E-13, 3.0031532816110484E-14 ] ]
    Error Statistics: {meanExponent=-13.050224881856357, negative=0, min=3.0031532816110484E-14, max=3.0031532816110484E-14, mean=1.2173595465014841E-13, count=3.0, positive=3, stdDev=8.50734789437196E-14, zeros=0}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 1.7236e-13 +- 3.8738e-13 [0.0000e+00 - 1.4255e-12] (12#)
    relativeTol: 2.6804e-13 +- 3.3823e-13 [1.2006e-14 - 9.4772e-13] (6#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=1.7236e-13 +- 3.8738e-13 [0.0000e+00 - 1.4255e-12] (12#), relativeTol=2.6804e-13 +- 3.3823e-13 [1.2006e-14 - 9.4772e-13] (6#)}
```



### Performance
Code from [StandardLayerTests.java:120](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L120) executed in 0.00 seconds: 
```java
    getPerformanceTester().test(layer, permPrototype);
```
Logging: 
```
    Evaluation performance: 0.000134s +- 0.000024s [0.000105s - 0.000175s]
    Learning performance: 0.000096s +- 0.000051s [0.000052s - 0.000183s]
    
```

