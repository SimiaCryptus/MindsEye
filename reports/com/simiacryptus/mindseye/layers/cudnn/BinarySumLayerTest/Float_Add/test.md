# BinarySumLayer
## Float_Add
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
      "class": "com.simiacryptus.mindseye.layers.cudnn.BinarySumLayer",
      "id": "12468f68-120c-4761-8e01-9b225466b5ac",
      "isFrozen": false,
      "name": "BinarySumLayer/12468f68-120c-4761-8e01-9b225466b5ac",
      "rightFactor": 1.0,
      "leftFactor": 1.0
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
    [[
    	[ [ 0.372 ], [ -1.676 ] ],
    	[ [ 0.84 ], [ 1.872 ] ]
    ],
    [
    	[ [ -0.652 ], [ 0.0 ] ],
    	[ [ 0.38 ], [ -1.888 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 0.0 ], [ 0.0 ] ],
    	[ [ 0.0 ], [ 0.0 ] ]
    ]
    --------------------
    Derivative: 
    [
    	[ [ 0.0 ], [ 0.0 ] ],
    	[ [ 0.0 ], [ 0.0 ] ]
    ],
    [
    	[ [ 0.0 ], [ 0.0 ] ],
    	[ [ 0.0 ], [ 0.0 ] ]
    ]
```



[GPU Log](etc/cuda.log)

### Batch Execution
Code from [StandardLayerTests.java:101](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L101) executed in 0.01 seconds: 
```java
    return getBatchingTester().test(layer, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (120#), relativeTol=0.0000e+00 +- 0.0000e+00 [Infinity - -Infinity] (0#)}
```



### Differential Validation
Code from [StandardLayerTests.java:109](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L109) executed in 0.01 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ 1.928 ], [ 0.324 ] ],
    	[ [ -0.912 ], [ -0.424 ] ]
    ],
    [
    	[ [ -0.272 ], [ 0.9 ] ],
    	[ [ 0.732 ], [ 1.308 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.15424681632635678, negative=2, min=-0.424, max=-0.424, mean=0.22900000000000004, count=4.0, positive=2, stdDev=1.0751646385554168, zeros=0},
    {meanExponent=-0.157517440369959, negative=1, min=1.308, max=1.308, mean=0.667, count=4.0, positive=3, stdDev=0.5811875772932523, zeros=0}
    Output: [
    	[ [ 0.0 ], [ 0.0 ] ],
    	[ [ 0.0 ], [ 0.0 ] ]
    ]
    Outputs Statistics: {meanExponent=NaN, negative=0, min=0.0, max=0.0, mean=0.0, count=4.0, positive=0, stdDev=0.0, zeros=4}
    Feedback for input 0
    Inputs Values: [
    	[ [ 1.928 ], [ 0.324 ] ],
    	[ [ -0.912 ], [ -0.424 ] ]
    ]
    Value Statistics: {meanExponent=-0.15424681632635678, negative=2, min=-0.424, max=-0.424, mean=0.22900000000000004, count=4.0, positive=2, stdDev=1.0751646385554168, zeros=0}
    Implemented Feedback: [ [ 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0 ] ]
    Imple
```
...[skipping 755 bytes](etc/47.txt)...
```
    , negative=1, min=1.308, max=1.308, mean=0.667, count=4.0, positive=3, stdDev=0.5811875772932523, zeros=0}
    Implemented Feedback: [ [ 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0 ] ]
    Implemented Statistics: {meanExponent=NaN, negative=0, min=0.0, max=0.0, mean=0.0, count=16.0, positive=0, stdDev=0.0, zeros=16}
    Measured Feedback: [ [ 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0 ] ]
    Measured Statistics: {meanExponent=NaN, negative=0, min=0.0, max=0.0, mean=0.0, count=16.0, positive=0, stdDev=0.0, zeros=16}
    Feedback Error: [ [ 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0 ] ]
    Error Statistics: {meanExponent=NaN, negative=0, min=0.0, max=0.0, mean=0.0, count=16.0, positive=0, stdDev=0.0, zeros=16}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (32#)
    relativeTol: 0.0000e+00 +- 0.0000e+00 [Infinity - -Infinity] (0#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (32#), relativeTol=0.0000e+00 +- 0.0000e+00 [Infinity - -Infinity] (0#)}
```



### Performance
Code from [StandardLayerTests.java:119](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L119) executed in 0.45 seconds: 
```java
    getPerformanceTester().test(layer, permPrototype);
```
Logging: 
```
    100 batches
    Input Dimensions:
    	[100, 100, 1]
    	[100, 100, 1]
    Performance:
    	Evaluation performance: 0.015454s +- 0.000830s [0.014438s - 0.016976s]
    	Learning performance: 0.058757s +- 0.063221s [0.026422s - 0.185196s]
    
```

