# ProductLayer
## Double
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
      "class": "com.simiacryptus.mindseye.layers.cudnn.ProductLayer",
      "id": "3f2bad4a-634d-4f46-8d3f-cf9490ef114d",
      "isFrozen": false,
      "name": "ProductLayer/3f2bad4a-634d-4f46-8d3f-cf9490ef114d"
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
    	[ [ -0.016 ], [ -0.792 ] ],
    	[ [ 0.432 ], [ 1.06 ] ]
    ],
    [
    	[ [ 0.984 ], [ -0.712 ] ],
    	[ [ -1.964 ], [ -1.008 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ -0.015744 ], [ 0.563904 ] ],
    	[ [ -0.848448 ], [ -1.06848 ] ]
    ]
    --------------------
    Derivative: 
    [
    	[ [ 0.984 ], [ -0.712 ] ],
    	[ [ -1.964 ], [ -1.008 ] ]
    ],
    [
    	[ [ -0.016 ], [ -0.792 ] ],
    	[ [ 0.432 ], [ 1.06 ] ]
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
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (120#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (120#)}
```



### Differential Validation
Code from [StandardLayerTests.java:109](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L109) executed in 0.01 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ -1.464 ], [ 0.84 ] ],
    	[ [ -0.676 ], [ 1.256 ] ]
    ],
    [
    	[ [ 1.144 ], [ -1.444 ] ],
    	[ [ 0.68 ], [ 0.636 ] ]
    ]
    Inputs Statistics: {meanExponent=0.004689174531766992, negative=2, min=1.256, max=1.256, mean=-0.011000000000000065, count=4.0, positive=2, stdDev=1.1048669603169423, zeros=0},
    {meanExponent=-0.03651018848868104, negative=1, min=0.636, max=0.636, mean=0.254, count=4.0, positive=3, stdDev=1.000337942897299, zeros=0}
    Output: [
    	[ [ -1.6748159999999999 ], [ -1.2129599999999998 ] ],
    	[ [ -0.4596800000000001 ], [ 0.798816 ] ]
    ]
    Outputs Statistics: {meanExponent=-0.03182101395691404, negative=3, min=0.798816, max=0.798816, mean=-0.63716, count=4.0, positive=1, stdDev=0.9356550963512142, zeros=0}
    Feedback for input 0
    Inputs Values: [
    	[ [ -1.464 ], [ 0.84 ] ],
    	[ [ -0.676 ], [ 1.256 ] ]
    ]
    Value Statistics: {meanExponent=0.004689174531766992, negative=2, min=1.256, max=1.256, mean=-0.011000000000000065, count=4.0, positive=2, stdDev=1.1048669603169423, zeros=0}
    Implemented Feedback: [ [ 1
```
...[skipping 1566 bytes](etc/79.txt)...
```
    , count=16.0, positive=2, stdDev=0.552454013923331, zeros=12}
    Measured Feedback: [ [ -1.4639999999999098, 0.0, 0.0, 0.0 ], [ 0.0, -0.6759999999994548, 0.0, 0.0 ], [ 0.0, 0.0, 0.8399999999997299, 0.0 ], [ 0.0, 0.0, 0.0, 1.25600000000059 ] ]
    Measured Statistics: {meanExponent=0.004689174531688822, negative=2, min=1.25600000000059, max=1.25600000000059, mean=-0.0027499999999403, count=16.0, positive=2, stdDev=0.5524540139233328, zeros=12}
    Feedback Error: [ [ 9.015010959956271E-14, 0.0, 0.0, 0.0 ], [ 0.0, 5.452305273934144E-13, 0.0, 0.0 ], [ 0.0, 0.0, -2.701172618913006E-13, 0.0 ], [ 0.0, 0.0, 0.0, 5.899725152858082E-13 ] ]
    Error Statistics: {meanExponent=-12.52651736459654, negative=1, min=5.899725152858082E-13, max=5.899725152858082E-13, mean=5.970224314921779E-14, count=16.0, positive=3, stdDev=2.045428666615597E-13, zeros=12}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 1.2846e-13 +- 3.1006e-13 [0.0000e+00 - 1.5552e-12] (32#)
    relativeTol: 2.5651e-13 +- 1.6551e-13 [3.0789e-14 - 5.3850e-13] (8#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=1.2846e-13 +- 3.1006e-13 [0.0000e+00 - 1.5552e-12] (32#), relativeTol=2.5651e-13 +- 1.6551e-13 [3.0789e-14 - 5.3850e-13] (8#)}
```



### Performance
Code from [StandardLayerTests.java:119](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L119) executed in 0.01 seconds: 
```java
    getPerformanceTester().test(layer, permPrototype);
```
Logging: 
```
    100 batches
    Input Dimensions:
    	[2, 2, 1]
    	[2, 2, 1]
    Performance:
    	Evaluation performance: 0.000318s +- 0.000062s [0.000274s - 0.000441s]
    	Learning performance: 0.000243s +- 0.000032s [0.000219s - 0.000303s]
    
```

