# ImgBandSelectLayer
## ImgBandSelectLayerTest
### Json Serialization
Code from [StandardLayerTests.java:68](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L68) executed in 0.00 seconds: 
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
      "class": "com.simiacryptus.mindseye.layers.java.ImgBandSelectLayer",
      "id": "9dc9b2b6-5008-43f3-9641-44ef9a26011e",
      "isFrozen": false,
      "name": "ImgBandSelectLayer/9dc9b2b6-5008-43f3-9641-44ef9a26011e",
      "bands": [
        0,
        2
      ]
    }
```



### Example Input/Output Pair
Code from [StandardLayerTests.java:152](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L152) executed in 0.00 seconds: 
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
    	[ [ -1.052, 1.72, -1.88 ], [ 1.664, -0.924, 1.816 ] ],
    	[ [ 1.752, -1.576, -1.644 ], [ 0.06, -1.044, 0.864 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ -1.052, -1.88 ], [ 1.664, 1.816 ] ],
    	[ [ 1.752, -1.644 ], [ 0.06, 0.864 ] ]
    ]
    --------------------
    Derivative: 
    [
    	[ [ 1.0, 0.0, 1.0 ], [ 1.0, 0.0, 1.0 ] ],
    	[ [ 1.0, 0.0, 1.0 ], [ 1.0, 0.0, 1.0 ] ]
    ]
```



### Batch Execution
Code from [StandardLayerTests.java:101](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L101) executed in 0.00 seconds: 
```java
    return getBatchingTester().test(layer, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (200#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (160#)}
```



### Differential Validation
Code from [StandardLayerTests.java:109](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L109) executed in 0.00 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ 1.292, 1.532, -1.624 ], [ -1.352, -1.344, 1.908 ] ],
    	[ [ 1.544, -0.704, 0.72 ], [ 1.448, -1.484, 0.456 ] ]
    ]
    Inputs Statistics: {meanExponent=0.07764844929949803, negative=5, min=0.456, max=0.456, mean=0.19933333333333336, count=12.0, positive=7, stdDev=1.3340543050749056, zeros=0}
    Output: [
    	[ [ 1.292, -1.624 ], [ -1.352, 1.908 ] ],
    	[ [ 1.544, 0.72 ], [ 1.448, 0.456 ] ]
    ]
    Outputs Statistics: {meanExponent=0.07488959968680803, negative=2, min=0.456, max=0.456, mean=0.549, count=8.0, positive=6, stdDev=1.253533804889202, zeros=0}
    Feedback for input 0
    Inputs Values: [
    	[ [ 1.292, 1.532, -1.624 ], [ -1.352, -1.344, 1.908 ] ],
    	[ [ 1.544, -0.704, 0.72 ], [ 1.448, -1.484, 0.456 ] ]
    ]
    Value Statistics: {meanExponent=0.07764844929949803, negative=5, min=0.456, max=0.456, mean=0.19933333333333336, count=12.0, positive=7, stdDev=1.3340543050749056, zeros=0}
    Implemented Feedback: [ [ 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 1.0, 0.0, 0.0, 
```
...[skipping 885 bytes](etc/115.txt)...
```
    30642341045674E-14, negative=0, min=0.9999999999998899, max=0.9999999999998899, mean=0.08333333333332416, count=96.0, positive=8, stdDev=0.2763853991962529, zeros=88}
    Feedback Error: [ [ -1.1013412404281553E-13, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, -1.1013412404281553E-13, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, -1.1013412404281553E-13, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, -1.1013412404281553E-13, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], ... ]
    Error Statistics: {meanExponent=-12.958078098036824, negative=8, min=-1.1013412404281553E-13, max=-1.1013412404281553E-13, mean=-9.177843670234628E-15, count=96.0, positive=0, stdDev=3.0439463838706555E-14, zeros=88}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 9.1778e-15 +- 3.0439e-14 [0.0000e+00 - 1.1013e-13] (96#)
    relativeTol: 5.5067e-14 +- 0.0000e+00 [5.5067e-14 - 5.5067e-14] (8#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=9.1778e-15 +- 3.0439e-14 [0.0000e+00 - 1.1013e-13] (96#), relativeTol=5.5067e-14 +- 0.0000e+00 [5.5067e-14 - 5.5067e-14] (8#)}
```



### Performance
Code from [StandardLayerTests.java:119](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L119) executed in 0.00 seconds: 
```java
    getPerformanceTester().test(layer, permPrototype);
```
Logging: 
```
    100 batches
    Input Dimensions:
    	[2, 2, 3]
    Performance:
    	Evaluation performance: 0.000200s +- 0.000037s [0.000166s - 0.000267s]
    	Learning performance: 0.000057s +- 0.000011s [0.000049s - 0.000079s]
    
```

