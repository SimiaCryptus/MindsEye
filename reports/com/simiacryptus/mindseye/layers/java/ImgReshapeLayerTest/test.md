# ImgReshapeLayer
## ImgReshapeLayerTest
### Json Serialization
Code from [StandardLayerTests.java:69](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L69) executed in 0.00 seconds: 
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
      "class": "com.simiacryptus.mindseye.layers.java.ImgReshapeLayer",
      "id": "098da96b-9f5e-46fa-ad66-052a66d31e8b",
      "isFrozen": false,
      "name": "ImgReshapeLayer/098da96b-9f5e-46fa-ad66-052a66d31e8b",
      "kernelSizeX": 2,
      "kernelSizeY": 2,
      "expand": false
    }
```



### Example Input/Output Pair
Code from [StandardLayerTests.java:153](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L153) executed in 0.00 seconds: 
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
    	[ [ 0.568, 1.608, -0.3 ], [ 0.796, -1.208, -0.74 ] ],
    	[ [ 0.76, -1.628, 1.256 ], [ -1.26, 0.58, -0.132 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 0.568, 1.608, -0.3, 0.796, -1.208, -0.74, 0.76, -1.628, ... ] ]
    ]
    --------------------
    Derivative: 
    [
    	[ [ 1.0, 1.0, 1.0 ], [ 1.0, 1.0, 1.0 ] ],
    	[ [ 1.0, 1.0, 1.0 ], [ 1.0, 1.0, 1.0 ] ]
    ]
```



### Batch Execution
Code from [StandardLayerTests.java:102](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L102) executed in 0.00 seconds: 
```java
    return getBatchingTester().test(layer, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (240#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (240#)}
```



### Differential Validation
Code from [StandardLayerTests.java:110](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L110) executed in 0.00 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ 1.708, 1.208, -1.94 ], [ -1.412, -0.456, 1.964 ] ],
    	[ [ 1.932, 0.688, 1.776 ], [ 1.02, 1.504, 1.48 ] ]
    ]
    Inputs Statistics: {meanExponent=0.11945381654746186, negative=3, min=1.48, max=1.48, mean=0.7893333333333333, count=12.0, positive=9, stdDev=1.277220245515845, zeros=0}
    Output: [
    	[ [ 1.708, 1.208, -1.94, -1.412, -0.456, 1.964, 1.932, 0.688, ... ] ]
    ]
    Outputs Statistics: {meanExponent=0.11945381654746186, negative=3, min=1.48, max=1.48, mean=0.7893333333333333, count=12.0, positive=9, stdDev=1.2772202455158452, zeros=0}
    Feedback for input 0
    Inputs Values: [
    	[ [ 1.708, 1.208, -1.94 ], [ -1.412, -0.456, 1.964 ] ],
    	[ [ 1.932, 0.688, 1.776 ], [ 1.02, 1.504, 1.48 ] ]
    ]
    Value Statistics: {meanExponent=0.11945381654746186, negative=3, min=1.48, max=1.48, mean=0.7893333333333333, count=12.0, positive=9, stdDev=1.277220245515845, zeros=0}
    Implemented Feedback: [ [ 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
```
...[skipping 1071 bytes](etc/75.txt)...
```
    8333333333332416, count=144.0, positive=12, stdDev=0.2763853991962529, zeros=132}
    Feedback Error: [ [ -1.1013412404281553E-13, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.1013412404281553E-13, 0.0, ... ], [ 0.0, 0.0, 0.0, -1.1013412404281553E-13, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, -1.1013412404281553E-13, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.1013412404281553E-13, ... ], [ 0.0, 0.0, 0.0, 0.0, -1.1013412404281553E-13, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], ... ]
    Error Statistics: {meanExponent=-12.958078098036824, negative=12, min=-1.1013412404281553E-13, max=-1.1013412404281553E-13, mean=-9.177843670234628E-15, count=144.0, positive=0, stdDev=3.0439463838706555E-14, zeros=132}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 9.1778e-15 +- 3.0439e-14 [0.0000e+00 - 1.1013e-13] (144#)
    relativeTol: 5.5067e-14 +- 0.0000e+00 [5.5067e-14 - 5.5067e-14] (12#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=9.1778e-15 +- 3.0439e-14 [0.0000e+00 - 1.1013e-13] (144#), relativeTol=5.5067e-14 +- 0.0000e+00 [5.5067e-14 - 5.5067e-14] (12#)}
```



### Performance
Code from [StandardLayerTests.java:120](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L120) executed in 0.00 seconds: 
```java
    getPerformanceTester().test(layer, permPrototype);
```
Logging: 
```
    Evaluation performance: 0.000301s +- 0.000060s [0.000182s - 0.000335s]
    Learning performance: 0.000047s +- 0.000002s [0.000044s - 0.000051s]
    
```

