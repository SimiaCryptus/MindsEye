# ImgReshapeLayer
## ImgReshapeLayerTest
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
      "class": "com.simiacryptus.mindseye.layers.java.ImgReshapeLayer",
      "id": "ce5a1df6-5a8c-4da4-bb63-47ccac18a89a",
      "isFrozen": false,
      "name": "ImgReshapeLayer/ce5a1df6-5a8c-4da4-bb63-47ccac18a89a",
      "kernelSizeX": 2,
      "kernelSizeY": 2,
      "expand": false
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
    	[ [ -0.132, -0.308, -0.188 ], [ 1.676, 1.18, 0.416 ] ],
    	[ [ 0.944, 0.064, 0.944 ], [ 0.76, 1.004, -0.996 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ -0.132, -0.308, -0.188, 1.676, 1.18, 0.416, 0.944, 0.064, ... ] ]
    ]
    --------------------
    Derivative: 
    [
    	[ [ 1.0, 1.0, 1.0 ], [ 1.0, 1.0, 1.0 ] ],
    	[ [ 1.0, 1.0, 1.0 ], [ 1.0, 1.0, 1.0 ] ]
    ]
```



### Batch Execution
Code from [StandardLayerTests.java:101](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L101) executed in 0.00 seconds: 
```java
    return getBatchingTester().test(layer, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (240#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (240#)}
```



### Differential Validation
Code from [StandardLayerTests.java:109](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L109) executed in 0.00 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ 1.308, -1.48, 1.28 ], [ -1.708, -0.612, -0.08 ] ],
    	[ [ -0.012, 1.068, -1.624 ], [ 1.548, 0.124, 0.436 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.2868811367546184, negative=6, min=0.436, max=0.436, mean=0.020666666666666677, count=12.0, positive=6, stdDev=1.125931712948091, zeros=0}
    Output: [
    	[ [ 1.308, -1.48, 1.28, -1.708, -0.612, -0.08, -0.012, 1.068, ... ] ]
    ]
    Outputs Statistics: {meanExponent=-0.28688113675461846, negative=6, min=0.436, max=0.436, mean=0.02066666666666668, count=12.0, positive=6, stdDev=1.125931712948091, zeros=0}
    Feedback for input 0
    Inputs Values: [
    	[ [ 1.308, -1.48, 1.28 ], [ -1.708, -0.612, -0.08 ] ],
    	[ [ -0.012, 1.068, -1.624 ], [ 1.548, 0.124, 0.436 ] ]
    ]
    Value Statistics: {meanExponent=-0.2868811367546184, negative=6, min=0.436, max=0.436, mean=0.020666666666666677, count=12.0, positive=6, stdDev=1.125931712948091, zeros=0}
    Implemented Feedback: [ [ 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, ... ], [ 0.0, 0.0, 0.
```
...[skipping 1087 bytes](etc/117.txt)...
```
    8333333333332679, count=144.0, positive=12, stdDev=0.27638539919626165, zeros=132}
    Feedback Error: [ [ -1.1013412404281553E-13, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -5.995204332975845E-15, 0.0, ... ], [ 0.0, 0.0, 0.0, -1.1013412404281553E-13, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, -1.1013412404281553E-13, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.1013412404281553E-13, ... ], [ 0.0, 0.0, 0.0, 0.0, -1.1013412404281553E-13, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], ... ]
    Error Statistics: {meanExponent=-13.160903251762917, negative=10, min=-1.1013412404281553E-13, max=-1.1013412404281553E-13, mean=-6.527186198942066E-15, count=144.0, positive=2, stdDev=2.6965461484229702E-14, zeros=132}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 7.3228e-15 +- 2.6760e-14 [0.0000e+00 - 1.1013e-13] (144#)
    relativeTol: 4.3937e-14 +- 1.9462e-14 [2.9976e-15 - 5.5067e-14] (12#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=7.3228e-15 +- 2.6760e-14 [0.0000e+00 - 1.1013e-13] (144#), relativeTol=4.3937e-14 +- 1.9462e-14 [2.9976e-15 - 5.5067e-14] (12#)}
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
    	Evaluation performance: 0.000281s +- 0.000058s [0.000197s - 0.000338s]
    	Learning performance: 0.000071s +- 0.000014s [0.000051s - 0.000093s]
    
```

