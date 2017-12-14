# ImgConcatLayer
## ImgConcatLayerTest
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
      "class": "com.simiacryptus.mindseye.layers.java.ImgConcatLayer",
      "id": "3bdb00e7-b6c1-446b-959a-1e6f14b9b828",
      "isFrozen": false,
      "name": "ImgConcatLayer/3bdb00e7-b6c1-446b-959a-1e6f14b9b828"
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
    	[ [ 1.144 ], [ 0.196 ] ],
    	[ [ -0.552 ], [ -0.02 ] ]
    ],
    [
    	[ [ 0.372 ], [ 0.652 ] ],
    	[ [ -0.468 ], [ 1.916 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 1.144, 0.372 ], [ 0.196, 0.652 ] ],
    	[ [ -0.552, -0.468 ], [ -0.02, 1.916 ] ]
    ]
    --------------------
    Derivative: 
    [
    	[ [ 1.0 ], [ 1.0 ] ],
    	[ [ 1.0 ], [ 1.0 ] ]
    ],
    [
    	[ [ 1.0 ], [ 1.0 ] ],
    	[ [ 1.0 ], [ 1.0 ] ]
    ]
```



### Batch Execution
Code from [StandardLayerTests.java:101](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L101) executed in 0.00 seconds: 
```java
    return getBatchingTester().test(layer, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (160#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (160#)}
```



### Differential Validation
Code from [StandardLayerTests.java:109](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L109) executed in 0.00 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ 1.228 ], [ -1.004 ] ],
    	[ [ -1.616 ], [ 0.352 ] ]
    ],
    [
    	[ [ 1.28 ], [ 1.84 ] ],
    	[ [ 0.52 ], [ -0.684 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.03852097511728805, negative=2, min=0.352, max=0.352, mean=-0.26, count=4.0, positive=2, stdDev=1.1159032216101896, zeros=0},
    {meanExponent=-0.01922819049691992, negative=1, min=-0.684, max=-0.684, mean=0.739, count=4.0, positive=3, stdDev=0.9457499669574407, zeros=0}
    Output: [
    	[ [ 1.228, 1.28 ], [ -1.004, 1.84 ] ],
    	[ [ -1.616, 0.52 ], [ 0.352, -0.684 ] ]
    ]
    Outputs Statistics: {meanExponent=-0.028874582807103987, negative=3, min=-0.684, max=-0.684, mean=0.2395, count=8.0, positive=5, stdDev=1.1486260270427449, zeros=0}
    Feedback for input 0
    Inputs Values: [
    	[ [ 1.228 ], [ -1.004 ] ],
    	[ [ -1.616 ], [ 0.352 ] ]
    ]
    Value Statistics: {meanExponent=-0.03852097511728805, negative=2, min=0.352, max=0.352, mean=-0.26, count=4.0, positive=2, stdDev=1.1159032216101896, zeros=0}
    Implemented Feedback: [ [ 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 1.0
```
...[skipping 1888 bytes](etc/116.txt)...
```
    998899, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9999999999998899, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9999999999998899 ] ]
    Measured Statistics: {meanExponent=-4.7830642341045674E-14, negative=0, min=0.9999999999998899, max=0.9999999999998899, mean=0.12499999999998623, count=32.0, positive=4, stdDev=0.3307189138830374, zeros=28}
    Feedback Error: [ [ 0.0, 0.0, 0.0, 0.0, -1.1013412404281553E-13, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, -1.1013412404281553E-13, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.1013412404281553E-13, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.1013412404281553E-13 ] ]
    Error Statistics: {meanExponent=-12.958078098036825, negative=4, min=-1.1013412404281553E-13, max=-1.1013412404281553E-13, mean=-1.3766765505351941E-14, count=32.0, positive=0, stdDev=3.6423437884903677E-14, zeros=28}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 1.3767e-14 +- 3.6423e-14 [0.0000e+00 - 1.1013e-13] (64#)
    relativeTol: 5.5067e-14 +- 0.0000e+00 [5.5067e-14 - 5.5067e-14] (8#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=1.3767e-14 +- 3.6423e-14 [0.0000e+00 - 1.1013e-13] (64#), relativeTol=5.5067e-14 +- 0.0000e+00 [5.5067e-14 - 5.5067e-14] (8#)}
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
    	[2, 2, 1]
    	[2, 2, 1]
    Performance:
    	Evaluation performance: 0.000127s +- 0.000024s [0.000105s - 0.000173s]
    	Learning performance: 0.000212s +- 0.000021s [0.000189s - 0.000250s]
    
```

