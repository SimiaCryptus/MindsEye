# ImgConcatLayer
## ImgConcatLayerTest
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
      "class": "com.simiacryptus.mindseye.layers.java.ImgConcatLayer",
      "id": "5ba3c547-3918-4b08-8965-5a062f500d09",
      "isFrozen": false,
      "name": "ImgConcatLayer/5ba3c547-3918-4b08-8965-5a062f500d09"
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
    	[ [ 1.02 ], [ -0.948 ] ],
    	[ [ 1.92 ], [ 1.72 ] ]
    ],
    [
    	[ [ -1.484 ], [ -1.948 ] ],
    	[ [ 1.028 ], [ -1.644 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 1.02, -1.484 ], [ -0.948, -1.948 ] ],
    	[ [ 1.92, 1.028 ], [ 1.72, -1.644 ] ]
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
Code from [StandardLayerTests.java:102](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L102) executed in 0.00 seconds: 
```java
    return getBatchingTester().test(layer, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (160#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (159#)}
```



### Differential Validation
Code from [StandardLayerTests.java:110](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L110) executed in 0.00 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ 0.928 ], [ 0.676 ] ],
    	[ [ 1.244 ], [ -0.704 ] ]
    ],
    [
    	[ [ 1.896 ], [ 1.608 ] ],
    	[ [ -1.588 ], [ -0.62 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.06502807208564747, negative=1, min=-0.704, max=-0.704, mean=0.536, count=4.0, positive=3, stdDev=0.7436612131878333, zeros=0},
    {meanExponent=0.11934164125095281, negative=2, min=-0.62, max=-0.62, mean=0.32399999999999995, count=4.0, positive=2, stdDev=1.4719646734891432, zeros=0}
    Output: [
    	[ [ 0.928, 1.896 ], [ 0.676, 1.608 ] ],
    	[ [ 1.244, -1.588 ], [ -0.704, -0.62 ] ]
    ]
    Outputs Statistics: {meanExponent=0.02715678458265267, negative=3, min=-0.62, max=-0.62, mean=0.43000000000000005, count=8.0, positive=5, stdDev=1.1709363774347434, zeros=0}
    Feedback for input 0
    Inputs Values: [
    	[ [ 0.928 ], [ 0.676 ] ],
    	[ [ 1.244 ], [ -0.704 ] ]
    ]
    Value Statistics: {meanExponent=-0.06502807208564747, negative=1, min=-0.704, max=-0.704, mean=0.536, count=4.0, positive=3, stdDev=0.7436612131878333, zeros=0}
    Implemented Feedback: [ [ 1.0, 0.0, 0.0, 0.0, 0.0, 
```
...[skipping 1929 bytes](etc/74.txt)...
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
Code from [StandardLayerTests.java:120](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L120) executed in 0.00 seconds: 
```java
    getPerformanceTester().test(layer, permPrototype);
```
Logging: 
```
    Evaluation performance: 0.000105s +- 0.000009s [0.000093s - 0.000118s]
    Learning performance: 0.000192s +- 0.000005s [0.000185s - 0.000201s]
    
```

