# ImgBandSelectLayer
## ImgBandSelectLayerTest
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
      "class": "com.simiacryptus.mindseye.layers.java.ImgBandSelectLayer",
      "id": "cb1deeb8-e489-4cdb-8d53-af13bca39e76",
      "isFrozen": false,
      "name": "ImgBandSelectLayer/cb1deeb8-e489-4cdb-8d53-af13bca39e76",
      "bands": [
        0,
        2
      ]
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
    	[ [ 1.816, -0.188, 1.528 ], [ 1.556, 0.04, -1.948 ] ],
    	[ [ -0.916, 0.26, 0.276 ], [ -0.364, 0.8, -1.364 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 1.816, 1.528 ], [ 1.556, -1.948 ] ],
    	[ [ -0.916, 0.276 ], [ -0.364, -1.364 ] ]
    ]
    --------------------
    Derivative: 
    [
    	[ [ 1.0, 0.0, 1.0 ], [ 1.0, 0.0, 1.0 ] ],
    	[ [ 1.0, 0.0, 1.0 ], [ 1.0, 0.0, 1.0 ] ]
    ]
```



### Batch Execution
Code from [StandardLayerTests.java:102](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L102) executed in 0.00 seconds: 
```java
    return getBatchingTester().test(layer, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (200#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (160#)}
```



### Differential Validation
Code from [StandardLayerTests.java:110](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L110) executed in 0.00 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ 1.308, -0.992, -1.236 ], [ 1.632, -1.532, -0.716 ] ],
    	[ [ 0.416, 0.864, 0.956 ], [ -1.828, 0.16, 0.104 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.12689797436235326, negative=5, min=0.104, max=0.104, mean=-0.07200000000000002, count=12.0, positive=7, stdDev=1.1141142969492255, zeros=0}
    Output: [
    	[ [ 1.308, -1.236 ], [ 1.632, -0.716 ] ],
    	[ [ 0.416, 0.956 ], [ -1.828, 0.104 ] ]
    ]
    Outputs Statistics: {meanExponent=-0.10564748186672762, negative=3, min=0.104, max=0.104, mean=0.07949999999999997, count=8.0, positive=5, stdDev=1.1620816451523535, zeros=0}
    Feedback for input 0
    Inputs Values: [
    	[ [ 1.308, -0.992, -1.236 ], [ 1.632, -1.532, -0.716 ] ],
    	[ [ 0.416, 0.864, 0.956 ], [ -1.828, 0.16, 0.104 ] ]
    ]
    Value Statistics: {meanExponent=-0.12689797436235326, negative=5, min=0.104, max=0.104, mean=-0.07200000000000002, count=12.0, positive=7, stdDev=1.1141142969492255, zeros=0}
    Implemented Feedback: [ [ 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0
```
...[skipping 902 bytes](etc/73.txt)...
```
    -4.029683400859781E-14, negative=0, min=1.0000000000000286, max=1.0000000000000286, mean=0.0833333333333256, count=96.0, positive=8, stdDev=0.27638539919625765, zeros=88}
    Feedback Error: [ [ -1.1013412404281553E-13, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, -1.1013412404281553E-13, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, -1.1013412404281553E-13, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, -1.1013412404281553E-13, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], ... ]
    Error Statistics: {meanExponent=-13.031189593810693, negative=7, min=2.864375403532904E-14, max=2.864375403532904E-14, mean=-7.732240773587288E-15, count=96.0, positive=1, stdDev=2.8865264781580964E-14, zeros=88}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 8.3290e-15 +- 2.8699e-14 [0.0000e+00 - 1.1013e-13] (96#)
    relativeTol: 4.9974e-14 +- 1.3475e-14 [1.4322e-14 - 5.5067e-14] (8#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=8.3290e-15 +- 2.8699e-14 [0.0000e+00 - 1.1013e-13] (96#), relativeTol=4.9974e-14 +- 1.3475e-14 [1.4322e-14 - 5.5067e-14] (8#)}
```



### Performance
Code from [StandardLayerTests.java:120](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L120) executed in 0.00 seconds: 
```java
    getPerformanceTester().test(layer, permPrototype);
```
Logging: 
```
    Evaluation performance: 0.000208s +- 0.000017s [0.000182s - 0.000230s]
    Learning performance: 0.000047s +- 0.000003s [0.000043s - 0.000051s]
    
```

