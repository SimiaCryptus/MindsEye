# BinarySumLayer
## Double_Add
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
      "id": "b6def34d-fb37-4d7e-8cd8-25efa48ff387",
      "isFrozen": false,
      "name": "BinarySumLayer/b6def34d-fb37-4d7e-8cd8-25efa48ff387",
      "rightFactor": 1.0,
      "leftFactor": 1.0
    }
```



### Example Input/Output Pair
Code from [StandardLayerTests.java:152](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L152) executed in 0.01 seconds: 
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
    	[ [ -1.804 ], [ -0.292 ] ],
    	[ [ 1.292 ], [ 0.64 ] ]
    ],
    [
    	[ [ -0.392 ], [ -1.144 ] ],
    	[ [ 1.324 ], [ -0.392 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ -2.196 ], [ -1.436 ] ],
    	[ [ 2.616 ], [ 0.248 ] ]
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
Code from [StandardLayerTests.java:109](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L109) executed in 0.02 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ -0.088 ], [ 0.316 ] ],
    	[ [ 0.44 ], [ 1.3 ] ]
    ],
    [
    	[ [ 0.304 ], [ 0.832 ] ],
    	[ [ 0.912 ], [ 1.512 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.4496085541096008, negative=1, min=1.3, max=1.3, mean=0.492, count=4.0, positive=3, stdDev=0.5057034704251099, zeros=0},
    {meanExponent=-0.11436411515172962, negative=0, min=1.512, max=1.512, mean=0.89, count=4.0, positive=4, stdDev=0.4284063491593, zeros=0}
    Output: [
    	[ [ 0.216 ], [ 1.148 ] ],
    	[ [ 1.352 ], [ 2.8120000000000003 ] ]
    ]
    Outputs Statistics: {meanExponent=-0.006403088208427707, negative=0, min=2.8120000000000003, max=2.8120000000000003, mean=1.3820000000000001, count=4.0, positive=4, stdDev=0.9300688146583566, zeros=0}
    Feedback for input 0
    Inputs Values: [
    	[ [ -0.088 ], [ 0.316 ] ],
    	[ [ 0.44 ], [ 1.3 ] ]
    ]
    Value Statistics: {meanExponent=-0.4496085541096008, negative=1, min=1.3, max=1.3, mean=0.492, count=4.0, positive=3, stdDev=0.5057034704251099, zeros=0}
    Implemented Feedback: [ [ 1.0, 0.0, 0.0, 0.0 ], [ 0.0, 1.0, 0.0, 0.0 ], [ 0.0, 
```
...[skipping 1464 bytes](etc/45.txt)...
```
    ve=4, stdDev=0.4330127018922193, zeros=12}
    Measured Feedback: [ [ 0.9999999999998899, 0.0, 0.0, 0.0 ], [ 0.0, 0.9999999999998899, 0.0, 0.0 ], [ 0.0, 0.0, 0.9999999999998899, 0.0 ], [ 0.0, 0.0, 0.0, 0.9999999999976694 ] ]
    Measured Statistics: {meanExponent=-2.8891250897966165E-13, negative=0, min=0.9999999999976694, max=0.9999999999976694, mean=0.2499999999998337, count=16.0, positive=4, stdDev=0.43301270189193125, zeros=12}
    Feedback Error: [ [ -1.1013412404281553E-13, 0.0, 0.0, 0.0 ], [ 0.0, -1.1013412404281553E-13, 0.0, 0.0 ], [ 0.0, 0.0, -1.1013412404281553E-13, 0.0 ], [ 0.0, 0.0, 0.0, -2.3305801732931286E-12 ] ]
    Error Statistics: {meanExponent=-12.62669256165148, negative=4, min=-2.3305801732931286E-12, max=-2.3305801732931286E-12, mean=-1.6631140908884845E-13, count=16.0, positive=0, stdDev=5.60437371796192E-13, zeros=12}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 1.6810e-13 +- 5.6035e-13 [0.0000e+00 - 2.3306e-12] (32#)
    relativeTol: 3.3620e-13 +- 4.7876e-13 [5.5067e-14 - 1.1653e-12] (8#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=1.6810e-13 +- 5.6035e-13 [0.0000e+00 - 2.3306e-12] (32#), relativeTol=3.3620e-13 +- 4.7876e-13 [5.5067e-14 - 1.1653e-12] (8#)}
```



### Performance
Code from [StandardLayerTests.java:119](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L119) executed in 0.30 seconds: 
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
    	Evaluation performance: 0.017714s +- 0.001040s [0.015709s - 0.018725s]
    	Learning performance: 0.027072s +- 0.002679s [0.023360s - 0.030890s]
    
```

