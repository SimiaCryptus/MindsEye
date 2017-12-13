# BinarySumLayer
## Double_Add
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
      "class": "com.simiacryptus.mindseye.layers.cudnn.BinarySumLayer",
      "id": "c99bda88-3817-4103-8b5f-2415ac890992",
      "isFrozen": false,
      "name": "BinarySumLayer/c99bda88-3817-4103-8b5f-2415ac890992",
      "rightFactor": 1.0,
      "leftFactor": 1.0
    }
```



### Example Input/Output Pair
Code from [StandardLayerTests.java:153](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L153) executed in 0.01 seconds: 
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
    	[ [ 1.028 ], [ -0.856 ] ],
    	[ [ -1.64 ], [ -1.408 ] ]
    ],
    [
    	[ [ -1.68 ], [ 1.444 ] ],
    	[ [ -1.984 ], [ -1.632 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ -0.6519999999999999 ], [ 0.588 ] ],
    	[ [ -3.6239999999999997 ], [ -3.04 ] ]
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
Code from [StandardLayerTests.java:102](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L102) executed in 0.01 seconds: 
```java
    return getBatchingTester().test(layer, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (120#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (120#)}
```



### Differential Validation
Code from [StandardLayerTests.java:110](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L110) executed in 0.02 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ 0.284 ], [ -1.836 ] ],
    	[ [ 1.828 ], [ -1.432 ] ]
    ],
    [
    	[ [ -0.768 ], [ -1.484 ] ],
    	[ [ 0.432 ], [ -1.72 ] ]
    ]
    Inputs Statistics: {meanExponent=0.033777556570477665, negative=2, min=-1.432, max=-1.432, mean=-0.289, count=4.0, positive=2, stdDev=1.4585674478747974, zeros=0},
    {meanExponent=-0.018048171325754682, negative=3, min=-1.72, max=-1.72, mean=-0.885, count=4.0, positive=1, stdDev=0.8372878835860459, zeros=0}
    Output: [
    	[ [ -0.48400000000000004 ], [ -3.3200000000000003 ] ],
    	[ [ 2.2600000000000002 ], [ -3.152 ] ]
    ]
    Outputs Statistics: {meanExponent=0.26466952332834187, negative=3, min=-3.152, max=-3.152, mean=-1.174, count=4.0, positive=1, stdDev=2.279597332863855, zeros=0}
    Feedback for input 0
    Inputs Values: [
    	[ [ 0.284 ], [ -1.836 ] ],
    	[ [ 1.828 ], [ -1.432 ] ]
    ]
    Value Statistics: {meanExponent=0.033777556570477665, negative=2, min=-1.432, max=-1.432, mean=-0.289, count=4.0, positive=2, stdDev=1.4585674478747974, zeros=0}
    Implemented Feedback: [ [ 1.0, 0.0, 0.0, 0.0 ], [ 0.0, 1
```
...[skipping 1482 bytes](etc/8.txt)...
```
    positive=4, stdDev=0.4330127018922193, zeros=12}
    Measured Feedback: [ [ 0.9999999999998899, 0.0, 0.0, 0.0 ], [ 0.0, 0.9999999999976694, 0.0, 0.0 ], [ 0.0, 0.0, 1.0000000000021103, 0.0 ], [ 0.0, 0.0, 0.0, 1.0000000000021103 ] ]
    Measured Statistics: {meanExponent=1.932512242964997E-13, negative=0, min=1.0000000000021103, max=1.0000000000021103, mean=0.25000000000011124, count=16.0, positive=4, stdDev=0.433012701892412, zeros=12}
    Feedback Error: [ [ -1.1013412404281553E-13, 0.0, 0.0, 0.0 ], [ 0.0, -2.3305801732931286E-12, 0.0, 0.0 ], [ 0.0, 0.0, 2.1103119252074976E-12, 0.0 ], [ 0.0, 0.0, 0.0, 2.1103119252074976E-12 ] ]
    Error Statistics: {meanExponent=-11.985480186078018, negative=2, min=2.1103119252074976E-12, max=2.1103119252074976E-12, mean=1.1124434706744069E-13, count=16.0, positive=2, stdDev=9.404972566646684E-13, zeros=12}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 4.1633e-13 +- 8.5063e-13 [0.0000e+00 - 2.3306e-12] (32#)
    relativeTol: 8.3267e-13 +- 4.5119e-13 [5.5067e-14 - 1.1653e-12] (8#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=4.1633e-13 +- 8.5063e-13 [0.0000e+00 - 2.3306e-12] (32#), relativeTol=8.3267e-13 +- 4.5119e-13 [5.5067e-14 - 1.1653e-12] (8#)}
```



### Performance
Code from [StandardLayerTests.java:120](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L120) executed in 0.01 seconds: 
```java
    getPerformanceTester().test(layer, permPrototype);
```
Logging: 
```
    Evaluation performance: 0.000321s +- 0.000055s [0.000257s - 0.000394s]
    Learning performance: 0.000244s +- 0.000036s [0.000186s - 0.000285s]
    
```

