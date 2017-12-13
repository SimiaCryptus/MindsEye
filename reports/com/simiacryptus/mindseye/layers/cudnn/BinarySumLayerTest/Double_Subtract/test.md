# BinarySumLayer
## Double_Subtract
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
      "id": "10bdbd05-f6b7-4f27-b47e-d9cfca5798b8",
      "isFrozen": false,
      "name": "BinarySumLayer/10bdbd05-f6b7-4f27-b47e-d9cfca5798b8",
      "rightFactor": -1.0,
      "leftFactor": 1.0
    }
```



### Example Input/Output Pair
Code from [StandardLayerTests.java:153](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L153) executed in 0.00 seconds: 
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
    	[ [ -0.344 ], [ -1.536 ] ],
    	[ [ -1.236 ], [ 1.852 ] ]
    ],
    [
    	[ [ -0.244 ], [ -1.092 ] ],
    	[ [ 0.144 ], [ 1.78 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ -0.09999999999999998 ], [ -0.44399999999999995 ] ],
    	[ [ -1.38 ], [ 0.07200000000000006 ] ]
    ]
    --------------------
    Derivative: 
    [
    	[ [ 1.0 ], [ 1.0 ] ],
    	[ [ 1.0 ], [ 1.0 ] ]
    ],
    [
    	[ [ -1.0 ], [ -1.0 ] ],
    	[ [ -1.0 ], [ -1.0 ] ]
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
Code from [StandardLayerTests.java:110](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L110) executed in 0.01 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ 1.916 ], [ 0.316 ] ],
    	[ [ 1.572 ], [ -1.096 ] ]
    ],
    [
    	[ [ 0.768 ], [ -0.352 ] ],
    	[ [ 1.94 ], [ -1.24 ] ]
    ]
    Inputs Statistics: {meanExponent=0.004586420803167221, negative=1, min=-1.096, max=-1.096, mean=0.6769999999999999, count=4.0, positive=3, stdDev=1.184271506032295, zeros=0},
    {meanExponent=-0.04671817534947397, negative=2, min=-1.24, max=-1.24, mean=0.2790000000000001, count=4.0, positive=2, stdDev=1.1941067791449806, zeros=0}
    Output: [
    	[ [ 1.148 ], [ 0.6679999999999999 ] ],
    	[ [ -0.3679999999999999 ], [ 0.1439999999999999 ] ]
    ]
    Outputs Statistics: {meanExponent=-0.3477678346734332, negative=1, min=0.1439999999999999, max=0.1439999999999999, mean=0.39799999999999996, count=4.0, positive=3, stdDev=0.5671578263587658, zeros=0}
    Feedback for input 0
    Inputs Values: [
    	[ [ 1.916 ], [ 0.316 ] ],
    	[ [ 1.572 ], [ -1.096 ] ]
    ]
    Value Statistics: {meanExponent=0.004586420803167221, negative=1, min=-1.096, max=-1.096, mean=0.6769999999999999, count=4.0, positive=3, stdDev=1.184271506032295, ze
```
...[skipping 1580 bytes](etc/9.txt)...
```
    0, stdDev=0.4330127018922193, zeros=12}
    Measured Feedback: [ [ -0.9999999999998899, 0.0, 0.0, 0.0 ], [ 0.0, -0.9999999999998899, 0.0, 0.0 ], [ 0.0, 0.0, -0.9999999999998899, 0.0 ], [ 0.0, 0.0, 0.0, -0.9999999999998899 ] ]
    Measured Statistics: {meanExponent=-4.7830642341045674E-14, negative=4, min=-0.9999999999998899, max=-0.9999999999998899, mean=-0.24999999999997247, count=16.0, positive=0, stdDev=0.4330127018921716, zeros=12}
    Feedback Error: [ [ 1.1013412404281553E-13, 0.0, 0.0, 0.0 ], [ 0.0, 1.1013412404281553E-13, 0.0, 0.0 ], [ 0.0, 0.0, 1.1013412404281553E-13, 0.0 ], [ 0.0, 0.0, 0.0, 1.1013412404281553E-13 ] ]
    Error Statistics: {meanExponent=-12.958078098036825, negative=0, min=1.1013412404281553E-13, max=1.1013412404281553E-13, mean=2.7533531010703882E-14, count=16.0, positive=4, stdDev=4.7689474622312385E-14, zeros=12}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 2.7534e-14 +- 4.7689e-14 [0.0000e+00 - 1.1013e-13] (32#)
    relativeTol: 5.5067e-14 +- 0.0000e+00 [5.5067e-14 - 5.5067e-14] (8#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=2.7534e-14 +- 4.7689e-14 [0.0000e+00 - 1.1013e-13] (32#), relativeTol=5.5067e-14 +- 0.0000e+00 [5.5067e-14 - 5.5067e-14] (8#)}
```



### Performance
Code from [StandardLayerTests.java:120](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L120) executed in 0.01 seconds: 
```java
    getPerformanceTester().test(layer, permPrototype);
```
Logging: 
```
    Evaluation performance: 0.000330s +- 0.000067s [0.000271s - 0.000459s]
    Learning performance: 0.000226s +- 0.000009s [0.000212s - 0.000238s]
    
```

