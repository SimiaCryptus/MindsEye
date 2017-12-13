# ImgBandBiasLayer
## Double
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
      "class": "com.simiacryptus.mindseye.layers.cudnn.ImgBandBiasLayer",
      "id": "ec214008-de46-437d-bfed-f46070a1ced7",
      "isFrozen": false,
      "name": "ImgBandBiasLayer/ec214008-de46-437d-bfed-f46070a1ced7",
      "bias": [
        0.0,
        0.0
      ]
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
    	[ [ -1.48, -0.772 ], [ -0.396, -1.3 ], [ -1.204, 1.844 ] ],
    	[ [ 1.86, -1.08 ], [ -0.676, 1.756 ], [ 1.06, 1.344 ] ],
    	[ [ -0.884, -1.452 ], [ -1.04, 0.64 ], [ -1.396, 0.152 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ -1.48, -0.772 ], [ -0.396, -1.3 ], [ -1.204, 1.844 ] ],
    	[ [ 1.86, -1.08 ], [ -0.676, 1.756 ], [ 1.06, 1.344 ] ],
    	[ [ -0.884, -1.452 ], [ -1.04, 0.64 ], [ -1.396, 0.152 ] ]
    ]
    --------------------
    Derivative: 
    [
    	[ [ 1.0, 1.0 ], [ 1.0, 1.0 ], [ 1.0, 1.0 ] ],
    	[ [ 1.0, 1.0 ], [ 1.0, 1.0 ], [ 1.0, 1.0 ] ],
    	[ [ 1.0, 1.0 ], [ 1.0, 1.0 ], [ 1.0, 1.0 ] ]
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
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (360#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (360#)}
```



### Differential Validation
Code from [StandardLayerTests.java:110](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L110) executed in 0.03 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ -0.12, 0.532 ], [ -1.956, 1.468 ], [ 1.816, -1.432 ] ],
    	[ [ 0.912, -1.38 ], [ 1.212, -1.06 ], [ 1.18, 1.492 ] ],
    	[ [ -0.948, 1.108 ], [ -0.692, -1.232 ], [ 1.864, -0.896 ] ]
    ]
    Inputs Statistics: {meanExponent=0.017077705805244748, negative=9, min=-0.896, max=-0.896, mean=0.10377777777777782, count=18.0, positive=9, stdDev=1.2644833620255769, zeros=0}
    Output: [
    	[ [ -0.12, 0.532 ], [ -1.956, 1.468 ], [ 1.816, -1.432 ] ],
    	[ [ 0.912, -1.38 ], [ 1.212, -1.06 ], [ 1.18, 1.492 ] ],
    	[ [ -0.948, 1.108 ], [ -0.692, -1.232 ], [ 1.864, -0.896 ] ]
    ]
    Outputs Statistics: {meanExponent=0.017077705805244748, negative=9, min=-0.896, max=-0.896, mean=0.10377777777777782, count=18.0, positive=9, stdDev=1.2644833620255769, zeros=0}
    Feedback for input 0
    Inputs Values: [
    	[ [ -0.12, 0.532 ], [ -1.956, 1.468 ], [ 1.816, -1.432 ] ],
    	[ [ 0.912, -1.38 ], [ 1.212, -1.06 ], [ 1.18, 1.492 ] ],
    	[ [ -0.948, 1.108 ], [ -0.692, -1.232 ], [ 1.864, -0.896 ] ]
    ]
    Value Statistics: {meanExponent=0.017077705805244748, ne
```
...[skipping 2671 bytes](etc/31.txt)...
```
    0.9999999999998899, 0.9999999999998899, 0.9999999999998899, 0.9999999999998899, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ] ]
    Measured Statistics: {meanExponent=-4.448228308217994E-14, negative=0, min=0.9999999999998899, max=0.9999999999998899, mean=0.49999999999994876, count=36.0, positive=18, stdDev=0.4999999999999488, zeros=18}
    Gradient Error: [ [ 2.864375403532904E-14, -1.1013412404281553E-13, -1.1013412404281553E-13, -1.1013412404281553E-13, -1.1013412404281553E-13, -1.1013412404281553E-13, -1.1013412404281553E-13, -1.1013412404281553E-13, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ] ]
    Error Statistics: {meanExponent=-12.990572096158543, negative=17, min=-1.1013412404281553E-13, max=-1.1013412404281553E-13, mean=-5.1212120963681526E-14, count=36.0, positive=1, stdDev=5.592799596435784E-14, zeros=18}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 1.0561e-14 +- 3.2227e-14 [0.0000e+00 - 1.1013e-13] (360#)
    relativeTol: 5.2803e-14 +- 9.3332e-15 [1.4322e-14 - 5.5067e-14] (36#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=1.0561e-14 +- 3.2227e-14 [0.0000e+00 - 1.1013e-13] (360#), relativeTol=5.2803e-14 +- 9.3332e-15 [1.4322e-14 - 5.5067e-14] (36#)}
```



### Performance
Code from [StandardLayerTests.java:120](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L120) executed in 0.01 seconds: 
```java
    getPerformanceTester().test(layer, permPrototype);
```
Logging: 
```
    Evaluation performance: 0.000270s +- 0.000056s [0.000226s - 0.000380s]
    Learning performance: 0.000456s +- 0.000030s [0.000423s - 0.000511s]
    
```

