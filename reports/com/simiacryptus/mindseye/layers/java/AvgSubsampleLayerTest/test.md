# AvgSubsampleLayer
## AvgSubsampleLayerTest
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
      "class": "com.simiacryptus.mindseye.layers.java.AvgSubsampleLayer",
      "id": "8750ccfd-c2f1-4da3-a467-c53531689bbb",
      "isFrozen": false,
      "name": "AvgSubsampleLayer/8750ccfd-c2f1-4da3-a467-c53531689bbb",
      "inner": [
        2,
        2,
        1
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
    	[ [ 0.564, 1.816, -0.26 ], [ -1.712, 0.568, -1.008 ] ],
    	[ [ -0.208, 1.752, 1.184 ], [ 1.628, -1.848, 1.764 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 0.0, 0.0, 1.06 ] ]
    ]
    --------------------
    Derivative: 
    [
    	[ [ 0.25, 0.25, 0.25 ], [ 0.25, 0.25, 0.25 ] ],
    	[ [ 0.25, 0.25, 0.25 ], [ 0.25, 0.25, 0.25 ] ]
    ]
```



### Batch Execution
Code from [StandardLayerTests.java:102](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L102) executed in 0.00 seconds: 
```java
    return getBatchingTester().test(layer, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (150#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (130#)}
```



### Differential Validation
Code from [StandardLayerTests.java:110](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L110) executed in 0.00 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ 0.36, 1.728, -0.536 ], [ 0.264, -0.416, -1.908 ] ],
    	[ [ 1.484, 1.408, -1.276 ], [ 0.748, 1.46, -1.156 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.0523844854720637, negative=5, min=-1.156, max=-1.156, mean=0.17999999999999985, count=12.0, positive=7, stdDev=1.1810176402859809, zeros=0}
    Output: [
    	[ [ 0.0, 0.0, 0.54 ] ]
    ]
    Outputs Statistics: {meanExponent=-0.26760624017703144, negative=0, min=0.54, max=0.54, mean=0.18000000000000002, count=3.0, positive=1, stdDev=0.2545584412271571, zeros=2}
    Feedback for input 0
    Inputs Values: [
    	[ [ 0.36, 1.728, -0.536 ], [ 0.264, -0.416, -1.908 ] ],
    	[ [ 1.484, 1.408, -1.276 ], [ 0.748, 1.46, -1.156 ] ]
    ]
    Value Statistics: {meanExponent=-0.0523844854720637, negative=5, min=-1.156, max=-1.156, mean=0.17999999999999985, count=12.0, positive=7, stdDev=1.1810176402859809, zeros=0}
    Implemented Feedback: [ [ 0.0, 0.0, 0.25 ], [ 0.0, 0.0, 0.25 ], [ 0.0, 0.0, 0.25 ], [ 0.0, 0.0, 0.25 ], [ 0.0, 0.0, 0.25 ], [ 0.0, 0.0, 0.25 ], [ 0.0, 0.0, 0.25 ], [ 0.0, 0.0, 0.25 
```
...[skipping 402 bytes](etc/56.txt)...
```
    99999999941735 ], [ 0.0, 0.0, 0.24999999999941735 ], [ 0.0, 0.0, 0.2500000000016378 ], ... ]
    Measured Statistics: {meanExponent=-0.6020599913280104, negative=0, min=0.24999999999941735, max=0.24999999999941735, mean=0.08333333333332416, count=36.0, positive=12, stdDev=0.11785113019774494, zeros=24}
    Feedback Error: [ [ 0.0, 0.0, 1.637801005927031E-12 ], [ 0.0, 0.0, 1.637801005927031E-12 ], [ 0.0, 0.0, -5.826450433232822E-13 ], [ 0.0, 0.0, -5.826450433232822E-13 ], [ 0.0, 0.0, -5.826450433232822E-13 ], [ 0.0, 0.0, -5.826450433232822E-13 ], [ 0.0, 0.0, -5.826450433232822E-13 ], [ 0.0, 0.0, 1.637801005927031E-12 ], ... ]
    Error Statistics: {meanExponent=-12.12238167448791, negative=9, min=-5.826450433232822E-13, max=-5.826450433232822E-13, mean=-9.177843670234628E-15, count=36.0, positive=3, stdDev=5.552632319277302E-13, zeros=24}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 2.8214e-13 +- 4.7833e-13 [0.0000e+00 - 1.6378e-12] (36#)
    relativeTol: 1.6929e-12 +- 9.1379e-13 [1.1653e-12 - 3.2756e-12] (12#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=2.8214e-13 +- 4.7833e-13 [0.0000e+00 - 1.6378e-12] (36#), relativeTol=1.6929e-12 +- 9.1379e-13 [1.1653e-12 - 3.2756e-12] (12#)}
```



### Performance
Code from [StandardLayerTests.java:120](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L120) executed in 0.01 seconds: 
```java
    getPerformanceTester().test(layer, permPrototype);
```
Logging: 
```
    Evaluation performance: 0.000660s +- 0.000086s [0.000588s - 0.000778s]
    Learning performance: 0.000036s +- 0.000013s [0.000025s - 0.000061s]
    
```

