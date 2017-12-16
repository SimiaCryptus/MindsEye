# AvgSubsampleLayer
## AvgSubsampleLayerTest
### Batch Execution
Code from [BatchingTester.java:66](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/BatchingTester.java#L66) executed in 0.00 seconds: 
```java
    return test(reference, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (150#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (130#)}
```



Code from [SingleDerivativeTester.java:77](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/SingleDerivativeTester.java#L77) executed in 0.00 seconds: 
```java
    return test(component, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ -1.548, -1.436, -1.124 ], [ 1.62, -1.6, 1.684 ] ],
    	[ [ 1.34, 0.08, 0.232 ], [ 0.432, -0.7, 0.712 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.1027988881198283, negative=5, min=0.712, max=0.712, mean=-0.025666666666666654, count=12.0, positive=7, stdDev=1.1822830550347165, zeros=0}
    Output: [
    	[ [ 0.0, 0.0, -0.07699999999999996 ] ]
    ]
    Outputs Statistics: {meanExponent=-1.1135092748275184, negative=1, min=-0.07699999999999996, max=-0.07699999999999996, mean=-0.025666666666666654, count=3.0, positive=0, stdDev=0.036298148100909415, zeros=2}
    Feedback for input 0
    Inputs Values: [
    	[ [ -1.548, -1.436, -1.124 ], [ 1.62, -1.6, 1.684 ] ],
    	[ [ 1.34, 0.08, 0.232 ], [ 0.432, -0.7, 0.712 ] ]
    ]
    Value Statistics: {meanExponent=-0.1027988881198283, negative=5, min=0.712, max=0.712, mean=-0.025666666666666654, count=12.0, positive=7, stdDev=1.1822830550347165, zeros=0}
    Implemented Feedback: [ [ 0.0, 0.0, 0.25 ], [ 0.0, 0.0, 0.25 ], [ 0.0, 0.0, 0.25 ], [ 0.0, 0.0, 0.25 ], [ 0.0, 0.0, 0.25 ], [ 0.0, 0.0, 0.25
```
...[skipping 446 bytes](etc/194.txt)...
```
    9999999941735 ], [ 0.0, 0.0, 0.24999999999941735 ], [ 0.0, 0.0, 0.24999999999941735 ], ... ]
    Measured Statistics: {meanExponent=-0.6020599913289747, negative=0, min=0.24999999999941735, max=0.24999999999941735, mean=0.08333333333313912, count=36.0, positive=12, stdDev=0.11785113019748325, zeros=24}
    Feedback Error: [ [ 0.0, 0.0, -5.826450433232822E-13 ], [ 0.0, 0.0, -5.826450433232822E-13 ], [ 0.0, 0.0, -5.826450433232822E-13 ], [ 0.0, 0.0, -5.826450433232822E-13 ], [ 0.0, 0.0, -5.826450433232822E-13 ], [ 0.0, 0.0, -5.826450433232822E-13 ], [ 0.0, 0.0, -5.826450433232822E-13 ], [ 0.0, 0.0, -5.826450433232822E-13 ], ... ]
    Error Statistics: {meanExponent=-12.234595943823395, negative=12, min=-5.826450433232822E-13, max=-5.826450433232822E-13, mean=-1.9421501444109404E-13, count=36.0, positive=0, stdDev=2.7466150743908175E-13, zeros=24}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 1.9422e-13 +- 2.7466e-13 [0.0000e+00 - 5.8265e-13] (36#)
    relativeTol: 1.1653e-12 +- NaN [1.1653e-12 - 1.1653e-12] (12#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=1.9422e-13 +- 2.7466e-13 [0.0000e+00 - 5.8265e-13] (36#), relativeTol=1.1653e-12 +- NaN [1.1653e-12 - 1.1653e-12] (12#)}
```



### Json Serialization
Code from [JsonTest.java:36](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/JsonTest.java#L36) executed in 0.00 seconds: 
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
      "id": "568de0f4-ea8b-49e0-85ba-ee9e260379db",
      "isFrozen": false,
      "name": "AvgSubsampleLayer/568de0f4-ea8b-49e0-85ba-ee9e260379db",
      "inner": [
        2,
        2,
        1
      ]
    }
```



### Example Input/Output Pair
Code from [ReferenceIO.java:68](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/ReferenceIO.java#L68) executed in 0.00 seconds: 
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
    	[ [ -1.944, -1.48, -0.816 ], [ 0.688, 0.9, -1.832 ] ],
    	[ [ 0.248, -0.276, 0.428 ], [ -0.9, -0.756, -0.688 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 0.0, 0.0, -1.6070000000000002 ] ]
    ]
    --------------------
    Derivative: 
    [
    	[ [ 0.25, 0.25, 0.25 ], [ 0.25, 0.25, 0.25 ] ],
    	[ [ 0.25, 0.25, 0.25 ], [ 0.25, 0.25, 0.25 ] ]
    ]
```



### Input Learning
In this test, we use a network to learn this target input, given it's pre-evaluated output:

Code from [LearningTester.java:127](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L127) executed in 0.01 seconds: 
```java
    return Arrays.stream(input_target).map(x -> x.prettyPrint()).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [
    	[ [ -0.708, 0.952, 1.1 ], [ 1.6, -0.708, 0.384 ], [ 1.256, -0.892, 0.476 ], [ -1.32, 1.224, 1.376 ], [ 1.92, -1.528, -0.268 ], [ 0.192, 1.404, 1.516 ], [ 0.272, -1.216, -0.836 ], [ -0.756, -1.136, 1.928 ], ... ],
    	[ [ 0.728, -1.116, 0.676 ], [ -1.536, 0.544, 1.088 ], [ -1.884, 0.732, 1.216 ], [ -0.98, 1.856, -1.164 ], [ -0.156, -1.36, 0.108 ], [ 0.68, -1.02, -1.46 ], [ -1.52, 1.18, -1.196 ], [ 0.416, -0.292, -1.996 ], ... ],
    	[ [ 0.844, -1.232, 1.264 ], [ -0.212, -0.024, 1.168 ], [ -0.216, -1.756, 0.528 ], [ -0.788, -0.124, -0.2 ], [ 1.528, -0.356, -0.704 ], [ -1.816, -1.58, 1.904 ], [ 1.02, 0.196, -0.36 ], [ 1.836, 1.404, -1.408 ], ... ],
    	[ [ -1.56, 1.212, -1.14 ], [ -0.272, -1.64, -0.928 ], [ -1.808, 1.028, 0.76 ], [ 0.692, -0.164, -1.992 ], [ -1.96, -1.412, 1.932 ], [ 0.444, -0.748, 0.804 ], [ 1.428, 0.472, -0.892 ], [ -0.004, 0.588, -1.14 ], ... ],
    	[ [ 1.944, -0.084, -0.08 ], [ 1.18, 1.612, -1.116 ], [ -0.048, -1.356, -0.932 ], [ -1.388, -0.856, 0.56 ], [ 0.78, 0.904, -0.676 ], [ 1.872, 1.872, 1.316 ], [ -1.604, -0.08, 0.768 ], [ -0.632, -0.312, 1.604 ], ... ],
    	[ [ 0.832, 1.628, 0.86 ], [ 1.568, 0.02, -0.5 ], [ -0.636, 0.96, 0.008 ], [ 0.172, -0.412, -1.044 ], [ -1.404, -0.068, -0.636 ], [ 1.42, -1.048, 1.312 ], [ 0.44, -1.7, -1.004 ], [ 1.108, -1.156, 0.936 ], ... ],
    	[ [ -0.228, 0.3, -0.732 ], [ 1.464, -1.14, -1.62 ], [ 0.296, 0.524, -0.852 ], [ 0.58, -0.052, 0.792 ], [ 1.816, 0.432, -0.788 ], [ -0.176, 1.252, 1.452 ], [ 0.892, -1.656, 1.836 ], [ 1.588, 0.74, 1.788 ], ... ],
    	[ [ -1.412, -1.616, 1.568 ], [ -0.568, -0.74, -1.86 ], [ 0.72, -0.432, 0.088 ], [ 0.12, -1.268, -1.312 ], [ 1.756, -0.096, -1.12 ], [ 1.492, -1.336, 0.644 ], [ -0.268, 1.956, -0.548 ], [ 0.932, -1.836, 1.9 ], ... ],
    	...
    ]
```



