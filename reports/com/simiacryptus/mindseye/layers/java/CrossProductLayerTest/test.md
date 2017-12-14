# CrossProductLayer
## CrossProductLayerTest
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
      "class": "com.simiacryptus.mindseye.layers.java.CrossProductLayer",
      "id": "f5b679ed-4846-4d6a-b50c-3b9eec8ac72f",
      "isFrozen": false,
      "name": "CrossProductLayer/f5b679ed-4846-4d6a-b50c-3b9eec8ac72f"
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
    [[ -0.956, -1.54, 0.432, 1.52 ]]
    --------------------
    Output: 
    [ 1.47224, -0.41299199999999997, -1.45312, -0.66528, -2.3408, 0.65664 ]
    --------------------
    Derivative: 
    [ 0.4119999999999999, 0.996, -0.976, -2.064 ]
```



### Batch Execution
Code from [StandardLayerTests.java:101](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L101) executed in 0.00 seconds: 
```java
    return getBatchingTester().test(layer, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (100#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (100#)}
```



### Differential Validation
Code from [StandardLayerTests.java:109](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L109) executed in 0.00 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [ -1.008, 1.224, -1.784, 1.8 ]
    Inputs Statistics: {meanExponent=0.1494773262656148, negative=2, min=1.8, max=1.8, mean=0.057999999999999996, count=4.0, positive=2, stdDev=1.4936063738482104, zeros=0}
    Output: [ -1.233792, 1.798272, -1.8144, -2.183616, 2.2032, -3.2112000000000003 ]
    Outputs Statistics: {meanExponent=0.2989546525312296, negative=4, min=-3.2112000000000003, max=-3.2112000000000003, mean=-0.740256, count=6.0, positive=2, stdDev=2.028721809723551, zeros=0}
    Feedback for input 0
    Inputs Values: [ -1.008, 1.224, -1.784, 1.8 ]
    Value Statistics: {meanExponent=0.1494773262656148, negative=2, min=1.8, max=1.8, mean=0.057999999999999996, count=4.0, positive=2, stdDev=1.4936063738482104, zeros=0}
    Implemented Feedback: [ [ 1.224, -1.784, 1.8, 0.0, 0.0, 0.0 ], [ -1.008, 0.0, 0.0, -1.784, 1.8, 0.0 ], [ 0.0, -1.008, 0.0, 1.224, 0.0, 1.8 ], [ 0.0, 0.0, -1.008, 0.0, 1.224, -1.784 ] ]
    Implemented Statistics: {meanExponent=0.1494773262656148, negative=6, min=-1.784, max=-1.784, mean=0.0289999999999999
```
...[skipping 372 bytes](etc/105.txt)...
```
    96698, -1.7839999999980094 ] ]
    Measured Statistics: {meanExponent=0.14947732626569868, negative=6, min=-1.7839999999980094, max=-1.7839999999980094, mean=0.029000000000093767, count=24.0, positive=6, stdDev=1.0565372686283572, zeros=12}
    Feedback Error: [ [ -3.3018032752352156E-13, -2.298161660974074E-13, -1.5305534617482408E-12, 0.0, 0.0, 0.0 ], [ -1.199040866595169E-13, 0.0, 0.0, -2.4502622153477205E-12, 2.9103386367523854E-12, 0.0 ], [ 0.0, -1.199040866595169E-13, 0.0, -3.3018032752352156E-13, 0.0, 2.9103386367523854E-12 ], [ 0.0, 0.0, -1.199040866595169E-13, 0.0, -3.3018032752352156E-13, 1.9906298831529057E-12 ] ]
    Error Statistics: {meanExponent=-12.25374377089567, negative=9, min=1.9906298831529057E-12, max=1.9906298831529057E-12, mean=9.376758628813302E-14, count=24.0, positive=3, stdDev=1.1079469575263047E-12, zeros=12}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 5.5717e-13 +- 9.6223e-13 [0.0000e+00 - 2.9103e-12] (24#)
    relativeTol: 3.2784e-13 +- 2.9585e-13 [5.9476e-14 - 8.0843e-13] (12#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=5.5717e-13 +- 9.6223e-13 [0.0000e+00 - 2.9103e-12] (24#), relativeTol=3.2784e-13 +- 2.9585e-13 [5.9476e-14 - 8.0843e-13] (12#)}
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
    	[4]
    Performance:
    	Evaluation performance: 0.000194s +- 0.000025s [0.000156s - 0.000225s]
    	Learning performance: 0.000041s +- 0.000004s [0.000038s - 0.000048s]
    
```

