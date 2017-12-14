# CrossDotMetaLayer
## CrossDotMetaLayerTest
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
      "class": "com.simiacryptus.mindseye.layers.java.CrossDotMetaLayer",
      "id": "e4de1448-7afa-4cf1-bea5-b882ca2cb27e",
      "isFrozen": false,
      "name": "CrossDotMetaLayer/e4de1448-7afa-4cf1-bea5-b882ca2cb27e"
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
    [[ 0.908, 0.992, 1.38 ]]
    --------------------
    Output: 
    [ [ 0.0, 0.900736, 1.25304 ], [ 0.900736, 0.0, 1.36896 ], [ 1.25304, 1.36896, 0.0 ] ]
    --------------------
    Derivative: 
    [ 4.744, 4.576, 3.8 ]
```



### Differential Validation
Code from [StandardLayerTests.java:109](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L109) executed in 0.00 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [ 0.52, -0.6, -1.484 ]
    Inputs Statistics: {meanExponent=-0.11147050167951633, negative=2, min=-1.484, max=-1.484, mean=-0.5213333333333333, count=3.0, positive=1, stdDev=0.8200184279772146, zeros=0}
    Output: [ [ 0.0, -0.312, -0.77168 ], [ -0.312, 0.0, 0.8904 ], [ -0.77168, 0.8904, 0.0 ] ]
    Outputs Statistics: {meanExponent=-0.2229410033590327, negative=4, min=0.0, max=0.0, mean=-0.04295111111111113, count=9.0, positive=2, stdDev=0.5729736452431384, zeros=3}
    Feedback for input 0
    Inputs Values: [ 0.52, -0.6, -1.484 ]
    Value Statistics: {meanExponent=-0.11147050167951633, negative=2, min=-1.484, max=-1.484, mean=-0.5213333333333333, count=3.0, positive=1, stdDev=0.8200184279772146, zeros=0}
    Implemented Feedback: [ [ 0.0, -0.6, -1.484, -0.6, 0.0, 0.0, -1.484, 0.0, 0.0 ], [ 0.0, 0.52, 0.0, 0.52, 0.0, -1.484, 0.0, -1.484, 0.0 ], [ 0.0, 0.0, 0.52, 0.0, 0.0, -0.6, 0.52, -0.6, 0.0 ] ]
    Implemented Statistics: {meanExponent=-0.11147050167951632, negative=8, min=0.0, max=0.0, mean=-0.23170370370370372, count
```
...[skipping 309 bytes](etc/104.txt)...
```
    0, 0.52000000000052, 0.0, 0.0, -0.5999999999994898, 0.52000000000052, -0.5999999999994898, 0.0 ] ]
    Measured Statistics: {meanExponent=-0.1114705016795118, negative=8, min=0.0, max=0.0, mean=-0.23170370370362292, count=27.0, positive=4, stdDev=0.6049513488130509, zeros=15}
    Feedback Error: [ [ 0.0, -4.496403249731884E-14, 6.252776074688882E-13, -4.496403249731884E-14, 0.0, 0.0, 6.252776074688882E-13, 0.0, 0.0 ], [ 0.0, -3.5083047578154947E-14, 0.0, -3.5083047578154947E-14, 0.0, -4.849454171562684E-13, 0.0, -4.849454171562684E-13, 0.0 ], [ 0.0, 0.0, 5.200284647344233E-13, 0.0, 0.0, 5.101474798152594E-13, 5.200284647344233E-13, 5.101474798152594E-13, 0.0 ] ]
    Error Statistics: {meanExponent=-12.649424806226628, negative=6, min=0.0, max=0.0, mean=8.07748929471725E-14, count=27.0, positive=6, stdDev=2.8179253056382405E-13, zeros=15}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 1.6448e-13 +- 2.4265e-13 [0.0000e+00 - 6.2528e-13] (27#)
    relativeTol: 2.2840e-13 +- 1.7859e-13 [3.3734e-14 - 5.0003e-13] (12#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=1.6448e-13 +- 2.4265e-13 [0.0000e+00 - 6.2528e-13] (27#), relativeTol=2.2840e-13 +- 1.7859e-13 [3.3734e-14 - 5.0003e-13] (12#)}
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
    	[3]
    Performance:
    	Evaluation performance: 0.000104s +- 0.000018s [0.000090s - 0.000139s]
    	Learning performance: 0.000003s +- 0.000001s [0.000002s - 0.000006s]
    
```

