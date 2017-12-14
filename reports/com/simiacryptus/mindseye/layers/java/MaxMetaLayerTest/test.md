# MaxMetaLayer
## MaxMetaLayerTest
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
      "class": "com.simiacryptus.mindseye.layers.java.MaxMetaLayer",
      "id": "d3f216fc-7899-440f-953a-a99eab25da20",
      "isFrozen": false,
      "name": "MaxMetaLayer/d3f216fc-7899-440f-953a-a99eab25da20"
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
    [[ -1.516, -1.536, 1.476 ]]
    --------------------
    Output: 
    [ -1.516, -1.536, 1.476 ]
    --------------------
    Derivative: 
    [ 1.0, 1.0, 1.0 ]
```



### Differential Validation
Code from [StandardLayerTests.java:109](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L109) executed in 0.00 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [ -1.592, -0.88, 0.356 ],
    [ -1.076, -0.184, 0.544 ],
    [ -0.332, 1.072, 1.204 ]
    Inputs Statistics: {meanExponent=-0.100708088825102, negative=2, min=0.356, max=0.356, mean=-0.7053333333333334, count=3.0, positive=1, stdDev=0.8048011486627544, zeros=0},
    {meanExponent=-0.3225903353206377, negative=2, min=0.544, max=0.544, mean=-0.23866666666666667, count=3.0, positive=1, stdDev=0.6624909223696747, zeros=0},
    {meanExponent=-0.12268021467246891, negative=1, min=1.204, max=1.204, mean=0.648, count=3.0, positive=2, stdDev=0.6950568322087052, zeros=0}
    Output: [ -1.592, -0.88, 0.356 ]
    Outputs Statistics: {meanExponent=-0.100708088825102, negative=2, min=0.356, max=0.356, mean=-0.7053333333333334, count=3.0, positive=1, stdDev=0.8048011486627544, zeros=0}
    Feedback for input 0
    Inputs Values: [ -1.592, -0.88, 0.356 ]
    Value Statistics: {meanExponent=-0.100708088825102, negative=2, min=0.356, max=0.356, mean=-0.7053333333333334, count=3.0, positive=1, stdDev=0.8048011486627544, zeros=0}
    Implemented Feedback: [
```
...[skipping 1749 bytes](etc/125.txt)...
```
    
    Implemented Statistics: {meanExponent=0.0, negative=0, min=1.0, max=1.0, mean=0.3333333333333333, count=9.0, positive=3, stdDev=0.4714045207910317, zeros=6}
    Measured Feedback: [ [ 0.9999999999998899, 0.0, 0.0 ], [ 0.0, 0.9999999999998899, 0.0 ], [ 0.0, 0.0, 0.9999999999998899 ] ]
    Measured Statistics: {meanExponent=-4.7830642341045674E-14, negative=0, min=0.9999999999998899, max=0.9999999999998899, mean=0.3333333333332966, count=9.0, positive=3, stdDev=0.4714045207909798, zeros=6}
    Feedback Error: [ [ -1.1013412404281553E-13, 0.0, 0.0 ], [ 0.0, -1.1013412404281553E-13, 0.0 ], [ 0.0, 0.0, -1.1013412404281553E-13 ] ]
    Error Statistics: {meanExponent=-12.958078098036827, negative=3, min=-1.1013412404281553E-13, max=-1.1013412404281553E-13, mean=-3.671137468093851E-14, count=9.0, positive=0, stdDev=5.1917723967143496E-14, zeros=6}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 1.2237e-14 +- 3.4612e-14 [0.0000e+00 - 1.1013e-13] (27#)
    relativeTol: 5.5067e-14 +- 0.0000e+00 [5.5067e-14 - 5.5067e-14] (3#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=1.2237e-14 +- 3.4612e-14 [0.0000e+00 - 1.1013e-13] (27#), relativeTol=5.5067e-14 +- 0.0000e+00 [5.5067e-14 - 5.5067e-14] (3#)}
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
    	Evaluation performance: 0.000162s +- 0.000082s [0.000061s - 0.000255s]
    	Learning performance: 0.000004s +- 0.000002s [0.000002s - 0.000006s]
    
```

