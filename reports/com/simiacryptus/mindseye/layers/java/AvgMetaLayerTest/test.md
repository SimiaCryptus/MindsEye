# AvgMetaLayer
## AvgMetaLayerTest
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
      "class": "com.simiacryptus.mindseye.layers.java.AvgMetaLayer",
      "id": "9fed18ff-da3c-41c8-9ee9-a43785038719",
      "isFrozen": false,
      "name": "AvgMetaLayer/9fed18ff-da3c-41c8-9ee9-a43785038719",
      "minBatchCount": 0
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
    [[ -0.004, -0.036, -0.48 ]]
    --------------------
    Output: 
    [ -0.004, -0.036, -0.48 ]
    --------------------
    Derivative: 
    [ 1.0, 1.0, 1.0 ]
```



### Differential Validation
Code from [StandardLayerTests.java:110](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L110) executed in 0.02 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [ 1.98, 1.46, -1.88 ],
    [ -0.112, 0.352, -0.544 ],
    [ 1.196, 0.38, 1.688 ]
    Inputs Statistics: {meanExponent=0.24505863176988266, negative=1, min=-1.88, max=-1.88, mean=0.52, count=3.0, positive=2, stdDev=1.7102826277158598, zeros=0},
    {meanExponent=-0.5562134713845025, negative=2, min=-0.544, max=-0.544, mean=-0.10133333333333334, count=3.0, positive=1, stdDev=0.36586822157103993, zeros=0},
    {meanExponent=-0.03837092714705386, negative=0, min=1.688, max=1.688, mean=1.088, count=3.0, positive=3, stdDev=0.5394219127918328, zeros=0}
    Output: [ 1.98, 1.46, -1.88 ]
    Outputs Statistics: {meanExponent=0.24505863176988266, negative=1, min=-1.88, max=-1.88, mean=0.52, count=3.0, positive=2, stdDev=1.7102826277158598, zeros=0}
    Feedback for input 0
    Inputs Values: [ 1.98, 1.46, -1.88 ]
    Value Statistics: {meanExponent=0.24505863176988266, negative=1, min=-1.88, max=-1.88, mean=0.52, count=3.0, positive=2, stdDev=1.7102826277158598, zeros=0}
    Implemented Feedback: [ [ 0.3333333333333333, 0.0, 0.0 ], [ 0.0, 0.333333
```
...[skipping 2579 bytes](etc/55.txt)...
```
    .47712125471966244, negative=0, min=0.3333333333333333, max=0.3333333333333333, mean=0.1111111111111111, count=9.0, positive=3, stdDev=0.15713484026367722, zeros=6}
    Measured Feedback: [ [ 0.3333333333310762, 0.0, 0.0 ], [ 0.0, 0.3333333333332966, 0.0 ], [ 0.0, 0.0, 0.3333333333332966 ] ]
    Measured Statistics: {meanExponent=-0.4771212547206746, negative=0, min=0.3333333333332966, max=0.3333333333332966, mean=0.11111111111085216, count=9.0, positive=3, stdDev=0.157134840263311, zeros=6}
    Feedback Error: [ [ -2.2571389202141745E-12, 0.0, 0.0 ], [ 0.0, -3.6692870963861424E-14, 0.0 ], [ 0.0, 0.0, -3.6692870963861424E-14 ] ]
    Error Statistics: {meanExponent=-12.839092774433162, negative=3, min=-3.6692870963861424E-14, max=-3.6692870963861424E-14, mean=-2.58947184682433E-13, count=9.0, positive=0, stdDev=7.066262597417965E-13, zeros=6}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 2.5895e-13 +- 7.0663e-13 [0.0000e+00 - 2.2571e-12] (27#)
    relativeTol: 1.1653e-12 +- 1.5701e-12 [5.5039e-14 - 3.3857e-12] (9#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=2.5895e-13 +- 7.0663e-13 [0.0000e+00 - 2.2571e-12] (27#), relativeTol=1.1653e-12 +- 1.5701e-12 [5.5039e-14 - 3.3857e-12] (9#)}
```



### Performance
Code from [StandardLayerTests.java:120](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L120) executed in 0.00 seconds: 
```java
    getPerformanceTester().test(layer, permPrototype);
```
Logging: 
```
    Evaluation performance: 0.000212s +- 0.000135s [0.000112s - 0.000460s]
    Learning performance: 0.000006s +- 0.000005s [0.000002s - 0.000013s]
    
```

