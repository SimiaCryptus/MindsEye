# CrossProductLayer
## CrossProductLayerTest
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
      "class": "com.simiacryptus.mindseye.layers.java.CrossProductLayer",
      "id": "37b116d0-6d07-4e99-8aa6-3cd26d417c77",
      "isFrozen": false,
      "name": "CrossProductLayer/37b116d0-6d07-4e99-8aa6-3cd26d417c77"
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
    [[ 1.736, 0.16, -0.936, 1.768 ]]
    --------------------
    Output: 
    [ 0.27776, -1.6248960000000001, 3.069248, -0.14976, 0.28288, -1.654848 ]
    --------------------
    Derivative: 
    [ 0.992, 2.568, 3.6639999999999997, 0.9599999999999999 ]
```



### Batch Execution
Code from [StandardLayerTests.java:102](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L102) executed in 0.00 seconds: 
```java
    return getBatchingTester().test(layer, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (100#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (100#)}
```



### Differential Validation
Code from [StandardLayerTests.java:110](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L110) executed in 0.00 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [ 0.748, -1.048, -1.976, -1.696 ]
    Inputs Statistics: {meanExponent=0.10486891717111838, negative=3, min=-1.696, max=-1.696, mean=-0.9929999999999999, count=4.0, positive=1, stdDev=1.0600240563307988, zeros=0}
    Output: [ -0.783904, -1.478048, -1.268608, 2.0708480000000002, 1.777408, 3.351296 ]
    Outputs Statistics: {meanExponent=0.20973783434223678, negative=3, min=3.351296, max=3.351296, mean=0.6114986666666667, count=6.0, positive=3, stdDev=1.863853062446954, zeros=0}
    Feedback for input 0
    Inputs Values: [ 0.748, -1.048, -1.976, -1.696 ]
    Value Statistics: {meanExponent=0.10486891717111838, negative=3, min=-1.696, max=-1.696, mean=-0.9929999999999999, count=4.0, positive=1, stdDev=1.0600240563307988, zeros=0}
    Implemented Feedback: [ [ -1.048, -1.976, -1.696, 0.0, 0.0, 0.0 ], [ 0.748, 0.0, 0.0, -1.976, -1.696, 0.0 ], [ 0.0, 0.748, 0.0, -1.048, 0.0, -1.696 ], [ 0.0, 0.0, 0.748, 0.0, -1.048, -1.976 ] ]
    Implemented Statistics: {meanExponent=0.10486891717111839, negative=9, min=-1.976, max=-1.976, mean
```
...[skipping 373 bytes](etc/64.txt)...
```
    .0480000000012701, -1.9760000000035305 ] ]
    Measured Statistics: {meanExponent=0.10486891717134865, negative=9, min=-1.9760000000035305, max=-1.9760000000035305, mean=-0.49650000000049194, count=24.0, positive=3, stdDev=0.8990760535131939, zeros=12}
    Feedback Error: [ [ -1.5987211554602254E-13, 9.103828801926284E-13, 8.01581023779363E-14, 0.0, 0.0, 0.0 ], [ 4.150013666048835E-13, 0.0, 0.0, -3.530509218307998E-12, 8.01581023779363E-14, 0.0 ], [ 0.0, -6.95221658020273E-13, 0.0, -3.490541189421492E-12, 0.0, 8.01581023779363E-14 ], [ 0.0, 0.0, -6.95221658020273E-13, 0.0, -1.270095140171179E-12, -3.530509218307998E-12 ] ]
    Error Statistics: {meanExponent=-12.256705089975526, negative=7, min=-3.530509218307998E-12, max=-3.530509218307998E-12, mean=-4.919213184943297E-13, count=24.0, positive=5, stdDev=1.2064213025680793E-12, zeros=12}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 6.2241e-13 +- 1.1446e-12 [0.0000e+00 - 3.5305e-12] (24#)
    relativeTol: 4.7020e-13 +- 4.7020e-13 [2.3632e-14 - 1.6653e-12] (12#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=6.2241e-13 +- 1.1446e-12 [0.0000e+00 - 3.5305e-12] (24#), relativeTol=4.7020e-13 +- 4.7020e-13 [2.3632e-14 - 1.6653e-12] (12#)}
```



### Performance
Code from [StandardLayerTests.java:120](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L120) executed in 0.00 seconds: 
```java
    getPerformanceTester().test(layer, permPrototype);
```
Logging: 
```
    Evaluation performance: 0.000239s +- 0.000064s [0.000178s - 0.000353s]
    Learning performance: 0.000042s +- 0.000005s [0.000038s - 0.000051s]
    
```

