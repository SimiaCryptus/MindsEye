# CrossProductLayer
## CrossProductLayerTest
### Json Serialization
Code from [LayerTestBase.java:83](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L83) executed in 0.00 seconds: 
```java
    JsonObject json = layer.getJson();
    NNLayer echo = NNLayer.fromJson(json);
    assert (echo != null) : "Failed to deserialize";
    assert (layer != echo) : "Serialization did not copy";
    Assert.assertEquals("Serialization not equal", layer, echo);
    return new GsonBuilder().setPrettyPrinting().create().toJson(json);
```

Returns: 

```
    {
      "class": "com.simiacryptus.mindseye.layers.java.CrossProductLayer",
      "id": "370a9587-74a1-4959-b406-fa4500002b9e",
      "isFrozen": false,
      "name": "CrossProductLayer/370a9587-74a1-4959-b406-fa4500002b9e"
    }
```



### Example Input/Output Pair
Code from [LayerTestBase.java:120](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L120) executed in 0.00 seconds: 
```java
    SimpleEval eval = SimpleEval.run(layer, inputPrototype);
    return String.format("--------------------\nInput: \n[%s]\n--------------------\nOutput: \n%s",
      Arrays.stream(inputPrototype).map(t->t.prettyPrint()).reduce((a,b)->a+",\n"+b).get(),
      eval.getOutput().prettyPrint());
```

Returns: 

```
    --------------------
    Input: 
    [[ -1.58, 1.004, 0.004, -0.268 ]]
    --------------------
    Output: 
    [ -1.5863200000000002, -0.00632, 0.42344000000000004, 0.0040160000000000005, -0.26907200000000003, -0.001072 ]
```



### Batch Execution
Code from [LayerTestBase.java:138](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L138) executed in 0.00 seconds: 
```java
    BatchingTester batchingTester = getBatchingTester();
    return batchingTester==null?null:batchingTester.test(layer, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (100#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (100#)}
```



### Differential Validation
Code from [LayerTestBase.java:144](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.00 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [ -1.58, 1.004, 0.004, -0.268 ]
    Inputs Statistics: {meanExponent=-0.6923536037199564, negative=2, min=-0.268, max=-0.268, mean=-0.21000000000000002, count=4.0, positive=2, stdDev=0.9219349217813587, zeros=0}
    Output: [ -1.5863200000000002, -0.00632, 0.42344000000000004, 0.0040160000000000005, -0.26907200000000003, -0.001072 ]
    Outputs Statistics: {meanExponent=-1.384707207439913, negative=4, min=-0.001072, max=-0.001072, mean=-0.23922133333333337, count=6.0, positive=2, stdDev=0.6357159296275516, zeros=0}
    Feedback for input 0
    Inputs Values: [ -1.58, 1.004, 0.004, -0.268 ]
    Value Statistics: {meanExponent=-0.6923536037199564, negative=2, min=-0.268, max=-0.268, mean=-0.21000000000000002, count=4.0, positive=2, stdDev=0.9219349217813587, zeros=0}
    Implemented Feedback: [ [ 1.004, 0.004, -0.268, 0.0, 0.0, 0.0 ], [ -1.58, 0.0, 0.0, 0.004, -0.268, 0.0 ], [ 0.0, -1.58, 0.0, 1.004, 0.0, -0.268 ], [ 0.0, 0.0, -1.58, 0.0, 1.004, 0.004 ] ]
    Implemented Statistics: {meanExponent=-0.6923536037199564, negative=6, min=0.004, max=0.004, mean=-0.105, count=24.0, positive=6, stdDev=0.6603082613446541, zeros=12}
    Measured Feedback: [ [ 1.0040000000000049, 0.003999999999993592, -0.26799999999993496, 0.0, 0.0, 0.0 ], [ -1.5799999999988046, 0.0, 0.0, 0.003999999999993592, -0.26799999999993496, 0.0 ], [ 0.0, -1.5800000000000103, 0.0, 1.0039999999999962, 0.0, -0.2680000000000022 ], [ 0.0, 0.0, -1.5799999999999148, 0.0, 1.0040000000000049, 0.004000000000000097 ] ]
    Measured Statistics: {meanExponent=-0.6923536037201176, negative=6, min=0.004000000000000097, max=0.004000000000000097, mean=-0.10499999999994199, count=24.0, positive=6, stdDev=0.6603082613445349, zeros=12}
    Feedback Error: [ [ 4.884981308350689E-15, -6.4080685202583254E-15, 6.505906924303417E-14, 0.0, 0.0, 0.0 ], [ 1.1954881529163686E-12, 0.0, 0.0, -6.4080685202583254E-15, 6.505906924303417E-14, 0.0 ], [ 0.0, -1.021405182655144E-14, 0.0, -3.774758283725532E-15, 0.0, -2.1649348980190553E-15 ], [ 0.0, 0.0, 8.526512829121202E-14, 0.0, 4.884981308350689E-15, 9.71445146547012E-17 ] ]
    Error Statistics: {meanExponent=-13.955411428946563, negative=5, min=9.71445146547012E-17, max=9.71445146547012E-17, mean=5.799036019900801E-14, count=24.0, positive=7, stdDev=2.384386181168421E-13, zeros=12}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 6.0405e-14 +- 2.3784e-13 [0.0000e+00 - 1.1955e-12] (24#)
    relativeTol: 1.8969e-13 +- 2.9226e-13 [1.8799e-15 - 8.0101e-13] (12#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=6.0405e-14 +- 2.3784e-13 [0.0000e+00 - 1.1955e-12] (24#), relativeTol=1.8969e-13 +- 2.9226e-13 [1.8799e-15 - 8.0101e-13] (12#)}
```



### Performance
Code from [LayerTestBase.java:149](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L149) executed in 0.00 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 0.2073 +- 0.0585 [0.1396 - 0.4731]
    Learning performance: 0.0038 +- 0.0020 [0.0000 - 0.0142]
    
```

