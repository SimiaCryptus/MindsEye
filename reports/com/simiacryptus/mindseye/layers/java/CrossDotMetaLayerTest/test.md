# CrossDotMetaLayer
## CrossDotMetaLayerTest
### Json Serialization
Code from [LayerTestBase.java:121](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L121) executed in 0.00 seconds: 
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
      "class": "com.simiacryptus.mindseye.layers.java.CrossDotMetaLayer",
      "id": "e2a3bda5-e7e7-4c05-aeb3-4ede00001e32",
      "isFrozen": false,
      "name": "CrossDotMetaLayer/e2a3bda5-e7e7-4c05-aeb3-4ede00001e32"
    }
```



### Example Input/Output Pair
Code from [LayerTestBase.java:159](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L159) executed in 0.00 seconds: 
```java
    SimpleEval eval = SimpleEval.run(layer, inputPrototype);
    return String.format("--------------------\nInput: \n[%s]\n--------------------\nOutput: \n%s",
      Arrays.stream(inputPrototype).map(t -> t.prettyPrint()).reduce((a, b) -> a + ",\n" + b).get(),
      eval.getOutput().prettyPrint());
```

Returns: 

```
    --------------------
    Input: 
    [[ 1.372, -1.456, 1.304 ]]
    --------------------
    Output: 
    [ [ 0.0, -1.997632, 1.7890880000000002 ], [ -1.997632, 0.0, -1.898624 ], [ 1.7890880000000002, -1.898624, 0.0 ] ]
```



### Differential Validation
Code from [LayerTestBase.java:186](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L186) executed in 0.00 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [ 1.372, -1.456, 1.304 ]
    Inputs Statistics: {meanExponent=0.13859769258121757, negative=1, min=1.304, max=1.304, mean=0.40666666666666673, count=3.0, positive=2, stdDev=1.3173967596066958, zeros=0}
    Output: [ [ 0.0, -1.997632, 1.7890880000000002 ], [ -1.997632, 0.0, -1.898624 ], [ 1.7890880000000002, -1.898624, 0.0 ] ]
    Outputs Statistics: {meanExponent=0.27719538516243514, negative=4, min=0.0, max=0.0, mean=-0.4682595555555555, count=9.0, positive=2, stdDev=1.4764402400807086, zeros=3}
    Feedback for input 0
    Inputs Values: [ 1.372, -1.456, 1.304 ]
    Value Statistics: {meanExponent=0.13859769258121757, negative=1, min=1.304, max=1.304, mean=0.40666666666666673, count=3.0, positive=2, stdDev=1.3173967596066958, zeros=0}
    Implemented Feedback: [ [ 0.0, -1.456, 1.304, -1.456, 0.0, 0.0, 1.304, 0.0, 0.0 ], [ 0.0, 1.372, 0.0, 1.372, 0.0, 1.304, 0.0, 1.304, 0.0 ], [ 0.0, 0.0, 1.372, 0.0, 0.0, -1.456, 1.372, -1.456, 0.0 ] ]
    Implemented Statistics: {meanExponent=0.13859769258121757, negative=4, min=0.0, max=0
```
...[skipping 348 bytes](etc/51.txt)...
```
    719999999994847, 0.0, 0.0, -1.4559999999996798, 1.3719999999994847, -1.4559999999996798, 0.0 ] ]
    Measured Statistics: {meanExponent=0.1385976925811036, negative=4, min=0.0, max=0.0, mean=0.18074074074067478, count=27.0, positive=8, stdDev=0.9012117189727649, zeros=15}
    Feedback Error: [ [ 0.0, 3.2018832030189515E-13, -2.502442697505103E-13, 3.2018832030189515E-13, 0.0, 0.0, -2.502442697505103E-13, 0.0, 0.0 ], [ 0.0, -5.153655280309977E-13, 0.0, -5.153655280309977E-13, 0.0, -2.502442697505103E-13, 0.0, -2.502442697505103E-13, 0.0 ], [ 0.0, 0.0, -5.153655280309977E-13, 0.0, 0.0, 3.2018832030189515E-13, -5.153655280309977E-13, 3.2018832030189515E-13, 0.0 ] ]
    Error Statistics: {meanExponent=-12.461371668898545, negative=8, min=0.0, max=0.0, mean=-6.598836703401671E-14, count=27.0, positive=4, stdDev=2.43843224237039E-13, zeros=15}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 1.6086e-13 +- 1.9478e-13 [0.0000e+00 - 5.1537e-13] (27#)
    relativeTol: 1.3124e-13 +- 4.0411e-14 [9.5953e-14 - 1.8782e-13] (12#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=1.6086e-13 +- 1.9478e-13 [0.0000e+00 - 5.1537e-13] (27#), relativeTol=1.3124e-13 +- 4.0411e-14 [9.5953e-14 - 1.8782e-13] (12#)}
```



### Performance
Code from [LayerTestBase.java:192](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L192) executed in 0.00 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 0.1377 +- 0.0556 [0.0997 - 0.5899]
    Learning performance: 0.0040 +- 0.0027 [0.0000 - 0.0200]
    
```

