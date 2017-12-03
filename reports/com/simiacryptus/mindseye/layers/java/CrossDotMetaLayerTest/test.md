# CrossDotMetaLayer
## CrossDotMetaLayerTest
### Json Serialization
Code from [LayerTestBase.java:84](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L84) executed in 0.00 seconds: 
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
      "id": "e2d0bffa-47dc-4875-864f-3d3d00001483",
      "isFrozen": false,
      "name": "CrossDotMetaLayer/e2d0bffa-47dc-4875-864f-3d3d00001483"
    }
```



### Example Input/Output Pair
Code from [LayerTestBase.java:121](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L121) executed in 0.00 seconds: 
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
    [[ -1.504, -0.284, 1.168 ]]
    --------------------
    Output: 
    [ [ 0.0, 0.42713599999999996, -1.7566719999999998 ], [ 0.42713599999999996, 0.0, -0.33171199999999995 ], [ -1.7566719999999998, -0.33171199999999995, 0.0 ] ]
```



### Differential Validation
Code from [LayerTestBase.java:139](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L139) executed in 0.00 seconds: 
```java
    getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Feedback for input 0
    Inputs: [ -1.504, -0.284, 1.168 ]
    Output: [ [ 0.0, 0.42713599999999996, -1.7566719999999998 ], [ 0.42713599999999996, 0.0, -0.33171199999999995 ], [ -1.7566719999999998, -0.33171199999999995, 0.0 ] ]
    Measured: [ [ 0.0, -0.28399999999983994, 1.1679999999980595, -0.28399999999983994, 0.0, 0.0, 1.1679999999980595, 0.0, 0.0 ], [ 0.0, -1.5039999999999498, 0.0, -1.5039999999999498, 0.0, 1.1679999999997248, 0.0, 1.1679999999997248, 0.0 ], [ 0.0, 0.0, -1.50400000000106, 0.0, 0.0, -0.28399999999983994, -1.50400000000106, -0.28399999999983994, 0.0 ] ]
    Implemented: [ [ 0.0, -0.284, 1.168, -0.284, 0.0, 0.0, 1.168, 0.0, 0.0 ], [ 0.0, -1.504, 0.0, -1.504, 0.0, 1.168, 0.0, 1.168, 0.0 ], [ 0.0, 0.0, -1.504, 0.0, 0.0, -0.284, -1.504, -0.284, 0.0 ] ]
    Error: [ [ 0.0, 1.6003864899971632E-13, -1.9404478024398486E-12, 1.6003864899971632E-13, 0.0, 0.0, -1.9404478024398486E-12, 0.0, 0.0 ], [ 0.0, 5.0182080713057076E-14, 0.0, 5.0182080713057076E-14, 0.0, -2.751132655021138E-13, 0.0, -2.751132655021138E-13, 0.0 ], [ 0.0, 0.0, -1.0600409439120995E-12, 0.0, 0.0, 1.6003864899971632E-13, -1.0600409439120995E-12, 1.6003864899971632E-13, 0.0 ] ]
    Finite-Difference Derivative Accuracy:
    absoluteTol: 2.7006e-13 +- 5.4663e-13 [0.0000e+00 - 1.9404e-12] (27#)
    relativeTol: 3.1351e-13 +- 2.5736e-13 [1.6683e-14 - 8.3067e-13] (12#)
    
```

### Performance
Code from [LayerTestBase.java:144](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.00 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 0.1567 +- 0.0339 [0.1083 - 0.3021]
    Learning performance: 0.0021 +- 0.0023 [0.0000 - 0.0142]
    
```

