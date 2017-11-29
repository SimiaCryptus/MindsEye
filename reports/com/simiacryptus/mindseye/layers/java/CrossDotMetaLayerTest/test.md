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
      "id": "c88cbdf1-1c2a-4a5e-b964-890900000f69",
      "isFrozen": false,
      "name": "CrossDotMetaLayer/c88cbdf1-1c2a-4a5e-b964-890900000f69"
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
    [[ -1.032, -1.212, -1.54 ]]
    --------------------
    Output: 
    [ [ 0.0, 1.250784, 1.58928 ], [ 1.250784, 0.0, 1.86648 ], [ 1.58928, 1.86648, 0.0 ] ]
```



### Differential Validation
Code from [LayerTestBase.java:139](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L139) executed in 0.00 seconds: 
```java
    getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Component: CrossDotMetaLayer/c88cbdf1-1c2a-4a5e-b964-890900000f69
    Inputs: [ -1.032, -1.212, -1.54 ]
    output=[ [ 0.0, 1.250784, 1.58928 ], [ 1.250784, 0.0, 1.86648 ], [ 1.58928, 1.86648, 0.0 ] ]
    measured/actual: [ [ 0.0, -1.2119999999993247, -1.5399999999998748, -1.2119999999993247, 0.0, 0.0, -1.5399999999998748, 0.0, 0.0 ], [ 0.0, -1.0319999999985896, 0.0, -1.0319999999985896, 0.0, -1.5399999999998748, 0.0, -1.5399999999998748, 0.0 ], [ 0.0, 0.0, -1.0319999999985896, 0.0, 0.0, -1.2119999999993247, -1.0319999999985896, -1.2119999999993247, 0.0 ] ]
    implemented/expected: [ [ 0.0, -1.212, -1.54, -1.212, 0.0, 0.0, -1.54, 0.0, 0.0 ], [ 0.0, -1.032, 0.0, -1.032, 0.0, -1.54, 0.0, -1.54, 0.0 ], [ 0.0, 0.0, -1.032, 0.0, 0.0, -1.212, -1.032, -1.212, 0.0 ] ]
    error: [ [ 0.0, 6.752376435770202E-13, 1.2523315717771766E-13, 6.752376435770202E-13, 0.0, 0.0, 1.2523315717771766E-13, 0.0, 0.0 ], [ 0.0, 1.4104273304837989E-12, 0.0, 1.4104273304837989E-12, 0.0, 1.2523315717771766E-13, 0.0, 1.2523315717771766E-13, 0.0 ], [ 0.0, 0.0, 1.4104273304837989E-12, 0.0, 0.0, 6.752376435770202E-13, 1.4104273304837989E-12, 6.752376435770202E-13, 0.0 ] ]
    Finite-Difference Derivative Accuracy:
    absoluteTol: 3.2754e-13 +- 5.0725e-13 [0.0000e+00 - 1.4104e-12] (27#)
    relativeTol: 3.3419e-13 +- 2.6531e-13 [4.0660e-14 - 6.8335e-13] (12#)
    
```

### Performance
Code from [LayerTestBase.java:144](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.00 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 0.1460 +- 0.0325 [0.1054 - 0.3021]
    Learning performance: 0.0037 +- 0.0021 [0.0000 - 0.0142]
    
```

