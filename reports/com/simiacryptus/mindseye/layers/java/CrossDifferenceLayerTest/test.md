# CrossDifferenceLayer
## CrossDifferenceLayerTest
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
      "class": "com.simiacryptus.mindseye.layers.java.CrossDifferenceLayer",
      "id": "c88cbdf1-1c2a-4a5e-b964-890900000f68",
      "isFrozen": false,
      "name": "CrossDifferenceLayer/c88cbdf1-1c2a-4a5e-b964-890900000f68"
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
    [[ 0.52, 1.816, 1.976, -0.892 ]]
    --------------------
    Output: 
    [ -1.296, -1.456, 1.412, -0.15999999999999992, 2.708, 2.868 ]
```



### Differential Validation
Code from [LayerTestBase.java:139](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L139) executed in 0.00 seconds: 
```java
    getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Component: CrossDifferenceLayer/c88cbdf1-1c2a-4a5e-b964-890900000f68
    Inputs: [ 0.52, 1.816, 1.976, -0.892 ]
    output=[ -1.296, -1.456, 1.412, -0.15999999999999992, 2.708, 2.868 ]
    measured/actual: [ [ 0.9999999999998899, 0.9999999999998899, 1.0000000000021103, 0.0, 0.0, 0.0 ], [ -0.9999999999998899, 0.0, 0.0, 0.9999999999998899, 0.9999999999976694, 0.0 ], [ 0.0, -0.9999999999998899, 0.0, -0.9999999999998899, 0.0, 1.0000000000021103 ], [ 0.0, 0.0, -0.9999999999976694, 0.0, -1.0000000000021103, -0.9999999999976694 ] ]
    implemented/expected: [ [ 1.0, 1.0, 1.0, 0.0, 0.0, 0.0 ], [ -1.0, 0.0, 0.0, 1.0, 1.0, 0.0 ], [ 0.0, -1.0, 0.0, -1.0, 0.0, 1.0 ], [ 0.0, 0.0, -1.0, 0.0, -1.0, -1.0 ] ]
    error: [ [ -1.1013412404281553E-13, -1.1013412404281553E-13, 2.1103119252074976E-12, 0.0, 0.0, 0.0 ], [ 1.1013412404281553E-13, 0.0, 0.0, -1.1013412404281553E-13, -2.3305801732931286E-12, 0.0 ], [ 0.0, 1.1013412404281553E-13, 0.0, 1.1013412404281553E-13, 0.0, 2.1103119252074976E-12 ], [ 0.0, 0.0, 2.3305801732931286E-12, 0.0, -2.1103119252074976E-12, 2.3305801732931286E-12 ] ]
    Finite-Difference Derivative Accuracy:
    absoluteTol: 5.8265e-13 +- 9.4825e-13 [0.0000e+00 - 2.3306e-12] (24#)
    relativeTol: 5.8265e-13 +- 5.2901e-13 [5.5067e-14 - 1.1653e-12] (12#)
    
```

### Performance
Code from [LayerTestBase.java:144](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.00 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 0.2254 +- 0.0925 [0.1368 - 0.8122]
    Learning performance: 0.0037 +- 0.0029 [0.0000 - 0.0200]
    
```

