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
      "id": "f4569375-56fe-4e46-925c-95f400000990",
      "isFrozen": false,
      "name": "CrossDifferenceLayer/f4569375-56fe-4e46-925c-95f400000990"
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
    [[ -1.14, 0.364, 1.22, -0.728 ]]
    --------------------
    Output: 
    [ -1.504, -2.36, -0.4119999999999999, -0.856, 1.092, 1.948 ]
```



### Differential Validation
Code from [LayerTestBase.java:139](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L139) executed in 0.00 seconds: 
```java
    getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Feedback for input 0
    Inputs: [ -1.14, 0.364, 1.22, -0.728 ]
    Output: [ -1.504, -2.36, -0.4119999999999999, -0.856, 1.092, 1.948 ]
    Measured: [ [ 1.0000000000021103, 1.0000000000021103, 0.9999999999998899, 0.0, 0.0, 0.0 ], [ -0.9999999999976694, 0.0, 0.0, 0.9999999999998899, 0.9999999999976694, 0.0 ], [ 0.0, -1.0000000000021103, 0.0, -0.9999999999998899, 0.0, 0.9999999999998899 ], [ 0.0, 0.0, -0.9999999999998899, 0.0, -1.0000000000021103, -0.9999999999998899 ] ]
    Implemented: [ [ 1.0, 1.0, 1.0, 0.0, 0.0, 0.0 ], [ -1.0, 0.0, 0.0, 1.0, 1.0, 0.0 ], [ 0.0, -1.0, 0.0, -1.0, 0.0, 1.0 ], [ 0.0, 0.0, -1.0, 0.0, -1.0, -1.0 ] ]
    Error: [ [ 2.1103119252074976E-12, 2.1103119252074976E-12, -1.1013412404281553E-13, 0.0, 0.0, 0.0 ], [ 2.3305801732931286E-12, 0.0, 0.0, -1.1013412404281553E-13, -2.3305801732931286E-12, 0.0 ], [ 0.0, -2.1103119252074976E-12, 0.0, 1.1013412404281553E-13, 0.0, -1.1013412404281553E-13 ], [ 0.0, 0.0, 1.1013412404281553E-13, 0.0, -2.1103119252074976E-12, 1.1013412404281553E-13 ] ]
    Finite-Difference Derivative Accuracy:
    absoluteTol: 5.7347e-13 +- 9.3222e-13 [0.0000e+00 - 2.3306e-12] (24#)
    relativeTol: 5.7347e-13 +- 5.1970e-13 [5.5067e-14 - 1.1653e-12] (12#)
    
```

### Performance
Code from [LayerTestBase.java:144](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.01 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 0.2347 +- 0.0923 [0.1482 - 0.6384]
    Learning performance: 0.0044 +- 0.0039 [0.0000 - 0.0314]
    
```

