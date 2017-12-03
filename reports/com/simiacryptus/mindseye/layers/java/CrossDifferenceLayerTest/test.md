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
      "id": "e2d0bffa-47dc-4875-864f-3d3d0000147c",
      "isFrozen": false,
      "name": "CrossDifferenceLayer/e2d0bffa-47dc-4875-864f-3d3d0000147c"
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
    [[ 1.82, 0.92, -1.892, 1.068 ]]
    --------------------
    Output: 
    [ 0.9, 3.7119999999999997, 0.752, 2.812, -0.14800000000000002, -2.96 ]
```



### Differential Validation
Code from [LayerTestBase.java:139](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L139) executed in 0.00 seconds: 
```java
    getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Feedback for input 0
    Inputs: [ 1.82, 0.92, -1.892, 1.068 ]
    Output: [ 0.9, 3.7119999999999997, 0.752, 2.812, -0.14800000000000002, -2.96 ]
    Measured: [ [ 0.9999999999998899, 1.0000000000021103, 0.9999999999998899, 0.0, 0.0, 0.0 ], [ -0.9999999999998899, 0.0, 0.0, 1.0000000000021103, 0.9999999999998899, 0.0 ], [ 0.0, -0.9999999999976694, 0.0, -0.9999999999976694, 0.0, 0.9999999999976694 ], [ 0.0, 0.0, -0.9999999999998899, 0.0, -0.9999999999998899, -0.9999999999976694 ] ]
    Implemented: [ [ 1.0, 1.0, 1.0, 0.0, 0.0, 0.0 ], [ -1.0, 0.0, 0.0, 1.0, 1.0, 0.0 ], [ 0.0, -1.0, 0.0, -1.0, 0.0, 1.0 ], [ 0.0, 0.0, -1.0, 0.0, -1.0, -1.0 ] ]
    Error: [ [ -1.1013412404281553E-13, 2.1103119252074976E-12, -1.1013412404281553E-13, 0.0, 0.0, 0.0 ], [ 1.1013412404281553E-13, 0.0, 0.0, 2.1103119252074976E-12, -1.1013412404281553E-13, 0.0 ], [ 0.0, 2.3305801732931286E-12, 0.0, 2.3305801732931286E-12, 0.0, -2.3305801732931286E-12 ], [ 0.0, 0.0, 1.1013412404281553E-13, 0.0, 1.1013412404281553E-13, 2.3305801732931286E-12 ] ]
    Finite-Difference Derivative Accuracy:
    absoluteTol: 5.9182e-13 +- 9.6393e-13 [0.0000e+00 - 2.3306e-12] (24#)
    relativeTol: 5.9182e-13 +- 5.3801e-13 [5.5067e-14 - 1.1653e-12] (12#)
    
```

### Performance
Code from [LayerTestBase.java:144](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.00 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 0.2611 +- 0.1042 [0.1339 - 0.7808]
    Learning performance: 0.0033 +- 0.0031 [0.0000 - 0.0257]
    
```

