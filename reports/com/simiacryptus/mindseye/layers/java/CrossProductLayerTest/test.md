# CrossProductLayer
## CrossProductLayerTest
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
      "class": "com.simiacryptus.mindseye.layers.java.CrossProductLayer",
      "id": "e2d0bffa-47dc-4875-864f-3d3d0000148d",
      "isFrozen": false,
      "name": "CrossProductLayer/e2d0bffa-47dc-4875-864f-3d3d0000148d"
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
    [[ -1.968, -1.108, 1.22, -1.116 ]]
    --------------------
    Output: 
    [ 2.1805440000000003, -2.40096, 2.196288, -1.35176, 1.2365280000000003, -1.36152 ]
```



### Differential Validation
Code from [LayerTestBase.java:139](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L139) executed in 0.00 seconds: 
```java
    getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Feedback for input 0
    Inputs: [ -1.968, -1.108, 1.22, -1.116 ]
    Output: [ 2.1805440000000003, -2.40096, 2.196288, -1.35176, 1.2365280000000003, -1.36152 ]
    Measured: [ [ -1.1080000000029955, 1.2200000000017752, -1.1159999999987846, 0.0, 0.0, 0.0 ], [ -1.9679999999988596, 0.0, 0.0, 1.2199999999995548, -1.116000000001005, 0.0 ], [ 0.0, -1.9679999999988596, 0.0, -1.108000000000775, 0.0, -1.116000000001005 ], [ 0.0, 0.0, -1.9679999999988596, 0.0, -1.108000000000775, 1.2199999999995548 ] ]
    Implemented: [ [ -1.108, 1.22, -1.116, 0.0, 0.0, 0.0 ], [ -1.968, 0.0, 0.0, 1.22, -1.116, 0.0 ], [ 0.0, -1.968, 0.0, -1.108, 0.0, -1.116 ], [ 0.0, 0.0, -1.968, 0.0, -1.108, 1.22 ] ]
    Error: [ [ -2.9953817204386723E-12, 1.7752466163756253E-12, 1.2154721673596214E-12, 0.0, 0.0, 0.0 ], [ 1.1404210908949608E-12, 0.0, 0.0, -4.4519943287468777E-13, -1.0049738818906917E-12, 0.0 ], [ 0.0, 1.1404210908949608E-12, 0.0, -7.749356711883593E-13, 0.0, -1.0049738818906917E-12 ], [ 0.0, 0.0, 1.1404210908949608E-12, 0.0, -7.749356711883593E-13, -4.4519943287468777E-13 ] ]
    Finite-Difference Derivative Accuracy:
    absoluteTol: 5.7740e-13 +- 7.3934e-13 [0.0000e+00 - 2.9954e-12] (24#)
    relativeTol: 4.5482e-13 +- 3.0810e-13 [1.8246e-13 - 1.3517e-12] (12#)
    
```

### Performance
Code from [LayerTestBase.java:144](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.00 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 0.2262 +- 0.0883 [0.1339 - 0.7124]
    Learning performance: 0.0040 +- 0.0034 [0.0000 - 0.0228]
    
```

