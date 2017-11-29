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
      "id": "c88cbdf1-1c2a-4a5e-b964-890900000f6a",
      "isFrozen": false,
      "name": "CrossProductLayer/c88cbdf1-1c2a-4a5e-b964-890900000f6a"
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
    [[ -0.284, 1.676, 0.504, 1.488 ]]
    --------------------
    Output: 
    [ -0.47598399999999996, -0.14313599999999999, -0.42259199999999997, 0.844704, 2.493888, 0.749952 ]
```



### Differential Validation
Code from [LayerTestBase.java:139](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L139) executed in 0.00 seconds: 
```java
    getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Component: CrossProductLayer/c88cbdf1-1c2a-4a5e-b964-890900000f6a
    Inputs: [ -0.284, 1.676, 0.504, 1.488 ]
    output=[ -0.47598399999999996, -0.14313599999999999, -0.42259199999999997, 0.844704, 2.493888, 0.749952 ]
    measured/actual: [ [ 1.6759999999998998, 0.5039999999997824, 1.4880000000000448, 0.0, 0.0, 0.0 ], [ -0.28399999999983994, 0.0, 0.0, 0.50400000000006, 1.4879999999983795, 0.0 ], [ 0.0, -0.2840000000001175, 0.0, 1.6759999999993447, 0.0, 1.4880000000006 ], [ 0.0, 0.0, -0.28399999999983994, 0.0, 1.6759999999971242, 0.50400000000006 ] ]
    implemented/expected: [ [ 1.676, 0.504, 1.488, 0.0, 0.0, 0.0 ], [ -0.284, 0.0, 0.0, 0.504, 1.488, 0.0 ], [ 0.0, -0.284, 0.0, 1.676, 0.0, 1.488 ], [ 0.0, 0.0, -0.284, 0.0, 1.676, 0.504 ] ]
    error: [ [ -1.0014211682118912E-13, -2.1760371282653068E-13, 4.4853010194856324E-14, 0.0, 0.0, 0.0 ], [ 1.6003864899971632E-13, 0.0, 0.0, 5.995204332975845E-14, -1.6204815267428785E-12, 0.0 ], [ 0.0, -1.1751710715657282E-13, 0.0, -6.552536291337674E-13, 0.0, 5.999645225074346E-13 ], [ 0.0, 0.0, 1.6003864899971632E-13, 0.0, -2.8756996783840805E-12, 5.995204332975845E-14 ] ]
    Finite-Difference Derivative Accuracy:
    absoluteTol: 2.7798e-13 +- 6.4427e-13 [0.0000e+00 - 2.8757e-12] (24#)
    relativeTol: 2.4581e-13 +- 2.3142e-13 [1.5072e-14 - 8.5791e-13] (12#)
    
```

### Performance
Code from [LayerTestBase.java:144](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.00 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 0.1801 +- 0.2096 [0.1168 - 2.0918]
    Learning performance: 0.0032 +- 0.0032 [0.0000 - 0.0285]
    
```

