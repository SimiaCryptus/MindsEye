# ScaleMetaLayer
## ScaleMetaLayerTest
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
      "class": "com.simiacryptus.mindseye.layers.java.ScaleMetaLayer",
      "id": "c88cbdf1-1c2a-4a5e-b964-890900000f98",
      "isFrozen": false,
      "name": "ScaleMetaLayer/c88cbdf1-1c2a-4a5e-b964-890900000f98"
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
    [[ 1.292, 1.148, -1.644 ],
    [ -0.628, -1.496, 1.196 ]]
    --------------------
    Output: 
    [ -0.811376, -1.7174079999999998, -1.9662239999999997 ]
```



### Differential Validation
Code from [LayerTestBase.java:139](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L139) executed in 0.00 seconds: 
```java
    getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Component: ScaleMetaLayer/c88cbdf1-1c2a-4a5e-b964-890900000f98
    Inputs: [ 1.292, 1.148, -1.644 ],
    [ -0.628, -1.496, 1.196 ]
    output=[ -0.811376, -1.7174079999999998, -1.9662239999999997 ]
    measured/actual: [ [ -0.628000000000295, 0.0, 0.0 ], [ 0.0, -1.49600000000083, 0.0 ], [ 0.0, 0.0, 1.1959999999988646 ] ]
    implemented/expected: [ [ -0.628, -0.0, -0.0 ], [ -0.0, -1.496, -0.0 ], [ 0.0, 0.0, 1.196 ] ]
    error: [ [ -2.949862576429041E-13, 0.0, 0.0 ], [ 0.0, -8.30002733209767E-13, 0.0 ], [ 0.0, 0.0, -1.135314064981685E-12 ] ]
    Component: ScaleMetaLayer/c88cbdf1-1c2a-4a5e-b964-890900000f98
    Inputs: [ 1.292, 1.148, -1.644 ],
    [ -0.628, -1.496, 1.196 ]
    output=[ -0.811376, -1.7174079999999998, -1.9662239999999997 ]
    measured/actual: [ [ 1.2919999999994047, 0.0, 0.0 ], [ 0.0, 1.1479999999997048, 0.0 ], [ 0.0, 0.0, -1.644000000000645 ] ]
    implemented/expected: [ [ 1.292, 0.0, 0.0 ], [ 0.0, 1.148, 0.0 ], [ 0.0, 0.0, -1.644 ] ]
    error: [ [ -5.953015858040089E-13, 0.0, 0.0 ], [ 0.0, -2.950972799453666E-13, 0.0 ], [ 0.0, 0.0, -6.45039577307216E-13 ] ]
    Finite-Difference Derivative Accuracy:
    absoluteTol: 2.1087e-13 +- 3.4335e-13 [0.0000e+00 - 1.1353e-12] (18#)
    relativeTol: 2.5700e-13 +- 1.0738e-13 [1.2853e-13 - 4.7463e-13] (6#)
    
```

### Performance
Code from [LayerTestBase.java:144](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.00 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 0.1771 +- 0.0509 [0.1197 - 0.4645]
    Learning performance: 0.0019 +- 0.0015 [0.0000 - 0.0057]
    
```

