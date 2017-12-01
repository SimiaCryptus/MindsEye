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
      "id": "f4569375-56fe-4e46-925c-95f400000997",
      "isFrozen": false,
      "name": "CrossDotMetaLayer/f4569375-56fe-4e46-925c-95f400000997"
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
    [[ -1.936, 0.38, 0.824 ]]
    --------------------
    Output: 
    [ [ 0.0, -0.73568, -1.5952639999999998 ], [ -0.73568, 0.0, 0.31312 ], [ -1.5952639999999998, 0.31312, 0.0 ] ]
```



### Differential Validation
Code from [LayerTestBase.java:139](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L139) executed in 0.00 seconds: 
```java
    getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Feedback for input 0
    Inputs: [ -1.936, 0.38, 0.824 ]
    Output: [ [ 0.0, -0.73568, -1.5952639999999998 ], [ -0.73568, 0.0, 0.31312 ], [ -1.5952639999999998, 0.31312, 0.0 ] ]
    Measured: [ [ 0.0, 0.3799999999998249, 0.8239999999992698, 0.3799999999998249, 0.0, 0.0, 0.8239999999992698, 0.0, 0.0 ], [ 0.0, -1.9360000000001598, 0.0, -1.9360000000001598, 0.0, 0.8239999999998249, 0.0, 0.8239999999998249, 0.0 ], [ 0.0, 0.0, -1.9360000000001598, 0.0, 0.0, 0.3799999999998249, -1.9360000000001598, 0.3799999999998249, 0.0 ] ]
    Implemented: [ [ 0.0, 0.38, 0.824, 0.38, 0.0, 0.0, 0.824, 0.0, 0.0 ], [ 0.0, -1.936, 0.0, -1.936, 0.0, 0.824, 0.0, 0.824, 0.0 ], [ 0.0, 0.0, -1.936, 0.0, 0.0, 0.38, -1.936, 0.38, 0.0 ] ]
    Error: [ [ 0.0, -1.7508217098338719E-13, -7.301936832959655E-13, -1.7508217098338719E-13, 0.0, 0.0, -7.301936832959655E-13, 0.0, 0.0 ], [ 0.0, -1.5987211554602254E-13, 0.0, -1.5987211554602254E-13, 0.0, -1.7508217098338719E-13, 0.0, -1.7508217098338719E-13, 0.0 ], [ 0.0, 0.0, -1.5987211554602254E-13, 0.0, 0.0, -1.7508217098338719E-13, -1.5987211554602254E-13, -1.7508217098338719E-13, 0.0 ] ]
    Finite-Difference Derivative Accuracy:
    absoluteTol: 1.1668e-13 +- 1.9100e-13 [0.0000e+00 - 7.3019e-13] (27#)
    relativeTol: 1.8211e-13 +- 1.4034e-13 [4.1289e-14 - 4.4308e-13] (12#)
    
```

### Performance
Code from [LayerTestBase.java:144](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.00 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 0.1561 +- 0.0265 [0.1254 - 0.2793]
    Learning performance: 0.0032 +- 0.0027 [0.0000 - 0.0200]
    
```

