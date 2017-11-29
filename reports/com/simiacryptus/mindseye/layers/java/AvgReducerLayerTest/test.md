# AvgReducerLayer
## AvgReducerLayerTest
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
      "class": "com.simiacryptus.mindseye.layers.java.AvgReducerLayer",
      "id": "c88cbdf1-1c2a-4a5e-b964-890900000f62",
      "isFrozen": false,
      "name": "AvgReducerLayer/c88cbdf1-1c2a-4a5e-b964-890900000f62"
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
    [[ -0.664, -0.312, 0.508 ]]
    --------------------
    Output: 
    [ -0.15600000000000003 ]
```



### Differential Validation
Code from [LayerTestBase.java:139](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L139) executed in 0.00 seconds: 
```java
    getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Component: AvgReducerLayer/c88cbdf1-1c2a-4a5e-b964-890900000f62
    Inputs: [ -0.664, -0.312, 0.508 ]
    output=[ -0.15600000000000003 ]
    measured/actual: [ [ 0.3333333333332966 ], [ 0.3333333333332966 ], [ 0.3333333333332966 ] ]
    implemented/expected: [ [ 0.3333333333333333 ], [ 0.3333333333333333 ], [ 0.3333333333333333 ] ]
    error: [ [ -3.6692870963861424E-14 ], [ -3.6692870963861424E-14 ], [ -3.6692870963861424E-14 ] ]
    Finite-Difference Derivative Accuracy:
    absoluteTol: 3.6693e-14 +- 0.0000e+00 [3.6693e-14 - 3.6693e-14] (3#)
    relativeTol: 5.5039e-14 +- 0.0000e+00 [5.5039e-14 - 5.5039e-14] (3#)
    
```

### Performance
Code from [LayerTestBase.java:144](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.00 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 0.1565 +- 0.0495 [0.1282 - 0.5130]
    Learning performance: 0.0032 +- 0.0020 [0.0000 - 0.0171]
    
```

