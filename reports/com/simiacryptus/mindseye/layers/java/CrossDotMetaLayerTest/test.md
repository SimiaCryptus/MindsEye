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
      "id": "0910987d-3688-428c-a892-e2c400000e0a",
      "isFrozen": false,
      "name": "CrossDotMetaLayer/0910987d-3688-428c-a892-e2c400000e0a"
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
    [[ 0.876, 1.656, 0.576 ]]
    --------------------
    Output: 
    [ [ 0.0, 1.450656, 0.5045759999999999 ], [ 1.450656, 0.0, 0.9538559999999999 ], [ 0.5045759999999999, 0.9538559999999999, 0.0 ] ]
```



### Differential Validation
Code from [LayerTestBase.java:139](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L139) executed in 0.00 seconds: 
```java
    getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Finite-Difference Derivative Accuracy:
    absoluteTol: 3.2562e-12 +- 4.6769e-12 [0.0000e+00 - 1.6774e-11] (27#)
    relativeTol: 3.7262e-12 +- 1.6438e-12 [1.7124e-12 - 6.6679e-12] (12#)
    
```

### Performance
Code from [LayerTestBase.java:144](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.00 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 0.0335 +- 0.0360 [0.0256 - 0.3705]
    Learning performance: 0.0013 +- 0.0024 [0.0000 - 0.0199]
    
```

