# CrossDotMetaLayer
## CrossDotMetaLayerTest
### Json Serialization
Code from [LayerTestBase.java:75](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L75) executed in 0.00 seconds: 
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
      "id": "b385277b-2d2d-42fe-8250-210c0000eccc",
      "isFrozen": false,
      "name": "CrossDotMetaLayer/b385277b-2d2d-42fe-8250-210c0000eccc"
    }
```



### Example Input/Output Pair
Code from [LayerTestBase.java:112](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L112) executed in 0.00 seconds: 
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
    [[ -0.696, -1.652, 0.536 ]]
    --------------------
    Output: 
    [ [ 0.0, 1.149792, -0.373056 ], [ 1.149792, 0.0, -0.885472 ], [ -0.373056, -0.885472, 0.0 ] ]
```



### Differential Validation
Code from [LayerTestBase.java:130](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L130) executed in 0.00 seconds: 
```java
    getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Finite-Difference Derivative Accuracy:
    absoluteTol: 2.3180e-11 +- 3.8562e-11 [0.0000e+00 - 1.2966e-10] (27#)
    relativeTol: 2.9503e-11 +- 2.1921e-11 [4.2640e-12 - 5.6047e-11] (12#)
    
```

### Performance
Code from [LayerTestBase.java:135](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L135) executed in 0.02 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 0.0086 +- 0.0172 [0.0057 - 1.5930]
    Learning performance: 0.0007 +- 0.0039 [0.0000 - 0.1881]
    
```

