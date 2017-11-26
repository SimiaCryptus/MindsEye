# MeanSqLossLayer
## MeanSqLossLayerTest
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
      "class": "com.simiacryptus.mindseye.layers.java.MeanSqLossLayer",
      "id": "b385277b-2d2d-42fe-8250-210c0000ed24",
      "isFrozen": false,
      "name": "MeanSqLossLayer/b385277b-2d2d-42fe-8250-210c0000ed24"
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
    [[ -0.716, 1.312, -2.0, 0.8 ],
    [ -1.56, 0.068, 0.46, -0.432 ]]
    --------------------
    Output: 
    [ 2.457324 ]
```



### Differential Validation
Code from [LayerTestBase.java:130](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L130) executed in 0.00 seconds: 
```java
    getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Finite-Difference Derivative Accuracy:
    absoluteTol: 2.5030e-07 +- 2.8151e-10 [2.4986e-07 - 2.5084e-07] (8#)
    relativeTol: 2.0063e-07 +- 6.8759e-08 [1.0167e-07 - 2.9643e-07] (8#)
    
```

### Performance
Code from [LayerTestBase.java:135](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L135) executed in 0.03 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 0.0141 +- 0.0108 [0.0085 - 0.5671]
    Learning performance: 0.0007 +- 0.0032 [0.0000 - 0.1824]
    
```

