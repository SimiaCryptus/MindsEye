# ScaleMetaLayer
## ScaleMetaLayerTest
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
      "class": "com.simiacryptus.mindseye.layers.java.ScaleMetaLayer",
      "id": "b385277b-2d2d-42fe-8250-210c0000ed5b",
      "isFrozen": false,
      "name": "ScaleMetaLayer/b385277b-2d2d-42fe-8250-210c0000ed5b"
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
    [[ 1.32, 0.048, 0.352 ],
    [ 1.904, -0.344, 0.052 ]]
    --------------------
    Output: 
    [ 2.51328, -0.016512, 0.018303999999999997 ]
```



### Differential Validation
Code from [LayerTestBase.java:130](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L130) executed in 0.00 seconds: 
```java
    getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Finite-Difference Derivative Accuracy:
    absoluteTol: 8.1528e-12 +- 2.6486e-11 [0.0000e+00 - 1.1400e-10] (18#)
    relativeTol: 9.7648e-12 +- 9.7404e-12 [9.5224e-13 - 2.9938e-11] (6#)
    
```

### Performance
Code from [LayerTestBase.java:135](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L135) executed in 0.10 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 0.0117 +- 0.0119 [0.0085 - 0.5386]
    Learning performance: 0.0007 +- 0.0028 [0.0000 - 0.1482]
    
```

