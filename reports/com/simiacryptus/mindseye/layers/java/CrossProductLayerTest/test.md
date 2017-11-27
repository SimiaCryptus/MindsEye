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
      "id": "0910987d-3688-428c-a892-e2c400000e0e",
      "isFrozen": false,
      "name": "CrossProductLayer/0910987d-3688-428c-a892-e2c400000e0e"
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
    [[ 0.492, 1.42, 1.668, 0.4 ]]
    --------------------
    Output: 
    [ 0.6986399999999999, 0.8206559999999999, 0.1968, 2.36856, 0.568, 0.6672 ]
```



### Differential Validation
Code from [LayerTestBase.java:139](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L139) executed in 0.00 seconds: 
```java
    getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Finite-Difference Derivative Accuracy:
    absoluteTol: 3.0292e-12 +- 4.5746e-12 [0.0000e+00 - 1.4678e-11] (24#)
    relativeTol: 3.9295e-12 +- 4.0350e-12 [2.6544e-13 - 1.4378e-11] (12#)
    
```

### Performance
Code from [LayerTestBase.java:144](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.00 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 0.0809 +- 0.0355 [0.0570 - 0.2536]
    Learning performance: 0.0030 +- 0.0027 [0.0000 - 0.0228]
    
```

