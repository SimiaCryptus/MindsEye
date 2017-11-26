# ImgBandSelectLayer
## ImgBandSelectLayerTest
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
      "class": "com.simiacryptus.mindseye.layers.java.ImgBandSelectLayer",
      "id": "b385277b-2d2d-42fe-8250-210c0000ecf6",
      "isFrozen": false,
      "name": "ImgBandSelectLayer/b385277b-2d2d-42fe-8250-210c0000ecf6",
      "bands": [
        0,
        2
      ]
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
    [[
    	[ [ 0.288, 1.52, -1.352 ], [ -1.408, -1.1, -1.504 ] ],
    	[ [ -1.468, 0.5, 1.46 ], [ 1.88, 0.868, 0.572 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 0.288, -1.352 ], [ -1.408, -1.504 ] ],
    	[ [ -1.468, 1.46 ], [ 1.88, 0.572 ] ]
    ]
```



### Differential Validation
Code from [LayerTestBase.java:130](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L130) executed in 0.00 seconds: 
```java
    getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Finite-Difference Derivative Accuracy:
    absoluteTol: 5.7199e-12 +- 2.0158e-11 [0.0000e+00 - 8.2267e-11] (96#)
    relativeTol: 3.4319e-11 +- 1.1805e-11 [1.3378e-11 - 4.1133e-11] (8#)
    
```

### Performance
Code from [LayerTestBase.java:135](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L135) executed in 0.11 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 0.0889 +- 6.6121 [0.0142 - 661.2610]
    Learning performance: 0.0008 +- 0.0196 [0.0000 - 1.9151]
    
```

