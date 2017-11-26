# ImgBandBiasLayer
## ImgBandBiasLayerTest
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
      "class": "com.simiacryptus.mindseye.layers.java.ImgBandBiasLayer",
      "id": "b385277b-2d2d-42fe-8250-210c0000ecee",
      "isFrozen": false,
      "name": "ImgBandBiasLayer/b385277b-2d2d-42fe-8250-210c0000ecee",
      "bias": [
        0.0,
        0.0,
        0.0
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
    	[ [ -1.22, -1.128, 1.572 ], [ 1.628, -0.804, 0.856 ] ],
    	[ [ 0.848, 0.408, 1.476 ], [ -1.68, 1.348, -1.204 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ -1.22, -1.128, 1.572 ], [ 1.628, -0.804, 0.856 ] ],
    	[ [ 0.848, 0.408, 1.476 ], [ -1.68, 1.348, -1.204 ] ]
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
    absoluteTol: 8.5684e-12 +- 2.3742e-11 [0.0000e+00 - 8.2267e-11] (180#)
    relativeTol: 3.2131e-11 +- 1.2733e-11 [1.3378e-11 - 4.1133e-11] (24#)
    
```

### Performance
Code from [LayerTestBase.java:135](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L135) executed in 0.18 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 0.1388 +- 10.5137 [0.0199 - 1051.4392]
    Learning performance: 0.0187 +- 0.0335 [0.0114 - 3.0692]
    
```

