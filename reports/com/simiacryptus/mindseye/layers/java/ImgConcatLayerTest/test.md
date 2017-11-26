# ImgConcatLayer
## ImgConcatLayerTest
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
      "class": "com.simiacryptus.mindseye.layers.java.ImgConcatLayer",
      "id": "b385277b-2d2d-42fe-8250-210c0000ecfa",
      "isFrozen": false,
      "name": "ImgConcatLayer/b385277b-2d2d-42fe-8250-210c0000ecfa"
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
    	[ [ -0.028 ], [ -0.16 ] ],
    	[ [ -0.592 ], [ 1.008 ] ]
    ],
    [
    	[ [ 0.492 ], [ -1.42 ] ],
    	[ [ 0.78 ], [ -0.412 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ -0.028, 0.492 ], [ -0.16, -1.42 ] ],
    	[ [ -0.592, 0.78 ], [ 1.008, -0.412 ] ]
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
    absoluteTol: 4.3368e-12 +- 1.5522e-11 [0.0000e+00 - 8.2267e-11] (64#)
    relativeTol: 1.7347e-11 +- 1.4783e-11 [5.0004e-13 - 4.1133e-11] (8#)
    
```

### Performance
Code from [LayerTestBase.java:135](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L135) executed in 0.05 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 0.0231 +- 0.0253 [0.0142 - 1.8866]
    Learning performance: 0.0081 +- 0.0079 [0.0057 - 0.4189]
    
```

