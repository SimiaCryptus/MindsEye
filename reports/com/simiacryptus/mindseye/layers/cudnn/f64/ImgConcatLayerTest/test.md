# ImgConcatLayer
## ImgConcatLayerTest
### Json Serialization
Code from [LayerTestBase.java:75](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L75) executed in 0.00 seconds: 
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
      "class": "com.simiacryptus.mindseye.layers.cudnn.f64.ImgConcatLayer",
      "id": "b385277b-2d2d-42fe-8250-210c0000ec98",
      "isFrozen": false,
      "name": "ImgConcatLayer/b385277b-2d2d-42fe-8250-210c0000ec98",
      "maxBands": -1
    }
```



### Example Input/Output Pair
Code from [LayerTestBase.java:112](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L112) executed in 0.00 seconds: 
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
    	[ [ -1.792 ], [ -1.928 ] ],
    	[ [ 1.048 ], [ -0.604 ] ]
    ],
    [
    	[ [ 1.048 ], [ 0.876 ] ],
    	[ [ 1.124 ], [ -0.204 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ -1.792, 1.048 ], [ -1.928, 0.876 ] ],
    	[ [ 1.048, 1.124 ], [ -0.604, -0.204 ] ]
    ]
```



### Differential Validation
Code from [LayerTestBase.java:130](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L130) executed in 0.02 seconds: 
```java
    getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Finite-Difference Derivative Accuracy:
    absoluteTol: 7.3413e-12 +- 2.2376e-11 [0.0000e+00 - 8.2267e-11] (64#)
    relativeTol: 2.9365e-11 +- 1.5712e-11 [5.0004e-13 - 4.1133e-11] (8#)
    
```

### Performance
Code from [LayerTestBase.java:135](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L135) executed in 5.34 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 2.1015 +- 0.4254 [1.7241 - 12.4279]
    Learning performance: 1.1812 +- 0.1441 [1.0259 - 5.2636]
    
```

