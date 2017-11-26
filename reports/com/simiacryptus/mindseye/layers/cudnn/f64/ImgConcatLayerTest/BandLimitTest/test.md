# ImgConcatLayer
## BandLimitTest
### Json Serialization
Code from [LayerTestBase.java:75](../../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L75) executed in 0.00 seconds: 
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
      "id": "b385277b-2d2d-42fe-8250-210c0000ec9c",
      "isFrozen": false,
      "name": "ImgConcatLayer/b385277b-2d2d-42fe-8250-210c0000ec9c",
      "maxBands": 3
    }
```



### Example Input/Output Pair
Code from [LayerTestBase.java:112](../../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L112) executed in 0.00 seconds: 
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
    	[ [ -0.9, 1.0 ], [ 1.444, 1.12 ] ],
    	[ [ -1.756, 0.48 ], [ 1.172, -0.632 ] ]
    ],
    [
    	[ [ -1.804, 1.588 ], [ -1.592, 1.028 ] ],
    	[ [ -0.74, -1.268 ], [ -1.952, 0.808 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ -0.9, 1.0, -1.804 ], [ 1.444, 1.12, -1.592 ] ],
    	[ [ -1.756, 0.48, -0.74 ], [ 1.172, -0.632, -1.952 ] ]
    ]
```



### Differential Validation
Code from [LayerTestBase.java:130](../../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L130) executed in 0.03 seconds: 
```java
    getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Finite-Difference Derivative Accuracy:
    absoluteTol: 4.0164e-12 +- 1.6808e-11 [0.0000e+00 - 8.2267e-11] (192#)
    relativeTol: 3.2131e-11 +- 1.2733e-11 [1.3378e-11 - 4.1133e-11] (12#)
    
```

### Performance
Code from [LayerTestBase.java:135](../../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L135) executed in 9.33 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 4.3457 +- 31.4264 [1.8011 - 3143.2119]
    Learning performance: 1.6526 +- 0.5630 [1.0259 - 5.6910]
    
```

