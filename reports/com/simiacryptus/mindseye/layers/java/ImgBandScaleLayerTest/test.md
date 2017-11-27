# ImgBandScaleLayer
## ImgBandScaleLayerTest
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
      "class": "com.simiacryptus.mindseye.layers.java.ImgBandScaleLayer",
      "id": "0910987d-3688-428c-a892-e2c400000e30",
      "isFrozen": false,
      "name": "ImgBandScaleLayer/0910987d-3688-428c-a892-e2c400000e30",
      "bias": [
        0.0,
        0.0,
        0.0
      ]
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
    [[
    	[ [ 0.348, 0.756, 1.14 ], [ 0.964, -1.072, -1.7 ] ],
    	[ [ -0.14, -1.548, 1.72 ], [ 1.744, -0.548, 1.304 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 0.0, 0.0, 0.0 ], [ 0.0, -0.0, -0.0 ] ],
    	[ [ -0.0, -0.0, 0.0 ], [ 0.0, -0.0, 0.0 ] ]
    ]
```



### Differential Validation
Code from [LayerTestBase.java:139](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L139) executed in 0.00 seconds: 
```java
    getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Finite-Difference Derivative Accuracy:
    absoluteTol: 5.7053e-18 +- 3.3702e-17 [0.0000e+00 - 2.2204e-16] (180#)
    relativeTol: 4.2545e-17 +- 4.3890e-17 [0.0000e+00 - 1.0357e-16] (12#)
    
```

### Performance
Code from [LayerTestBase.java:144](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.00 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 0.0722 +- 0.0229 [0.0598 - 0.2308]
    Learning performance: 0.0560 +- 0.0235 [0.0427 - 0.2223]
    
```

