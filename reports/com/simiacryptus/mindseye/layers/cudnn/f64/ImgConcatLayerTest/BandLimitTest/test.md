# ImgConcatLayer
## BandLimitTest
### Json Serialization
Code from [LayerTestBase.java:84](../../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L84) executed in 0.05 seconds: 
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
      "id": "100ef1c9-ed3b-4ea2-b0d5-09ff00000001",
      "isFrozen": false,
      "name": "ImgConcatLayer/100ef1c9-ed3b-4ea2-b0d5-09ff00000001",
      "maxBands": 3
    }
```



### Example Input/Output Pair
Code from [LayerTestBase.java:121](../../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L121) executed in 0.02 seconds: 
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
    	[ [ -1.456, -1.364 ], [ 1.848, -1.376 ] ],
    	[ [ -1.58, 1.96 ], [ 2.0, -1.128 ] ]
    ],
    [
    	[ [ -0.412, -0.512 ], [ -1.632, -1.584 ] ],
    	[ [ 0.644, -0.6 ], [ -0.648, 1.964 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ -1.456, -1.364, -0.412 ], [ 1.848, -1.376, -1.632 ] ],
    	[ [ -1.58, 1.96, 0.644 ], [ 2.0, -1.128, -0.648 ] ]
    ]
```



### Differential Validation
Code from [LayerTestBase.java:139](../../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L139) executed in 0.06 seconds: 
```java
    getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Finite-Difference Derivative Accuracy:
    absoluteTol: 3.5969e-13 +- 1.4503e-12 [0.0000e+00 - 6.5510e-12] (192#)
    relativeTol: 2.8775e-12 +- 8.0638e-13 [4.9993e-13 - 3.2755e-12] (12#)
    
```

### Performance
Code from [LayerTestBase.java:144](../../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.08 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 3.1099 +- 1.3053 [2.3340 - 15.0440]
    Learning performance: 1.3886 +- 0.5495 [1.1884 - 6.7768]
    
```

