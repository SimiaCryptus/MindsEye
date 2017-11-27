# ImgConcatLayer
## ImgConcatLayerTest
### Json Serialization
Code from [LayerTestBase.java:84](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L84) executed in 0.00 seconds: 
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
      "id": "0910987d-3688-428c-a892-e2c400000dd6",
      "isFrozen": false,
      "name": "ImgConcatLayer/0910987d-3688-428c-a892-e2c400000dd6",
      "maxBands": -1
    }
```



### Example Input/Output Pair
Code from [LayerTestBase.java:121](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L121) executed in 0.00 seconds: 
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
    	[ [ -0.428 ], [ -0.008 ] ],
    	[ [ -0.66 ], [ -1.916 ] ]
    ],
    [
    	[ [ -1.108 ], [ 1.092 ] ],
    	[ [ -0.924 ], [ 0.264 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ -0.428, -1.108 ], [ -0.008, 1.092 ] ],
    	[ [ -0.66, -0.924 ], [ -1.916, 0.264 ] ]
    ]
```



### Differential Validation
Code from [LayerTestBase.java:139](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L139) executed in 0.02 seconds: 
```java
    getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Finite-Difference Derivative Accuracy:
    absoluteTol: 4.8118e-13 +- 1.5680e-12 [0.0000e+00 - 6.5510e-12] (64#)
    relativeTol: 1.9247e-12 +- 1.2945e-12 [2.0428e-14 - 3.2755e-12] (8#)
    
```

### Performance
Code from [LayerTestBase.java:144](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.06 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 2.1331 +- 0.2665 [1.8267 - 3.8928]
    Learning performance: 1.1677 +- 0.1616 [1.0459 - 2.6788]
    
```

