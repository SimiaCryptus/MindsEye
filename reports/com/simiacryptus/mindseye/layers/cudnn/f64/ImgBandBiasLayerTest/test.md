# ImgBandBiasLayer
## ImgBandBiasLayerTest
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
      "class": "com.simiacryptus.mindseye.layers.cudnn.f64.ImgBandBiasLayer",
      "id": "0910987d-3688-428c-a892-e2c400000dd2",
      "isFrozen": false,
      "name": "ImgBandBiasLayer/0910987d-3688-428c-a892-e2c400000dd2",
      "bias": [
        0.0,
        0.0
      ]
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
    	[ [ 0.852, -1.0 ], [ 1.076, 1.432 ], [ -1.692, -1.004 ] ],
    	[ [ 1.572, -1.72 ], [ -1.84, -0.992 ], [ 1.904, -0.492 ] ],
    	[ [ 0.292, 1.748 ], [ -0.008, -0.98 ], [ -1.616, 1.244 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 0.852, -1.0 ], [ 1.076, 1.432 ], [ -1.692, -1.004 ] ],
    	[ [ 1.572, -1.72 ], [ -1.84, -0.992 ], [ 1.904, -0.492 ] ],
    	[ [ 0.292, 1.748 ], [ -0.008, -0.98 ], [ -1.616, 1.244 ] ]
    ]
```



### Differential Validation
Code from [LayerTestBase.java:139](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L139) executed in 0.04 seconds: 
```java
    getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Finite-Difference Derivative Accuracy:
    absoluteTol: 5.1281e-13 +- 1.6826e-12 [0.0000e+00 - 6.5510e-12] (360#)
    relativeTol: 2.5641e-12 +- 1.0773e-12 [2.0428e-14 - 3.2755e-12] (36#)
    
```

### Performance
Code from [LayerTestBase.java:144](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.07 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 1.9749 +- 0.6547 [1.6358 - 7.8284]
    Learning performance: 2.6643 +- 0.2734 [2.1630 - 3.3912]
    
```

