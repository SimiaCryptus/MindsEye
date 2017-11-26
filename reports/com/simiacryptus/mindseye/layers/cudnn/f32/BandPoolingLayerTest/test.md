# BandPoolingLayer
## BandPoolingLayerTest
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
      "class": "com.simiacryptus.mindseye.layers.cudnn.f32.BandPoolingLayer",
      "id": "b385277b-2d2d-42fe-8250-210c00000011",
      "isFrozen": false,
      "name": "BandPoolingLayer/b385277b-2d2d-42fe-8250-210c00000011",
      "mode": 0
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
    	[ [ -0.064, 1.684 ], [ -0.692, 0.408 ], [ -0.74, 0.64 ] ],
    	[ [ 0.516, -0.056 ], [ 0.288, 1.512 ], [ 0.308, 0.852 ] ],
    	[ [ 1.508, 1.26 ], [ -0.124, -1.78 ], [ -1.804, 1.796 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 1.5080000162124634, 1.7960000038146973 ] ]
    ]
```



### Differential Validation
Code from [LayerTestBase.java:130](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L130) executed in 0.01 seconds: 
```java
    getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Finite-Difference Derivative Accuracy:
    absoluteTol: 9.2189e-06 +- 3.8010e-05 [0.0000e+00 - 1.6594e-04] (36#)
    relativeTol: 8.2963e-05 +- 0.0000e+00 [8.2963e-05 - 8.2963e-05] (2#)
    
```

### Performance
Code from [LayerTestBase.java:135](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L135) executed in 6.44 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 2.6721 +- 19.4605 [1.2282 - 1946.1687]
    Learning performance: 1.5421 +- 0.5395 [0.7552 - 5.8335]
    
```

