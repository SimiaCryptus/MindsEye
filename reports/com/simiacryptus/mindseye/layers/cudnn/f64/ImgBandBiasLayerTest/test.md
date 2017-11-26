# ImgBandBiasLayer
## ImgBandBiasLayerTest
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
      "class": "com.simiacryptus.mindseye.layers.cudnn.f64.ImgBandBiasLayer",
      "id": "b385277b-2d2d-42fe-8250-210c0000ec94",
      "isFrozen": false,
      "name": "ImgBandBiasLayer/b385277b-2d2d-42fe-8250-210c0000ec94",
      "bias": [
        0.0,
        0.0
      ]
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
    	[ [ -0.264, 1.656 ], [ 0.548, -1.74 ], [ 1.028, 1.784 ] ],
    	[ [ 0.436, -1.86 ], [ -1.408, -1.852 ], [ 0.848, 1.384 ] ],
    	[ [ 1.344, 0.308 ], [ -1.764, -0.732 ], [ 1.74, -1.484 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ -0.264, 1.656 ], [ 0.548, -1.74 ], [ 1.028, 1.784 ] ],
    	[ [ 0.436, -1.86 ], [ -1.408, -1.852 ], [ 0.848, 1.384 ] ],
    	[ [ 1.344, 0.308 ], [ -1.764, -0.732 ], [ 1.74, -1.484 ] ]
    ]
```



### Differential Validation
Code from [LayerTestBase.java:130](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L130) executed in 0.04 seconds: 
```java
    getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Finite-Difference Derivative Accuracy:
    absoluteTol: 6.4096e-12 +- 2.0876e-11 [0.0000e+00 - 8.2267e-11] (360#)
    relativeTol: 3.2048e-11 +- 1.2852e-11 [1.3378e-11 - 4.1133e-11] (36#)
    
```

### Performance
Code from [LayerTestBase.java:135](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L135) executed in 6.44 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 2.0698 +- 3.4709 [1.5132 - 339.8396]
    Learning performance: 2.6793 +- 0.3495 [2.0974 - 7.9566]
    
```

