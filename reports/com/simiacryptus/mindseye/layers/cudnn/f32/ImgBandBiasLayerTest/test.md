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
      "class": "com.simiacryptus.mindseye.layers.cudnn.f32.ImgBandBiasLayer",
      "id": "b385277b-2d2d-42fe-8250-210c000000c3",
      "isFrozen": false,
      "name": "ImgBandBiasLayer/b385277b-2d2d-42fe-8250-210c000000c3",
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
    	[ [ 0.304, 1.12 ], [ -0.76, 1.248 ], [ 0.908, 1.856 ] ],
    	[ [ 0.568, -0.432 ], [ 0.672, -1.52 ], [ 1.768, -0.896 ] ],
    	[ [ -0.544, 0.288 ], [ -1.532, -0.776 ], [ 1.616, 0.628 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 0.30399999022483826, 1.1200000047683716 ], [ -0.7599999904632568, 1.2480000257492065 ], [ 0.9079999923706055, 1.8559999465942383 ] ],
    	[ [ 0.5680000185966492, -0.4320000112056732 ], [ 0.671999990940094, -1.5199999809265137 ], [ 1.7680000066757202, -0.8960000276565552 ] ],
    	[ [ -0.5440000295639038, 0.2879999876022339 ], [ -1.531999945640564, -0.7760000228881836 ], [ 1.6160000562667847, 0.628000020980835 ] ]
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
    absoluteTol: 2.2558e-05 +- 9.3447e-05 [0.0000e+00 - 1.0262e-03] (360#)
    relativeTol: 1.1280e-04 +- 1.0195e-04 [6.6046e-05 - 5.1334e-04] (36#)
    
```

### Performance
Code from [LayerTestBase.java:135](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L135) executed in 7.07 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 2.1758 +- 0.4281 [1.8552 - 11.6157]
    Learning performance: 2.7274 +- 0.3519 [2.1630 - 9.9258]
    
```

