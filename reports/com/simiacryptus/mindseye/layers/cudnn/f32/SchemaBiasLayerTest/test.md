# SchemaBiasLayer
## SchemaBiasLayerTest
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
      "class": "com.simiacryptus.mindseye.layers.cudnn.f32.SchemaBiasLayer",
      "id": "b385277b-2d2d-42fe-8250-210c000000d3",
      "isFrozen": false,
      "name": "SchemaBiasLayer/b385277b-2d2d-42fe-8250-210c000000d3",
      "selected": [
        "test1",
        "test2"
      ],
      "features": {
        "test2": 0.0,
        "test1": 0.0
      }
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
    	[ [ -1.8, -0.04 ], [ 0.592, 1.516 ], [ 0.528, 0.02 ] ],
    	[ [ -0.188, -1.192 ], [ -0.98, 1.648 ], [ -1.804, -0.156 ] ],
    	[ [ -1.408, -1.716 ], [ 1.864, -0.264 ], [ 1.54, 0.368 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ -1.7999999523162842, -0.03999999910593033 ], [ 0.5920000076293945, 1.5160000324249268 ], [ 0.527999997138977, 0.019999999552965164 ] ],
    	[ [ -0.18799999356269836, -1.1920000314712524 ], [ -0.9800000190734863, 1.6480000019073486 ], [ -1.8040000200271606, -0.15600000321865082 ] ],
    	[ [ -1.4079999923706055, -1.715999960899353 ], [ 1.8639999628067017, -0.2639999985694885 ], [ 1.5399999618530273, 0.36800000071525574 ] ]
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
    absoluteTol: 1.5541e-05 +- 6.8479e-05 [0.0000e+00 - 1.0262e-03] (360#)
    relativeTol: 7.7706e-05 +- 7.9346e-05 [8.4937e-07 - 5.1334e-04] (36#)
    
```

### Performance
Code from [LayerTestBase.java:135](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L135) executed in 10.23 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 2.1871 +- 0.3621 [1.8780 - 11.0914]
    Learning performance: 4.2656 +- 29.0938 [2.2086 - 2912.1514]
    
```

