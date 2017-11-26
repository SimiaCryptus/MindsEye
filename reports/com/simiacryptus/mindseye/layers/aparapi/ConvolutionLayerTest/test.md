# ConvolutionLayer
## ConvolutionLayerTest
### Json Serialization
Code from [LayerTestBase.java:75](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L75) executed in 0.04 seconds: 
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
      "class": "com.simiacryptus.mindseye.layers.aparapi.ConvolutionLayer",
      "id": "b385277b-2d2d-42fe-8250-210c00000001",
      "isFrozen": false,
      "name": "ConvolutionLayer/b385277b-2d2d-42fe-8250-210c00000001",
      "filter": {
        "dimensions": [
          3,
          3,
          4
        ],
        "data": [
          0.08,
          -0.236,
          -0.752,
          1.836,
          1.284,
          1.196,
          -1.728,
          1.72,
          0.728,
          -0.224,
          1.372,
          -0.9,
          0.712,
          0.468,
          0.284,
          1.992,
          0.076,
          1.1,
          0.896,
          1.468,
          1.264,
          0.644,
          -1.82,
          -0.74,
          0.576,
          0.68,
          -0.276,
          0.904,
          1.956,
          1.78,
          -0.196,
          -1.388,
          -1.452,
          0.724,
          -0.276,
          -1.688
        ]
      },
      "skip": {
        "dimensions": [
          1,
          1
        ]
      },
      "simple": true
    }
```



### Example Input/Output Pair
Code from [LayerTestBase.java:112](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L112) executed in 0.29 seconds: 
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
    	[ [ -0.452, 0.5 ], [ 1.852, 0.444 ], [ 1.72, -0.268 ] ],
    	[ [ 0.364, -0.692 ], [ -1.24, 1.384 ], [ -1.016, -1.992 ] ],
    	[ [ -1.184, 1.24 ], [ -1.42, -0.916 ], [ -0.532, 0.568 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 0.08787199999999995, 4.427568000000001 ], [ -3.9459359999999997, -0.589504 ], [ 5.975279999999999, -0.6058719999999999 ] ],
    	[ [ -5.120000000011227E-4, -1.1902080000000006 ], [ -6.8232800000000005, -12.717552000000001 ], [ 5.925551999999999, -0.003408000000000036 ] ],
    	[ [ -1.1573440000000002, -1.3274880000000002 ], [ -4.195136, -2.868864 ], [ -5.807888000000001, -1.9888159999999997 ] ]
    ]
```



### Differential Validation
Code from [LayerTestBase.java:130](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L130) executed in 0.53 seconds: 
```java
    getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Finite-Difference Derivative Accuracy:
    absoluteTol: 1.0127e-10 +- 1.9998e-10 [0.0000e+00 - 1.5246e-09] (972#)
    relativeTol: 1.9905e-10 +- 4.4613e-10 [3.2437e-13 - 6.3105e-09] (392#)
    
```

### Performance
Code from [LayerTestBase.java:135](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L135) executed in 61.59 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 21.5691 +- 1.9804 [5.2094 - 155.1793]
    Learning performance: 17.9384 +- 1.3351 [11.1711 - 54.4993]
    
```

