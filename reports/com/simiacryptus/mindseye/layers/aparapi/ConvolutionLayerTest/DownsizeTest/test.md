# ConvolutionLayer
## DownsizeTest
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
      "class": "com.simiacryptus.mindseye.layers.aparapi.ConvolutionLayer",
      "id": "b385277b-2d2d-42fe-8250-210c00000005",
      "isFrozen": false,
      "name": "ConvolutionLayer/b385277b-2d2d-42fe-8250-210c00000005",
      "filter": {
        "dimensions": [
          3,
          3,
          21
        ],
        "data": [
          -1.908,
          0.704,
          0.064,
          -1.756,
          1.504,
          0.732,
          1.508,
          -0.852,
          1.964,
          0.212,
          1.44,
          0.86,
          1.824,
          -1.468,
          -0.784,
          1.356,
          0.784,
          -0.468,
          -0.744,
          0.444,
          -0.932,
          -1.164,
          -0.928,
          1.712,
          1.828,
          0.372,
          0.764,
          0.156,
          -0.336,
          1.592,
          -0.02,
          1.888,
          0.56,
          -1.556,
          1.332,
          0.196,
          1.52,
          1.356,
          1.272,
          0.896,
          0.296,
          0.124,
          1.452,
          0.86,
          1.848,
          0.468,
          1.596,
          1.156,
          -1.716,
          -0.604,
          -1.756,
          1.076,
          -0.856,
          1.54,
          0.916,
          -0.928,
          0.672,
          0.136,
          -1.94,
          0.86,
          -0.988,
          0.844,
          -0.244,
          0.56,
          -0.888,
          0.44,
          -0.652,
          -0.404,
          -1.904,
          -0.552,
          -0.344,
          1.62,
          0.848,
          -1.888,
          0.568,
          -0.06,
          1.752,
          0.696,
          -0.488,
          1.536,
          0.804,
          0.94,
          -1.088,
          -0.192,
          -0.348,
          -0.404,
          -0.976,
          0.148,
          -1.752,
          -1.7,
          1.844,
          0.58,
          0.452,
          -1.984,
          0.344,
          -0.408,
          -0.508,
          -1.788,
          -0.908,
          -1.808,
          1.892,
          -0.396,
          1.36,
          -0.016,
          0.14,
          0.252,
          1.792,
          1.62,
          1.176,
          1.636,
          -0.472,
          0.348,
          0.764,
          0.536,
          -0.9,
          0.22,
          -0.112,
          -1.288,
          0.644,
          -1.28,
          1.264,
          -0.2,
          -1.66,
          0.624,
          0.896,
          -0.084,
          -1.38,
          1.116,
          0.176,
          -1.92,
          -1.932,
          1.74,
          0.288,
          -0.388,
          1.956,
          0.608,
          1.328,
          -1.996,
          -1.888,
          0.736,
          -1.712,
          1.948,
          -1.344,
          -0.052,
          -0.34,
          1.628,
          0.008,
          -1.152,
          -0.34,
          -0.828,
          1.04,
          -1.932,
          0.536,
          0.764,
          -0.208,
          0.396,
          0.84,
          -0.692,
          -0.804,
          1.788,
          1.92,
          1.428,
          -1.976,
          -0.876,
          -1.404,
          -1.708,
          -0.684,
          -0.896,
          -1.62,
          1.772,
          0.804,
          -1.82,
          0.184,
          1.844,
          1.656,
          -1.112,
          1.112,
          1.776,
          -1.16,
          -1.208,
          1.892,
          -0.676,
          0.74,
          -1.044,
          1.36,
          -1.576,
          -0.428,
          -1.624,
          -1.504
        ]
      },
      "skip": {
        "dimensions": [
          1,
          1
        ]
      },
      "simple": false
    }
```



### Example Input/Output Pair
Code from [LayerTestBase.java:112](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L112) executed in 0.01 seconds: 
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
    	[ [ -0.632, 1.764, 1.636, -1.748, 1.328, 0.832, 0.468 ], [ -0.34, 0.364, -1.72, 1.036, -1.576, -1.312, -0.304 ], [ -1.98, -1.32, -1.972, 1.772, 0.452, 0.172, -0.004 ] ],
    	[ [ -1.116, -0.632, 0.984, -1.228, -1.86, -1.348, -1.652 ], [ 0.168, 0.72, -1.96, -0.676, -0.84, 1.58, -0.676 ], [ -1.232, 0.616, -0.912, -1.548, -0.012, 0.248, 1.028 ] ],
    	[ [ 0.088, -1.004, 0.148, 1.024, -0.06, 0.82, -0.5 ], [ 0.752, 1.012, 0.148, 1.416, 0.924, 1.02, -1.948 ], [ 1.216, 1.516, 0.252, -0.152, -1.076, -1.488, 0.464 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 2.4793119999999997, -2.60496, 5.531936 ] ]
    ]
```



### Differential Validation
Code from [LayerTestBase.java:130](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L130) executed in 1.07 seconds: 
```java
    getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Finite-Difference Derivative Accuracy:
    absoluteTol: 1.1020e-11 +- 5.5840e-11 [0.0000e+00 - 5.3082e-10] (756#)
    relativeTol: 1.1996e-10 +- 1.2842e-10 [1.3329e-12 - 5.6712e-10] (42#)
    
```

### Performance
Code from [LayerTestBase.java:135](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L135) executed in 66.37 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 23.6024 +- 3.1472 [20.4871 - 238.0767]
    Learning performance: 18.3465 +- 2.2248 [10.5727 - 156.7837]
    
```

