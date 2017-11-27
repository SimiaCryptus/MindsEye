# ConvolutionLayer
## DownsizeTest
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
      "class": "com.simiacryptus.mindseye.layers.aparapi.ConvolutionLayer",
      "id": "0910987d-3688-428c-a892-e2c400000005",
      "isFrozen": false,
      "name": "ConvolutionLayer/0910987d-3688-428c-a892-e2c400000005",
      "filter": {
        "dimensions": [
          3,
          3,
          21
        ],
        "data": [
          0.0,
          -1.964,
          -0.124,
          0.264,
          0.828,
          0.032,
          0.572,
          -1.66,
          0.556,
          -1.776,
          1.992,
          -1.552,
          -0.208,
          1.844,
          1.872,
          -1.804,
          1.532,
          1.732,
          1.208,
          -1.74,
          0.296,
          -1.208,
          -0.472,
          -0.304,
          1.824,
          0.464,
          -0.932,
          1.456,
          0.848,
          -0.88,
          -1.936,
          -1.312,
          1.148,
          0.252,
          0.912,
          -1.668,
          0.1,
          -0.416,
          -0.916,
          -0.036,
          -1.448,
          -1.896,
          -0.308,
          1.192,
          0.264,
          0.004,
          1.028,
          0.008,
          1.932,
          1.028,
          -0.156,
          0.42,
          -1.004,
          0.14,
          0.332,
          -0.172,
          1.608,
          0.464,
          0.548,
          0.396,
          1.488,
          -1.756,
          -1.764,
          -0.112,
          -1.708,
          0.344,
          -0.824,
          0.144,
          0.928,
          -0.58,
          -1.708,
          0.208,
          1.26,
          0.464,
          0.384,
          -0.472,
          1.752,
          -1.48,
          -0.468,
          0.88,
          -0.3,
          -0.74,
          0.66,
          -0.008,
          0.616,
          0.2,
          0.212,
          0.072,
          -1.148,
          -0.884,
          -1.572,
          0.044,
          -1.032,
          1.64,
          -0.14,
          0.036,
          1.868,
          1.204,
          -1.596,
          -0.072,
          1.408,
          0.492,
          1.552,
          0.624,
          -0.184,
          1.9,
          1.836,
          -1.452,
          -1.752,
          -1.152,
          1.516,
          1.808,
          -0.372,
          -0.488,
          0.292,
          -0.756,
          0.26,
          -0.528,
          1.772,
          -1.092,
          1.16,
          -0.436,
          -0.272,
          0.136,
          -0.896,
          -1.98,
          -0.2,
          -1.004,
          -1.996,
          -0.82,
          -0.028,
          1.56,
          -1.924,
          -1.676,
          1.636,
          0.128,
          -0.244,
          -1.42,
          1.288,
          0.104,
          -1.808,
          -0.704,
          1.704,
          1.824,
          -0.872,
          1.412,
          -1.552,
          0.02,
          1.16,
          -1.928,
          1.444,
          -0.152,
          1.056,
          -1.384,
          1.96,
          0.964,
          -1.856,
          1.324,
          1.704,
          1.948,
          -0.564,
          0.168,
          -1.52,
          -1.564,
          1.208,
          1.44,
          1.888,
          0.036,
          0.836,
          -0.532,
          1.072,
          -0.2,
          -1.644,
          -0.22,
          1.276,
          -1.936,
          -0.972,
          -0.812,
          -0.356,
          0.276,
          -0.904,
          0.188,
          0.576,
          -1.768,
          -0.14,
          -0.256,
          1.732,
          1.216,
          -0.108
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
Code from [LayerTestBase.java:121](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L121) executed in 0.01 seconds: 
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
    	[ [ 1.256, -0.872, 1.192, 1.068, 1.868, 1.996, 0.616 ], [ -0.904, -0.08, 1.488, -0.6, -0.36, -0.2, 1.256 ], [ 1.644, -0.628, -0.684, 1.452, 1.036, 1.612, 1.176 ] ],
    	[ [ -0.528, 1.564, 1.208, -0.932, 1.124, 0.788, -1.5 ], [ -0.74, 0.048, -0.92, 1.568, 1.28, -1.6, 0.76 ], [ 1.228, -1.308, -1.648, 0.66, -0.832, 1.676, -0.868 ] ],
    	[ [ 0.356, -1.372, 1.148, -0.284, 1.384, 0.416, 1.048 ], [ -0.412, -0.72, 0.468, -0.96, 1.776, -1.368, -1.52 ], [ -1.1, 1.312, 0.0, -0.3, -1.26, 1.984, -1.244 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ -5.617776000000001, -6.980272, -0.7541439999999999 ] ]
    ]
```



### Differential Validation
Code from [LayerTestBase.java:139](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L139) executed in 1.14 seconds: 
```java
    getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Finite-Difference Derivative Accuracy:
    absoluteTol: 1.1442e-12 +- 5.9904e-12 [0.0000e+00 - 5.9302e-11] (756#)
    relativeTol: 9.3426e-11 +- 4.6647e-10 [2.3044e-13 - 3.0387e-09] (41#)
    
```

### Performance
Code from [LayerTestBase.java:144](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.74 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 25.4049 +- 2.7373 [22.0972 - 39.2130]
    Learning performance: 20.0653 +- 3.2771 [17.5946 - 42.1312]
    
```

