# ConvolutionLayer
## DownsizeTest
### Json Serialization
Code from [LayerTestBase.java:83](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L83) executed in 0.00 seconds: 
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
      "id": "370a9587-74a1-4959-b406-fa4500000014",
      "isFrozen": false,
      "name": "ConvolutionLayer/370a9587-74a1-4959-b406-fa4500000014",
      "filter": {
        "dimensions": [
          3,
          3,
          21
        ],
        "data": [
          1.572,
          0.18,
          0.492,
          0.424,
          -1.428,
          -0.648,
          1.156,
          1.248,
          -1.232,
          1.04,
          -1.432,
          -1.892,
          -0.544,
          0.204,
          -1.92,
          -1.08,
          -1.48,
          0.552,
          1.944,
          -0.456,
          1.036,
          1.972,
          1.48,
          -0.752,
          -0.976,
          -0.364,
          1.656,
          0.336,
          1.94,
          -1.084,
          -0.368,
          -1.864,
          1.472,
          0.776,
          -0.504,
          -0.036,
          0.896,
          0.168,
          0.276,
          1.1,
          1.02,
          0.608,
          -0.908,
          -1.808,
          -1.456,
          -1.024,
          0.692,
          -1.62,
          0.356,
          1.532,
          0.972,
          -0.956,
          -0.86,
          -1.576,
          -1.54,
          1.812,
          -1.104,
          1.86,
          1.852,
          0.196,
          -1.696,
          -0.256,
          1.42,
          0.248,
          1.136,
          -1.964,
          1.972,
          0.964,
          0.5,
          1.532,
          0.496,
          1.696,
          -0.284,
          0.152,
          -1.764,
          1.9,
          -1.036,
          -0.664,
          -1.06,
          -1.94,
          1.296,
          -1.544,
          -0.956,
          1.104,
          -0.452,
          1.688,
          1.632,
          -0.836,
          -0.236,
          1.02,
          -0.848,
          -0.304,
          -1.072,
          -1.628,
          1.856,
          -0.364,
          -1.628,
          -0.644,
          0.704,
          1.808,
          1.916,
          1.992,
          0.784,
          0.9,
          -1.548,
          -1.168,
          1.084,
          -1.376,
          1.524,
          -1.4,
          -0.42,
          1.416,
          -0.356,
          -1.076,
          -0.992,
          0.796,
          -0.14,
          0.14,
          1.212,
          -0.424,
          -0.044,
          -1.572,
          -1.936,
          0.72,
          -0.204,
          -0.748,
          -1.092,
          -0.728,
          1.46,
          0.792,
          -0.06,
          -0.1,
          1.296,
          0.416,
          1.56,
          -0.42,
          -1.128,
          1.864,
          1.396,
          0.488,
          -0.624,
          -1.904,
          -0.928,
          -0.044,
          1.536,
          -0.9,
          0.728,
          -0.172,
          -1.888,
          -0.648,
          0.396,
          0.588,
          -0.756,
          -0.42,
          -0.248,
          0.496,
          -0.44,
          1.584,
          -1.664,
          -0.832,
          0.064,
          1.156,
          -1.932,
          -0.324,
          -1.532,
          0.46,
          -0.884,
          -1.904,
          -0.268,
          1.292,
          0.7,
          -1.124,
          -0.692,
          0.052,
          -1.332,
          0.808,
          -0.856,
          -1.108,
          0.204,
          -1.676,
          -0.076,
          -0.444,
          0.18,
          -1.044,
          1.152,
          -0.952,
          -1.232,
          -1.904,
          1.664
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
Code from [LayerTestBase.java:120](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L120) executed in 0.01 seconds: 
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
    	[ [ 1.392, -0.684, 1.6, 0.336, 1.092, 1.152, -0.708 ], [ 1.416, 0.408, 0.74, -0.892, -1.172, 0.788, 0.44 ], [ 1.092, -0.964, 0.812, 0.208, 1.672, 1.9, -0.848 ] ],
    	[ [ 1.584, 1.804, -1.052, -1.188, 0.676, -1.38, -1.616 ], [ 1.256, -0.972, 0.78, -1.348, 1.92, -0.976, 0.976 ], [ -0.092, 1.068, -0.94, -1.916, 0.984, 1.584, -1.14 ] ],
    	[ [ 1.84, -1.268, -1.324, -0.268, 1.904, -0.344, 1.888 ], [ 1.78, 1.696, -0.928, -0.34, 1.64, -0.548, -0.352 ], [ 0.844, -0.196, 1.396, 0.848, -1.536, -1.64, 0.396 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 1.5238399999999996, 3.6648319999999996, 1.9370559999999994 ] ]
    ]
```



### Batch Execution
Code from [LayerTestBase.java:138](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L138) executed in 0.10 seconds: 
```java
    BatchingTester batchingTester = getBatchingTester();
    return batchingTester==null?null:batchingTester.test(layer, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (660#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (100#)}
```



### Differential Validation
Code from [LayerTestBase.java:144](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 1.15 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ 1.392, -0.684, 1.6, 0.336, 1.092, 1.152, -0.708 ], [ 1.416, 0.408, 0.74, -0.892, -1.172, 0.788, 0.44 ], [ 1.092, -0.964, 0.812, 0.208, 1.672, 1.9, -0.848 ] ],
    	[ [ 1.584, 1.804, -1.052, -1.188, 0.676, -1.38, -1.616 ], [ 1.256, -0.972, 0.78, -1.348, 1.92, -0.976, 0.976 ], [ -0.092, 1.068, -0.94, -1.916, 0.984, 1.584, -1.14 ] ],
    	[ [ 1.84, -1.268, -1.324, -0.268, 1.904, -0.344, 1.888 ], [ 1.78, 1.696, -0.928, -0.34, 1.64, -0.548, -0.352 ], [ 0.844, -0.196, 1.396, 0.848, -1.536, -1.64, 0.396 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.033195219468134356, negative=28, min=0.396, max=0.396, mean=0.24253968253968256, count=63.0, positive=35, stdDev=1.1765277514842551, zeros=0}
    Output: [
    	[ [ 1.5238399999999996, 3.6648319999999996, 1.9370559999999994 ] ]
    ]
    Outputs Statistics: {meanExponent=0.34471187218881855, negative=0, min=1.9370559999999994, max=1.9370559999999994, mean=2.3752426666666664, count=3.0, positive=3, stdDev=0.927350114447732, zeros=0}
    Feedback for input 0
    Inputs Values: [
    	[ [ 1.392, -0.684, 1.6, 0.336, 1.092, 1.152, -0.708 ], [ 1.416, 0.408, 0.74, -0.892, -1.172, 0.788, 0.44 ], [ 1.092, -0.964, 0.812, 0.208, 1.672, 1.9, -0.848 ] ],
    	[ [ 1.584, 1.804, -1.052, -1.188, 0.676, -1.38, -1.616 ], [ 1.256, -0.972, 0.78, -1.348, 1.92, -0.976, 0.976 ], [ -0.092, 1.068, -0.94, -1.916, 0.984, 1.584, -1.14 ] ],
    	[ [ 1.84, -1.268, -1.324, -0.268, 1.904, -0.344, 1.888 ], [ 1.78, 1.696, -0.928, -0.34, 1.64, -0.548, -0.352 ], [ 0.844, -0.196, 1.396, 0.848, -1.536, -1.64, 0.396 ] ]
    ]
    Value Statistics: {meanExponent=-0.033195219468134356, negative=28, min=0.396, max=0.396, mean=0.24253968253968256, count=63.0, positive=35, stdDev=1.1765277514842551, zeros=0}
    Implemented Feedback: [ [ 1.572, 1.04, 1.944 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], ... ]
    Implemented Statistics: {meanExponent=-0.12292980168929335, negative=11, min=0.0, max=0.0, mean=0.003915343915343914, count=189.0, positive=10, stdDev=0.394589644
```
...[skipping 204 bytes](etc/1.txt)...
```
    .0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], ... ]
    Measured Statistics: {meanExponent=-0.12292980168909672, negative=11, min=0.0, max=0.0, mean=0.003915343915433007, count=189.0, positive=10, stdDev=0.39458964449458633, zeros=168}
    Feedback Error: [ [ -1.425526363618701E-12, 3.2605029787191597E-12, 2.610356375498668E-12 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], ... ]
    Error Statistics: {meanExponent=-12.126107852648238, negative=6, min=0.0, max=0.0, mean=8.909121236159185E-14, count=189.0, positive=15, stdDev=5.799746279629891E-13, zeros=168}
    Learning Gradient for weight set 0
    Weights: [ 1.572, 0.18, 0.492, 0.424, -1.428, -0.648, 1.156, 1.248, ... ]
    Implemented Gradient: [ [ 1.392, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], ... ]
    Implemented Statistics: {meanExponent=-0.04873386111627014, negative=6, min=0.0, max=0.0, mean=0.022116402116402114, count=567.0, positive=15, stdDev=0.20582523843928766, zeros=546}
    Measured Gradient: [ [ 1.3920000000000599, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], ... ]
    Measured Statistics: {meanExponent=-0.048733861116037745, negative=6, min=0.0, max=0.0, mean=0.022116402116415423, count=567.0, positive=15, stdDev=0.2058252384393572, zeros=546}
    Gradient Error: [ [ 5.995204332975845E-14, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], ... ]
    Error Statistics: {meanExponent=-12.360402952927673, negative=7, min=0.0, max=0.0, mean=1.3307109676373536E-14, count=567.0, positive=14, stdDev=1.9852802121759062E-13, zeros=546}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 6.0568e-14 +- 3.3482e-13 [0.0000e+00 - 3.2605e-12] (756#)
    relativeTol: 9.3018e-13 +- 1.4093e-12 [3.3077e-15 - 6.4476e-12] (42#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=6.0568e-14 +- 3.3482e-13 [0.0000e+00 - 3.2605e-12] (756#), relativeTol=9.3018e-13 +- 1.4093e-12 [3.3077e-15 - 6.4476e-12] (42#)}
```



### Performance
Code from [LayerTestBase.java:149](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L149) executed in 0.83 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 29.2247 +- 3.3755 [24.9357 - 40.6323]
    Learning performance: 22.3162 +- 3.0400 [19.0052 - 40.5069]
    
```

