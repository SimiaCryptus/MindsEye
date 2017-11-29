# ConvolutionLayer
## IrregularTest
### Json Serialization
Code from [LayerTestBase.java:84](../../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L84) executed in 0.06 seconds: 
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
      "class": "com.simiacryptus.mindseye.layers.cudnn.f64.ConvolutionLayer",
      "id": "453f716d-88bc-4c50-9d54-78bf00000002",
      "isFrozen": false,
      "name": "ConvolutionLayer/453f716d-88bc-4c50-9d54-78bf00000002",
      "filter": {
        "dimensions": [
          3,
          3,
          6
        ],
        "data": [
          1.412,
          -0.38,
          1.988,
          -1.704,
          -0.7,
          -1.932,
          0.888,
          1.252,
          -0.864,
          -1.36,
          1.848,
          -1.644,
          1.572,
          -0.404,
          1.444,
          -1.664,
          0.288,
          1.024,
          1.732,
          1.464,
          -1.544,
          1.552,
          -0.792,
          0.208,
          -0.168,
          1.276,
          1.036,
          -0.456,
          -1.668,
          -1.112,
          0.028,
          0.924,
          -0.392,
          -1.98,
          -0.776,
          1.92,
          1.504,
          -1.96,
          -0.304,
          -1.22,
          -1.036,
          1.772,
          1.372,
          0.2,
          -1.956,
          1.624,
          -1.428,
          -0.556,
          1.968,
          -1.08,
          -0.356,
          1.648,
          -0.408,
          -1.236
        ]
      },
      "strideX": 1,
      "strideY": 1
    }
```



### Example Input/Output Pair
Code from [LayerTestBase.java:121](../../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L121) executed in 0.02 seconds: 
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
    	[ [ -0.68, -1.064 ], [ 1.872, -0.636 ], [ -1.296, -1.412 ] ],
    	[ [ 0.992, 1.836 ], [ -1.576, -1.124 ], [ -1.188, -1.536 ] ],
    	[ [ -1.536, -1.308 ], [ 1.24, -1.456 ], [ 1.148, 1.064 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ -3.509376, 5.855408000000001, 5.9343200000000005 ], [ -0.1534559999999997, -1.0658719999999993, -7.461312000000003 ], [ 5.247168, 3.4850719999999993, -1.2546559999999998 ] ],
    	[ [ 14.630927999999999, -13.45888, -10.945024 ], [ -2.72576, 13.316016, 8.323312 ], [ 0.5878559999999999, -3.3744, 5.273823999999999 ] ],
    	[ [ -2.69544, 14.739392, 9.134688 ], [ 0.16736000000000015, -4.083312000000001, 0.844448 ], [ 4.962784, -5.35272, 0.17419199999999974 ] ]
    ]
```



### Reference Implementation
Code from [LayerTestBase.java:132](../../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L132) executed in 0.74 seconds: 
```java
    System.out.println(new GsonBuilder().setPrettyPrinting().create().toJson(referenceLayer.getJson()));
    getEquivalencyTester().test(referenceLayer, layer, inputPrototype);
```
Logging: 
```
    {
      "class": "com.simiacryptus.mindseye.layers.aparapi.ConvolutionLayer",
      "id": "453f716d-88bc-4c50-9d54-78bf0000000c",
      "isFrozen": false,
      "name": "ConvolutionLayer/453f716d-88bc-4c50-9d54-78bf0000000c",
      "filter": {
        "dimensions": [
          3,
          3,
          6
        ],
        "data": [
          1.412,
          -0.38,
          1.988,
          -1.704,
          -0.7,
          -1.932,
          0.888,
          1.252,
          -0.864,
          -1.36,
          1.848,
          -1.644,
          1.572,
          -0.404,
          1.444,
          -1.664,
          0.288,
          1.024,
          1.732,
          1.464,
          -1.544,
          1.552,
          -0.792,
          0.208,
          -0.168,
          1.276,
          1.036,
          -0.456,
          -1.668,
          -1.112,
          0.028,
          0.924,
          -0.392,
          -1.98,
          -0.776,
          1.92,
          1.504,
          -1.96,
          -0.304,
          -1.22,
          -1.036,
          1.772,
          1.372,
          0.2,
          -1.956,
          1.624,
          -1.428,
          -0.556,
          1.968,
          -1.08,
          -0.356,
          1.648,
          -0.408,
          -1.236
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
    Inputs: [
    	[ [ -0.68, -1.064 ], [ 1.872, -0.636 ], [ -1.296, -1.412 ] ],
    	[ [ 0.992, 1.836 ], [ -1.576, -1.124 ], [ -1.188, -1.536 ] ],
    	[ [ -1.536, -1.308 ], [ 1.24, -1.456 ], [ 1.148, 1.064 ] ]
    ]
    Error: [
    	[ [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ] ],
    	[ [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ] ],
    	[ [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ] ]
    ]
    Accuracy:
    absoluteTol: 0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (27#)
    relativeTol: 0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (27#)
    
```

### Differential Validation
Code from [LayerTestBase.java:139](../../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L139) executed in 0.56 seconds: 
```java
    getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Finite-Difference Derivative Accuracy:
    absoluteTol: 1.0371e-11 +- 2.4238e-11 [0.0000e+00 - 1.6929e-10] (1944#)
    relativeTol: 2.0929e-11 +- 4.2455e-11 [5.1776e-14 - 6.5969e-10] (588#)
    
```

### Performance
Code from [LayerTestBase.java:144](../../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.55 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 23.5874 +- 16.0767 [11.7810 - 157.7184]
    Learning performance: 9.3388 +- 1.5945 [7.4094 - 15.9930]
    
```

