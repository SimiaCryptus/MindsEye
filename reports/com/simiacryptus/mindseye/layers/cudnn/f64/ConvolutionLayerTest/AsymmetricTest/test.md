# ConvolutionLayer
## AsymmetricTest
### Json Serialization
Code from [LayerTestBase.java:84](../../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L84) executed in 0.05 seconds: 
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
      "id": "38461315-075d-47c5-a9f0-62e000000001",
      "isFrozen": false,
      "name": "ConvolutionLayer/38461315-075d-47c5-a9f0-62e000000001",
      "filter": {
        "dimensions": [
          1,
          1,
          8
        ],
        "data": [
          1.412,
          -0.188,
          -0.836,
          1.816,
          0.22,
          -0.748,
          -0.624,
          -0.496
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
    	[ [ 1.308, 1.784 ], [ -1.884, -1.112 ], [ -0.396, 0.516 ] ],
    	[ [ 0.264, -1.284 ], [ -1.216, 0.184 ], [ -0.872, -0.468 ] ],
    	[ [ -1.08, -1.884 ], [ -1.152, -0.424 ], [ -1.6, -0.82 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 2.239376, -1.580336, -2.206704, 1.490464 ], [ -2.904848, 1.1859680000000001, 2.268912, -2.869792 ], [ -0.445632, -0.31152, 0.009072000000000009, -0.9750720000000002 ] ],
    	[ [ 0.09028799999999998, 0.9108, 0.580512, 1.116288 ], [ -1.6765119999999998, 0.09097600000000002, 0.9017599999999999, -2.29952 ], [ -1.3342239999999999, 0.514, 1.021024, -1.351424 ] ],
    	[ [ -1.93944, 1.612272, 2.078496, -1.0268160000000002 ], [ -1.7199039999999999, 0.533728, 1.2276479999999999, -1.881728 ], [ -2.4396, 0.91416, 1.84928, -2.49888 ] ]
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
      "id": "38461315-075d-47c5-a9f0-62e00000000b",
      "isFrozen": false,
      "name": "ConvolutionLayer/38461315-075d-47c5-a9f0-62e00000000b",
      "filter": {
        "dimensions": [
          1,
          1,
          8
        ],
        "data": [
          1.412,
          -0.188,
          -0.836,
          1.816,
          0.22,
          -0.748,
          -0.624,
          -0.496
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
    	[ [ 1.308, 1.784 ], [ -1.884, -1.112 ], [ -0.396, 0.516 ] ],
    	[ [ 0.264, -1.284 ], [ -1.216, 0.184 ], [ -0.872, -0.468 ] ],
    	[ [ -1.08, -1.884 ], [ -1.152, -0.424 ], [ -1.6, -0.82 ] ]
    ]
    Error: [
    	[ [ 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0 ] ],
    	[ [ 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0 ] ],
    	[ [ 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0 ] ]
    ]
    Accuracy:
    absoluteTol: 0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (36#)
    relativeTol: 0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (36#)
    
```

### Differential Validation
Code from [LayerTestBase.java:139](../../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L139) executed in 0.35 seconds: 
```java
    getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Finite-Difference Derivative Accuracy:
    absoluteTol: 1.1689e-12 +- 3.8976e-12 [0.0000e+00 - 3.6863e-11] (936#)
    relativeTol: 5.6518e-12 +- 6.2132e-12 [5.1776e-14 - 3.2094e-11] (144#)
    
```

### Performance
Code from [LayerTestBase.java:144](../../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.42 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 18.2044 +- 19.0014 [11.2225 - 198.9434]
    Learning performance: 8.9107 +- 1.5028 [7.1758 - 18.8513]
    
```

