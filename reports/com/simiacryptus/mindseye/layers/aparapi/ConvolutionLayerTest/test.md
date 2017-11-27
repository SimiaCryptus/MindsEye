# ConvolutionLayer
## ConvolutionLayerTest
### Json Serialization
Code from [LayerTestBase.java:84](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L84) executed in 0.04 seconds: 
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
      "id": "0910987d-3688-428c-a892-e2c400000001",
      "isFrozen": false,
      "name": "ConvolutionLayer/0910987d-3688-428c-a892-e2c400000001",
      "filter": {
        "dimensions": [
          3,
          3,
          4
        ],
        "data": [
          -1.492,
          0.908,
          0.076,
          1.98,
          1.368,
          -1.776,
          0.708,
          -1.744,
          1.524,
          1.276,
          -0.796,
          -1.072,
          0.356,
          -1.888,
          -1.756,
          -0.756,
          -0.464,
          0.824,
          1.092,
          -1.64,
          -0.456,
          1.304,
          1.828,
          -0.976,
          1.1,
          -1.596,
          1.312,
          1.98,
          0.18,
          1.528,
          -1.464,
          -1.048,
          -1.864,
          0.36,
          1.74,
          1.232
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
Code from [LayerTestBase.java:121](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L121) executed in 0.34 seconds: 
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
    	[ [ -0.08, -0.032 ], [ 0.484, 0.236 ], [ 0.736, -1.244 ] ],
    	[ [ 1.496, -1.14 ], [ -0.844, 1.816 ], [ -1.372, -0.752 ] ],
    	[ [ -0.924, 1.784 ], [ 0.908, -1.864 ], [ 0.652, 1.264 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 4.602336, 4.5620639999999995 ], [ 5.720512, -9.729552 ], [ -4.785056, 2.004528 ] ],
    	[ [ -6.572912, -6.061967999999999 ], [ 1.8152479999999998, 1.6119360000000003 ], [ -2.192016000000001, 5.669984000000001 ] ],
    	[ [ 3.442048, 1.994192 ], [ -4.132608, 1.7275840000000002 ], [ 8.860880000000002, -0.8675039999999999 ] ]
    ]
```



### Differential Validation
Code from [LayerTestBase.java:139](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L139) executed in 0.53 seconds: 
```java
    getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Finite-Difference Derivative Accuracy:
    absoluteTol: 1.0868e-11 +- 1.9493e-11 [0.0000e+00 - 1.5368e-10] (972#)
    relativeTol: 2.8025e-11 +- 6.7422e-11 [2.3042e-13 - 8.1829e-10] (392#)
    
```

### Performance
Code from [LayerTestBase.java:144](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.72 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 24.7749 +- 3.7479 [20.9288 - 39.7915]
    Learning performance: 20.6582 +- 3.2359 [17.5832 - 36.8050]
    
```

