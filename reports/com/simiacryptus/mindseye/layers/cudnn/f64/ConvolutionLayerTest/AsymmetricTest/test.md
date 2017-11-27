# ConvolutionLayer
## AsymmetricTest
### Json Serialization
Code from [LayerTestBase.java:84](../../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L84) executed in 0.00 seconds: 
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
      "id": "0910987d-3688-428c-a892-e2c4000007b0",
      "isFrozen": false,
      "name": "ConvolutionLayer/0910987d-3688-428c-a892-e2c4000007b0",
      "filter": {
        "dimensions": [
          3,
          3,
          8
        ],
        "data": [
          0.128,
          0.016,
          1.652,
          1.636,
          0.316,
          0.792,
          -0.316,
          0.912,
          0.688,
          0.112,
          -1.812,
          -1.256,
          -0.696,
          0.204,
          1.808,
          -0.804,
          0.172,
          -1.936,
          -1.136,
          1.324,
          0.656,
          1.4,
          0.2,
          0.732,
          0.904,
          0.704,
          -0.424,
          1.588,
          -0.712,
          0.38,
          1.248,
          0.64,
          1.5,
          -1.336,
          1.568,
          -0.06,
          -0.112,
          -0.636,
          0.944,
          1.656,
          -0.668,
          -0.216,
          1.792,
          -1.816,
          1.508,
          -1.9,
          0.64,
          1.248,
          -1.616,
          0.228,
          -1.552,
          -0.932,
          1.348,
          0.848,
          0.24,
          1.292,
          -1.868,
          1.144,
          1.9,
          -0.18,
          0.988,
          0.916,
          -1.472,
          1.196,
          -0.668,
          0.196,
          -1.968,
          1.44,
          1.304,
          -1.432,
          -0.408,
          1.104
        ]
      },
      "strideX": 1,
      "strideY": 1
    }
```



### Example Input/Output Pair
Code from [LayerTestBase.java:121](../../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L121) executed in 0.00 seconds: 
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
    	[ [ 1.216, 0.964 ], [ 1.528, -1.44 ], [ -0.1, -1.864 ] ],
    	[ [ 0.848, 1.752 ], [ -1.952, 1.728 ], [ 0.92, 1.344 ] ],
    	[ [ -0.428, -1.44 ], [ 0.844, -0.34 ], [ 0.812, -1.328 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 4.52584, -10.145744, 8.061072, -1.3823359999999998 ], [ 4.493984, -5.718816, -2.21296, -4.676159999999999 ], [ 12.66632, -4.9770080000000005, -1.0367360000000003, -1.128192 ] ],
    	[ [ -3.0516, 5.566960000000001, 4.268464000000001, 9.913984 ], [ -4.5075520000000004, 7.218928, 8.650464, 5.360736 ], [ -8.068896, 2.239712000000001, 4.724992, -2.51752 ] ],
    	[ [ -0.2438719999999995, 1.2597759999999996, -6.346464, 0.4322080000000005 ], [ 7.670912, -10.086656000000001, -9.323712, 2.0976000000000004 ], [ 4.232048, 4.3716, -3.3625119999999997, 5.22688 ] ]
    ]
```



### Reference Implementation
Code from [LayerTestBase.java:132](../../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L132) executed in 0.02 seconds: 
```java
    System.out.println(new GsonBuilder().setPrettyPrinting().create().toJson(referenceLayer.getJson()));
    getEquivalencyTester().test(referenceLayer, layer, inputPrototype);
```
Logging: 
```
    {
      "class": "com.simiacryptus.mindseye.layers.aparapi.ConvolutionLayer",
      "id": "0910987d-3688-428c-a892-e2c4000007ba",
      "isFrozen": false,
      "name": "ConvolutionLayer/0910987d-3688-428c-a892-e2c4000007ba",
      "filter": {
        "dimensions": [
          3,
          3,
          8
        ],
        "data": [
          0.128,
          0.016,
          1.652,
          1.636,
          0.316,
          0.792,
          -0.316,
          0.912,
          0.688,
          0.112,
          -1.812,
          -1.256,
          -0.696,
          0.204,
          1.808,
          -0.804,
          0.172,
          -1.936,
          -1.136,
          1.324,
          0.656,
          1.4,
          0.2,
          0.732,
          0.904,
          0.704,
          -0.424,
          1.588,
          -0.712,
          0.38,
          1.248,
          0.64,
          1.5,
          -1.336,
          1.568,
          -0.06,
          -0.112,
          -0.636,
          0.944,
          1.656,
          -0.668,
          -0.216,
          1.792,
          -1.816,
          1.508,
          -1.9,
          0.64,
          1.248,
          -1.616,
          0.228,
          -1.552,
          -0.932,
          1.348,
          0.848,
          0.24,
          1.292,
          -1.868,
          1.144,
          1.9,
          -0.18,
          0.988,
          0.916,
          -1.472,
          1.196,
          -0.668,
          0.196,
          -1.968,
          1.44,
          1.304,
          -1.432,
          -0.408,
          1.104
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
    	[ [ 1.216, 0.964 ], [ 1.528, -1.44 ], [ -0.1, -1.864 ] ],
    	[ [ 0.848, 1.752 ], [ -1.952, 1.728 ], [ 0.92, 1.344 ] ],
    	[ [ -0.428, -1.44 ], [ 0.844, -0.34 ], [ 0.812, -1.328 ] ]
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
Code from [LayerTestBase.java:139](../../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L139) executed in 0.48 seconds: 
```java
    getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Finite-Difference Derivative Accuracy:
    absoluteTol: 7.5649e-12 +- 2.0041e-11 [0.0000e+00 - 1.8076e-10] (3240#)
    relativeTol: 2.7842e-11 +- 5.0590e-11 [2.3040e-13 - 3.7420e-10] (784#)
    
```

### Performance
Code from [LayerTestBase.java:144](../../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.43 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 16.3140 +- 4.1456 [10.7722 - 25.0952]
    Learning performance: 8.5488 +- 2.0253 [6.8452 - 18.0762]
    
```

