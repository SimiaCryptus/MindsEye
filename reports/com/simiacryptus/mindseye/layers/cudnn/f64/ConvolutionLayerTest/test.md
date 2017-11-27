# ConvolutionLayer
## ConvolutionLayerTest
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
      "class": "com.simiacryptus.mindseye.layers.cudnn.f64.ConvolutionLayer",
      "id": "0910987d-3688-428c-a892-e2c40000042d",
      "isFrozen": false,
      "name": "ConvolutionLayer/0910987d-3688-428c-a892-e2c40000042d",
      "filter": {
        "dimensions": [
          3,
          3,
          4
        ],
        "data": [
          1.548,
          0.236,
          0.868,
          -1.848,
          -1.928,
          -0.964,
          1.476,
          -1.508,
          -1.644,
          0.736,
          -1.616,
          0.556,
          0.6,
          0.244,
          -0.096,
          -1.56,
          1.096,
          -1.132,
          1.956,
          1.416,
          -0.36,
          0.508,
          -1.756,
          0.576,
          0.528,
          0.372,
          -0.832,
          1.88,
          0.792,
          1.084,
          -0.384,
          -1.38,
          -1.392,
          -0.828,
          -0.216,
          -1.848
        ]
      },
      "strideX": 1,
      "strideY": 1
    }
```



### Example Input/Output Pair
Code from [LayerTestBase.java:121](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L121) executed in 0.00 seconds: 
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
    	[ [ -0.528, -1.028 ], [ -1.56, 0.828 ], [ -1.412, 0.868 ] ],
    	[ [ 1.224, -0.812 ], [ -1.528, 0.544 ], [ 1.472, -0.268 ] ],
    	[ [ 1.44, 1.764 ], [ -1.708, 1.528 ], [ 1.024, 1.304 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ -0.34828799999999976, 5.410864 ], [ 9.095808, -0.6939200000000001 ], [ -0.9658720000000002, -0.5116159999999998 ] ],
    	[ [ -3.679824, 7.633344000000001 ], [ 13.101983999999998, -2.6605599999999985 ], [ 0.9319200000000007, -0.387152 ] ],
    	[ [ -7.283136, 2.6402720000000004 ], [ 3.0065119999999994, -1.9178239999999995 ], [ -0.6339680000000001, -2.7955520000000003 ] ]
    ]
```



### Reference Implementation
Code from [LayerTestBase.java:132](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L132) executed in 0.02 seconds: 
```java
    System.out.println(new GsonBuilder().setPrettyPrinting().create().toJson(referenceLayer.getJson()));
    getEquivalencyTester().test(referenceLayer, layer, inputPrototype);
```
Logging: 
```
    {
      "class": "com.simiacryptus.mindseye.layers.aparapi.ConvolutionLayer",
      "id": "0910987d-3688-428c-a892-e2c400000435",
      "isFrozen": false,
      "name": "ConvolutionLayer/0910987d-3688-428c-a892-e2c400000435",
      "filter": {
        "dimensions": [
          3,
          3,
          4
        ],
        "data": [
          1.548,
          0.236,
          0.868,
          -1.848,
          -1.928,
          -0.964,
          1.476,
          -1.508,
          -1.644,
          0.736,
          -1.616,
          0.556,
          0.6,
          0.244,
          -0.096,
          -1.56,
          1.096,
          -1.132,
          1.956,
          1.416,
          -0.36,
          0.508,
          -1.756,
          0.576,
          0.528,
          0.372,
          -0.832,
          1.88,
          0.792,
          1.084,
          -0.384,
          -1.38,
          -1.392,
          -0.828,
          -0.216,
          -1.848
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
    	[ [ -0.528, -1.028 ], [ -1.56, 0.828 ], [ -1.412, 0.868 ] ],
    	[ [ 1.224, -0.812 ], [ -1.528, 0.544 ], [ 1.472, -0.268 ] ],
    	[ [ 1.44, 1.764 ], [ -1.708, 1.528 ], [ 1.024, 1.304 ] ]
    ]
    Error: [
    	[ [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ] ],
    	[ [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ] ],
    	[ [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ] ]
    ]
    Accuracy:
    absoluteTol: 0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (18#)
    relativeTol: 0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (18#)
    
```

### Differential Validation
Code from [LayerTestBase.java:139](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L139) executed in 0.21 seconds: 
```java
    getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Finite-Difference Derivative Accuracy:
    absoluteTol: 1.0587e-11 +- 2.1960e-11 [0.0000e+00 - 1.4724e-10] (972#)
    relativeTol: 1.7201e-11 +- 2.5728e-11 [8.6275e-14 - 1.9941e-10] (392#)
    
```

### Performance
Code from [LayerTestBase.java:144](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.31 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 14.8428 +- 5.0215 [6.9535 - 33.0632]
    Learning performance: 5.4706 +- 1.0551 [4.4742 - 11.8751]
    
```

