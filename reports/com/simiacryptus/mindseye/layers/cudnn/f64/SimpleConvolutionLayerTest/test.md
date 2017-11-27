# SimpleConvolutionLayer
## SimpleConvolutionLayerTest
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
      "class": "com.simiacryptus.mindseye.layers.cudnn.f64.SimpleConvolutionLayer",
      "id": "0910987d-3688-428c-a892-e2c400000de2",
      "isFrozen": false,
      "name": "SimpleConvolutionLayer/0910987d-3688-428c-a892-e2c400000de2",
      "filter": {
        "dimensions": [
          3,
          3,
          1
        ],
        "data": [
          -1.8,
          0.432,
          -1.416,
          -0.272,
          -0.224,
          1.0,
          -1.568,
          -0.532,
          0.332
        ]
      },
      "strideX": 1,
      "strideY": 1,
      "simple": false
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
    	[ [ -1.796 ], [ 0.384 ], [ 0.316 ] ],
    	[ [ -0.216 ], [ 1.8 ], [ -1.688 ] ],
    	[ [ 0.468 ], [ 1.448 ], [ -0.48 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ -2.6130560000000003 ], [ 3.8934559999999996 ], [ -2.6383360000000002 ] ],
    	[ [ -4.247456 ], [ -1.9409120000000004 ], [ -2.275904 ] ],
    	[ [ -2.244096 ], [ 3.337808 ], [ -1.7532159999999999 ] ]
    ]
```



### Reference Implementation
Code from [LayerTestBase.java:132](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L132) executed in 0.01 seconds: 
```java
    System.out.println(new GsonBuilder().setPrettyPrinting().create().toJson(referenceLayer.getJson()));
    getEquivalencyTester().test(referenceLayer, layer, inputPrototype);
```
Logging: 
```
    {
      "class": "com.simiacryptus.mindseye.layers.aparapi.ConvolutionLayer",
      "id": "0910987d-3688-428c-a892-e2c400000de4",
      "isFrozen": false,
      "name": "ConvolutionLayer/0910987d-3688-428c-a892-e2c400000de4",
      "filter": {
        "dimensions": [
          3,
          3,
          1
        ],
        "data": [
          -1.8,
          0.432,
          -1.416,
          -0.272,
          -0.224,
          1.0,
          -1.568,
          -0.532,
          0.332
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
    	[ [ -1.796 ], [ 0.384 ], [ 0.316 ] ],
    	[ [ -0.216 ], [ 1.8 ], [ -1.688 ] ],
    	[ [ 0.468 ], [ 1.448 ], [ -0.48 ] ]
    ]
    Error: [
    	[ [ 0.0 ], [ 0.0 ], [ 0.0 ] ],
    	[ [ 0.0 ], [ 0.0 ], [ 0.0 ] ],
    	[ [ 0.0 ], [ 0.0 ], [ 0.0 ] ]
    ]
    Accuracy:
    absoluteTol: 0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (9#)
    relativeTol: 0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (9#)
    
```

### Differential Validation
Code from [LayerTestBase.java:139](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L139) executed in 0.03 seconds: 
```java
    getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Finite-Difference Derivative Accuracy:
    absoluteTol: 8.8981e-12 +- 1.1033e-11 [0.0000e+00 - 5.0960e-11] (162#)
    relativeTol: 1.6449e-11 +- 1.7964e-11 [9.1949e-14 - 6.4924e-11] (98#)
    
```

### Performance
Code from [LayerTestBase.java:144](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.10 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 3.5389 +- 0.2882 [3.3400 - 5.7309]
    Learning performance: 2.8579 +- 0.2652 [2.4822 - 4.8845]
    
```

