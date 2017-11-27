# ImgBandBiasLayer
## ImgBandBiasLayerTest
### Json Serialization
Code from [LayerTestBase.java:84](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L84) executed in 0.00 seconds: 
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
      "class": "com.simiacryptus.mindseye.layers.java.ImgBandBiasLayer",
      "id": "0910987d-3688-428c-a892-e2c400000e2c",
      "isFrozen": false,
      "name": "ImgBandBiasLayer/0910987d-3688-428c-a892-e2c400000e2c",
      "bias": [
        0.0,
        0.0,
        0.0
      ]
    }
```



### Example Input/Output Pair
Code from [LayerTestBase.java:121](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L121) executed in 0.00 seconds: 
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
    	[ [ -1.852, 0.432, -0.124 ], [ -0.752, 1.464, -1.16 ] ],
    	[ [ 0.048, -1.172, 1.128 ], [ 0.016, -1.952, 1.852 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ -1.852, 0.432, -0.124 ], [ -0.752, 1.464, -1.16 ] ],
    	[ [ 0.048, -1.172, 1.128 ], [ 0.016, -1.952, 1.852 ] ]
    ]
```



### Differential Validation
Code from [LayerTestBase.java:139](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L139) executed in 0.00 seconds: 
```java
    getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Finite-Difference Derivative Accuracy:
    absoluteTol: 5.7936e-13 +- 1.8017e-12 [0.0000e+00 - 6.5510e-12] (180#)
    relativeTol: 2.1726e-12 +- 1.4127e-12 [2.0428e-14 - 3.2755e-12] (24#)
    
```

### Performance
Code from [LayerTestBase.java:144](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.00 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 0.0716 +- 0.0172 [0.0627 - 0.1767]
    Learning performance: 0.0698 +- 0.0569 [0.0456 - 0.5101]
    
```

