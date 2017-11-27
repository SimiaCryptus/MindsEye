# ImgConcatLayer
## ImgConcatLayerTest
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
      "class": "com.simiacryptus.mindseye.layers.java.ImgConcatLayer",
      "id": "0910987d-3688-428c-a892-e2c400000e38",
      "isFrozen": false,
      "name": "ImgConcatLayer/0910987d-3688-428c-a892-e2c400000e38"
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
    	[ [ -1.808 ], [ -1.892 ] ],
    	[ [ -1.076 ], [ 1.648 ] ]
    ],
    [
    	[ [ -1.644 ], [ 1.108 ] ],
    	[ [ 0.796 ], [ -0.244 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ -1.808, -1.644 ], [ -1.892, 1.108 ] ],
    	[ [ -1.076, 0.796 ], [ 1.648, -0.244 ] ]
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
    absoluteTol: 7.0089e-13 +- 1.9676e-12 [0.0000e+00 - 6.5510e-12] (64#)
    relativeTol: 2.8036e-12 +- 9.3017e-13 [4.9993e-13 - 3.2755e-12] (8#)
    
```

### Performance
Code from [LayerTestBase.java:144](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.00 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 0.0686 +- 0.0275 [0.0599 - 0.3192]
    Learning performance: 0.0332 +- 0.0086 [0.0285 - 0.0884]
    
```

