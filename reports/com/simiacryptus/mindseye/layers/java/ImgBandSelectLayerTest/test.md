# ImgBandSelectLayer
## ImgBandSelectLayerTest
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
      "class": "com.simiacryptus.mindseye.layers.java.ImgBandSelectLayer",
      "id": "0910987d-3688-428c-a892-e2c400000e34",
      "isFrozen": false,
      "name": "ImgBandSelectLayer/0910987d-3688-428c-a892-e2c400000e34",
      "bands": [
        0,
        2
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
    	[ [ -0.952, 0.848, 0.292 ], [ 0.58, -1.052, -1.42 ] ],
    	[ [ -1.24, -0.168, 1.004 ], [ 0.24, -0.432, -1.628 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ -0.952, 0.292 ], [ 0.58, -1.42 ] ],
    	[ [ -1.24, 1.004 ], [ 0.24, -1.628 ] ]
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
    absoluteTol: 3.8860e-13 +- 1.4455e-12 [0.0000e+00 - 6.5510e-12] (96#)
    relativeTol: 2.3316e-12 +- 1.1336e-12 [4.9993e-13 - 3.2755e-12] (8#)
    
```

### Performance
Code from [LayerTestBase.java:144](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.00 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 0.0671 +- 0.0326 [0.0456 - 0.3220]
    Learning performance: 0.0016 +- 0.0021 [0.0000 - 0.0171]
    
```

