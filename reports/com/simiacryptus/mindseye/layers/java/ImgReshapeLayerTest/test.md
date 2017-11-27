# ImgReshapeLayer
## ImgReshapeLayerTest
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
      "class": "com.simiacryptus.mindseye.layers.java.ImgReshapeLayer",
      "id": "0910987d-3688-428c-a892-e2c400000e3c",
      "isFrozen": false,
      "name": "ImgReshapeLayer/0910987d-3688-428c-a892-e2c400000e3c",
      "kernelSizeX": 2,
      "kernelSizeY": 2,
      "expand": false
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
    	[ [ 1.78, 1.512, -1.108 ], [ 0.792, -1.472, -1.004 ] ],
    	[ [ -1.008, 1.82, 0.008 ], [ -0.236, -0.564, 0.788 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 1.78, 1.512, -1.108, 0.792, -1.472, -1.004, -1.008, 1.82, 0.008, -0.236 ] ]
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
    absoluteTol: 4.2049e-13 +- 1.5323e-12 [0.0000e+00 - 6.5510e-12] (144#)
    relativeTol: 2.5230e-12 +- 1.0994e-12 [2.0428e-14 - 3.2755e-12] (12#)
    
```

### Performance
Code from [LayerTestBase.java:144](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.00 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 0.0614 +- 0.0116 [0.0570 - 0.1453]
    Learning performance: 0.0016 +- 0.0021 [0.0000 - 0.0142]
    
```

