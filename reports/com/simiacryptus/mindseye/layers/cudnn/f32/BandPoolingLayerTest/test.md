# BandPoolingLayer
## BandPoolingLayerTest
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
      "class": "com.simiacryptus.mindseye.layers.cudnn.f32.BandPoolingLayer",
      "id": "0910987d-3688-428c-a892-e2c400000015",
      "isFrozen": false,
      "name": "BandPoolingLayer/0910987d-3688-428c-a892-e2c400000015",
      "mode": 0
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
    	[ [ 1.032, 0.86 ], [ -0.848, 0.768 ], [ -1.62, -0.66 ] ],
    	[ [ -1.156, 1.884 ], [ 0.992, -1.452 ], [ 1.116, -0.296 ] ],
    	[ [ -0.184, -1.812 ], [ -0.192, 0.264 ], [ -0.556, 0.168 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 1.1160000562667847, 1.8839999437332153 ] ]
    ]
```



### Differential Validation
Code from [LayerTestBase.java:139](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L139) executed in 0.01 seconds: 
```java
    getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Finite-Difference Derivative Accuracy:
    absoluteTol: 3.3114e-05 +- 1.7005e-04 [0.0000e+00 - 1.0262e-03] (36#)
    relativeTol: 2.9815e-04 +- 2.1519e-04 [8.2963e-05 - 5.1334e-04] (2#)
    
```

### Performance
Code from [LayerTestBase.java:144](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.05 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 2.0288 +- 0.4323 [1.6016 - 3.9441]
    Learning performance: 1.3709 +- 0.3669 [1.0117 - 3.8016]
    
```

