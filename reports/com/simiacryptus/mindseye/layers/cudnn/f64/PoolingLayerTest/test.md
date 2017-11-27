# PoolingLayer
## PoolingLayerTest
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
      "class": "com.simiacryptus.mindseye.layers.cudnn.f64.PoolingLayer",
      "id": "0910987d-3688-428c-a892-e2c400000dde",
      "isFrozen": false,
      "name": "PoolingLayer/0910987d-3688-428c-a892-e2c400000dde",
      "mode": 0,
      "windowX": 2,
      "windowY": 2,
      "paddingX": 0,
      "paddingY": 0,
      "strideX": 2,
      "strideY": 2
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
    	[ [ 1.12, 0.236 ], [ -0.436, 1.332 ] ],
    	[ [ 1.656, -1.904 ], [ 1.196, 0.652 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 1.656, 1.332 ] ]
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
    absoluteTol: 8.1887e-13 +- 2.1665e-12 [0.0000e+00 - 6.5510e-12] (16#)
    relativeTol: 3.2755e-12 +- 0.0000e+00 [3.2755e-12 - 3.2755e-12] (2#)
    
```

### Performance
Code from [LayerTestBase.java:144](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.04 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 1.6703 +- 0.2840 [1.3793 - 3.5964]
    Learning performance: 1.3181 +- 0.2242 [1.0459 - 2.5392]
    
```

