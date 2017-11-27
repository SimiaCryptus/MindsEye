# ProductInputsLayer
## ProductInputsLayerTest
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
      "class": "com.simiacryptus.mindseye.layers.cudnn.f32.ProductInputsLayer",
      "id": "0910987d-3688-428c-a892-e2c400000413",
      "isFrozen": false,
      "name": "ProductInputsLayer/0910987d-3688-428c-a892-e2c400000413"
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
    	[ [ 0.716, -1.136 ], [ 1.136, -0.748 ] ],
    	[ [ 0.052, 0.532 ], [ 0.848, -1.38 ] ]
    ],
    [
    	[ [ 1.376, 1.744 ], [ 0.564, -1.084 ] ],
    	[ [ -0.892, 1.428 ], [ 1.896, -1.956 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 0.9852160811424255, -1.9811840057373047 ], [ 0.6407040357589722, 0.8108320236206055 ] ],
    	[ [ -0.04638400301337242, 0.7596960067749023 ], [ 1.607807993888855, 2.699280023574829 ] ]
    ]
```



### Differential Validation
Code from [LayerTestBase.java:139](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L139) executed in 0.03 seconds: 
```java
    getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Finite-Difference Derivative Accuracy:
    absoluteTol: 4.6394e-05 +- 1.8759e-04 [0.0000e+00 - 1.4166e-03] (128#)
    relativeTol: 1.6975e-04 +- 1.6521e-04 [9.1593e-06 - 5.2002e-04] (16#)
    
```

### Performance
Code from [LayerTestBase.java:144](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.06 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 2.6175 +- 0.4871 [2.1202 - 5.7024]
    Learning performance: 0.5170 +- 0.1969 [0.2793 - 1.1798]
    
```

