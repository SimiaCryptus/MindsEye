# AvgSubsampleLayer
## AvgSubsampleLayerTest
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
      "class": "com.simiacryptus.mindseye.layers.java.AvgSubsampleLayer",
      "id": "0910987d-3688-428c-a892-e2c400000df5",
      "isFrozen": false,
      "name": "AvgSubsampleLayer/0910987d-3688-428c-a892-e2c400000df5",
      "inner": [
        2,
        2,
        1
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
    	[ [ 1.576, 1.436, -1.576 ], [ 1.74, -1.068, 0.264 ] ],
    	[ [ 0.18, -1.332, -0.936 ], [ 1.696, -0.668, 1.124 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 1.298, -0.40800000000000003, -0.281 ] ]
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
    absoluteTol: 2.0956e-12 +- 3.9213e-12 [0.0000e+00 - 1.2740e-11] (36#)
    relativeTol: 1.2573e-11 +- 8.8953e-12 [3.2755e-12 - 2.5480e-11] (12#)
    
```

### Performance
Code from [LayerTestBase.java:144](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.00 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 0.1181 +- 0.0403 [0.0997 - 0.4218]
    Learning performance: 0.0030 +- 0.0115 [0.0000 - 0.1168]
    
```

