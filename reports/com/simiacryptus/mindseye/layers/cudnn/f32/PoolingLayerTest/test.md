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
      "class": "com.simiacryptus.mindseye.layers.cudnn.f32.PoolingLayer",
      "id": "e2d0bffa-47dc-4875-864f-3d3d0000036a",
      "isFrozen": false,
      "name": "PoolingLayer/e2d0bffa-47dc-4875-864f-3d3d0000036a",
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
    	[ [ -0.444, -0.048 ], [ 1.908, 0.272 ] ],
    	[ [ -0.424, -1.432 ], [ 1.824, -1.736 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 1.9079999923706055, 0.2720000147819519 ] ]
    ]
```



### Differential Validation
Code from [LayerTestBase.java:139](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L139) executed in 0.01 seconds: 
```java
    getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Feedback for input 0
    Inputs: [
    	[ [ -0.444, -0.048 ], [ 1.908, 0.272 ] ],
    	[ [ -0.424, -1.432 ], [ 1.824, -1.736 ] ]
    ]
    Output: [
    	[ [ 1.9079999923706055, 0.2720000147819519 ] ]
    ]
    Measured: [ [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 1.0001659393310547, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.9998679161071777 ], [ 0.0, 0.0 ] ]
    Implemented: [ [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 1.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 1.0 ], [ 0.0, 0.0 ] ]
    Error: [ [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 1.659393310546875E-4, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, -1.3208389282226562E-4 ], [ 0.0, 0.0 ] ]
    Finite-Difference Derivative Accuracy:
    absoluteTol: 1.8626e-05 +- 4.9643e-05 [0.0000e+00 - 1.6594e-04] (16#)
    relativeTol: 7.4505e-05 +- 8.4582e-06 [6.6046e-05 - 8.2963e-05] (2#)
    
```

### Performance
Code from [LayerTestBase.java:144](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.08 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 3.1337 +- 0.4912 [2.5050 - 5.5343]
    Learning performance: 1.4394 +- 0.3637 [1.1428 - 4.4457]
    
```

