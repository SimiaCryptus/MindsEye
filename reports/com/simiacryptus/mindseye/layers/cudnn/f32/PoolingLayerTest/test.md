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
      "id": "f4569375-56fe-4e46-925c-95f400000121",
      "isFrozen": false,
      "name": "PoolingLayer/f4569375-56fe-4e46-925c-95f400000121",
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
    	[ [ -0.488, -1.708 ], [ 1.652, -1.184 ] ],
    	[ [ -0.18, 0.848 ], [ 0.676, 1.516 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 1.6519999504089355, 1.5160000324249268 ] ]
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
    	[ [ -0.488, -1.708 ], [ 1.652, -1.184 ] ],
    	[ [ -0.18, 0.848 ], [ 0.676, 1.516 ] ]
    ]
    Output: [
    	[ [ 1.6519999504089355, 1.5160000324249268 ] ]
    ]
    Measured: [ [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 1.0001659393310547, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 1.0001659393310547 ] ]
    Implemented: [ [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 1.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 1.0 ] ]
    Error: [ [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 1.659393310546875E-4, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 1.659393310546875E-4 ] ]
    Finite-Difference Derivative Accuracy:
    absoluteTol: 2.0742e-05 +- 5.4879e-05 [0.0000e+00 - 1.6594e-04] (16#)
    relativeTol: 8.2963e-05 +- 0.0000e+00 [8.2963e-05 - 8.2963e-05] (2#)
    
```

### Performance
Code from [LayerTestBase.java:144](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.05 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 1.8857 +- 0.3041 [1.6842 - 4.0211]
    Learning performance: 1.2372 +- 0.1826 [0.9034 - 2.1630]
    
```

