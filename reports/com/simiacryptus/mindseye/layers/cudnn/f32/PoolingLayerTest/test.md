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
      "id": "c88cbdf1-1c2a-4a5e-b964-8909000002f2",
      "isFrozen": false,
      "name": "PoolingLayer/c88cbdf1-1c2a-4a5e-b964-8909000002f2",
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
    	[ [ -1.872, 1.604 ], [ 1.708, 1.596 ] ],
    	[ [ -0.968, -0.456 ], [ -1.524, 0.464 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 1.7079999446868896, 1.6039999723434448 ] ]
    ]
```



### Differential Validation
Code from [LayerTestBase.java:139](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L139) executed in 0.01 seconds: 
```java
    getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Component: PoolingLayer/c88cbdf1-1c2a-4a5e-b964-8909000002f2
    Inputs: [
    	[ [ -1.872, 1.604 ], [ 1.708, 1.596 ] ],
    	[ [ -0.968, -0.456 ], [ -1.524, 0.464 ] ]
    ]
    output=[
    	[ [ 1.7079999446868896, 1.6039999723434448 ] ]
    ]
    measured/actual: [ [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 1.0001659393310547, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 1.0001659393310547 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ] ]
    implemented/expected: [ [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 1.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 1.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ] ]
    error: [ [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 1.659393310546875E-4, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 1.659393310546875E-4 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ] ]
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
    Evaluation performance: 2.0673 +- 0.5226 [1.6301 - 5.7024]
    Learning performance: 1.2161 +- 0.1703 [0.9547 - 2.4280]
    
```

