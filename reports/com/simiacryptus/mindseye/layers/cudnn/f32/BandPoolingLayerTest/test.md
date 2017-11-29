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
      "id": "c88cbdf1-1c2a-4a5e-b964-890900000006",
      "isFrozen": false,
      "name": "BandPoolingLayer/c88cbdf1-1c2a-4a5e-b964-890900000006",
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
    	[ [ -0.28, 1.716 ], [ -0.744, 1.268 ], [ -0.796, -1.596 ] ],
    	[ [ 1.592, -0.628 ], [ 0.212, 0.18 ], [ -1.504, -1.144 ] ],
    	[ [ 1.972, 1.436 ], [ 0.288, -1.188 ], [ 0.604, -0.044 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 1.972000002861023, 1.715999960899353 ] ]
    ]
```



### Differential Validation
Code from [LayerTestBase.java:139](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L139) executed in 0.01 seconds: 
```java
    getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Component: BandPoolingLayer/c88cbdf1-1c2a-4a5e-b964-890900000006
    Inputs: [
    	[ [ -0.28, 1.716 ], [ -0.744, 1.268 ], [ -0.796, -1.596 ] ],
    	[ [ 1.592, -0.628 ], [ 0.212, 0.18 ], [ -1.504, -1.144 ] ],
    	[ [ 1.972, 1.436 ], [ 0.288, -1.188 ], [ 0.604, -0.044 ] ]
    ]
    output=[
    	[ [ 1.972000002861023, 1.715999960899353 ] ]
    ]
    measured/actual: [ [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 1.0001659393310547, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 1.0001659393310547 ] ]
    implemented/expected: [ [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 1.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 1.0 ] ]
    error: [ [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 1.659393310546875E-4, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 1.659393310546875E-4 ] ]
    Finite-Difference Derivative Accuracy:
    absoluteTol: 9.2189e-06 +- 3.8010e-05 [0.0000e+00 - 1.6594e-04] (36#)
    relativeTol: 8.2963e-05 +- 0.0000e+00 [8.2963e-05 - 8.2963e-05] (2#)
    
```

### Performance
Code from [LayerTestBase.java:144](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.05 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 2.1329 +- 0.2949 [1.7697 - 3.3343]
    Learning performance: 1.2173 +- 0.2167 [0.9974 - 2.8013]
    
```

