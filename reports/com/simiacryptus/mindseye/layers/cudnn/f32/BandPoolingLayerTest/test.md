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
      "id": "f4569375-56fe-4e46-925c-95f400000033",
      "isFrozen": false,
      "name": "BandPoolingLayer/f4569375-56fe-4e46-925c-95f400000033",
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
    	[ [ -0.484, 1.092 ], [ 0.38, 0.408 ], [ 1.644, -0.112 ] ],
    	[ [ 0.28, -1.272 ], [ 1.208, -1.776 ], [ -0.068, -1.764 ] ],
    	[ [ 1.588, 1.628 ], [ -0.532, -1.816 ], [ -1.088, 0.568 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 1.6440000534057617, 1.628000020980835 ] ]
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
    	[ [ -0.484, 1.092 ], [ 0.38, 0.408 ], [ 1.644, -0.112 ] ],
    	[ [ 0.28, -1.272 ], [ 1.208, -1.776 ], [ -0.068, -1.764 ] ],
    	[ [ 1.588, 1.628 ], [ -0.532, -1.816 ], [ -1.088, 0.568 ] ]
    ]
    Output: [
    	[ [ 1.6440000534057617, 1.628000020980835 ] ]
    ]
    Measured: [ [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.9989738464355469, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ] ]
    Implemented: [ [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 1.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ] ]
    Error: [ [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ -0.001026153564453125, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ] ]
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
    Evaluation performance: 2.2109 +- 0.6134 [1.7469 - 5.8506]
    Learning performance: 1.1701 +- 0.1648 [0.9547 - 1.9293]
    
```

