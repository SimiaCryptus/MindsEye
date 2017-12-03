# ImgConcatLayer
## ImgConcatLayerTest
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
      "class": "com.simiacryptus.mindseye.layers.cudnn.f32.ImgConcatLayer",
      "id": "e2d0bffa-47dc-4875-864f-3d3d00000340",
      "isFrozen": false,
      "name": "ImgConcatLayer/e2d0bffa-47dc-4875-864f-3d3d00000340",
      "maxBands": -1
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
    	[ [ 0.584 ], [ 1.532 ] ],
    	[ [ 1.032 ], [ -1.512 ] ]
    ],
    [
    	[ [ 0.144 ], [ 0.428 ] ],
    	[ [ 1.532 ], [ -0.824 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 0.5839999914169312, 0.14399999380111694 ], [ 1.531999945640564, 0.42800000309944153 ] ],
    	[ [ 1.031999945640564, 1.531999945640564 ], [ -1.5119999647140503, -0.8240000009536743 ] ]
    ]
```



### Differential Validation
Code from [LayerTestBase.java:139](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L139) executed in 0.03 seconds: 
```java
    getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Feedback for input 0
    Inputs: [
    	[ [ 0.584 ], [ 1.532 ] ],
    	[ [ 1.032 ], [ -1.512 ] ]
    ],
    [
    	[ [ 0.144 ], [ 0.428 ] ],
    	[ [ 1.532 ], [ -0.824 ] ]
    ]
    Output: [
    	[ [ 0.5839999914169312, 0.14399999380111694 ], [ 1.531999945640564, 0.42800000309944153 ] ],
    	[ [ 1.031999945640564, 1.531999945640564 ], [ -1.5119999647140503, -0.8240000009536743 ] ]
    ]
    Measured: [ [ 1.0001659393310547, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 1.0001659393310547, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 1.0001659393310547, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 1.0001659393310547, 0.0, 0.0, 0.0, 0.0 ] ]
    Implemented: [ [ 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0 ] ]
    Error: [ [ 1.659393310546875E-4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 1.659393310546875E-4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 1.659393310546875E-4, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 1.659393310546875E-4, 0.0, 0.0, 0.0, 0.0 ] ]
    Feedback for input 1
    Inputs: [
    	[ [ 0.584 ], [ 1.532 ] ],
    	[ [ 1.032 ], [ -1.512 ] ]
    ],
    [
    	[ [ 0.144 ], [ 0.428 ] ],
    	[ [ 1.532 ], [ -0.824 ] ]
    ]
    Output: [
    	[ [ 0.5839999914169312, 0.14399999380111694 ], [ 1.531999945640564, 0.42800000309944153 ] ],
    	[ [ 1.031999945640564, 1.531999945640564 ], [ -1.5119999647140503, -0.8240000009536743 ] ]
    ]
    Measured: [ [ 0.0, 0.0, 0.0, 0.0, 1.0000169277191162, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 1.0001659393310547, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9998679161071777, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0001659393310547 ] ]
    Implemented: [ [ 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0 ] ]
    Error: [ [ 0.0, 0.0, 0.0, 0.0, 1.6927719116210938E-5, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 1.659393310546875E-4, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.3208389282226562E-4, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.659393310546875E-4 ] ]
    Finite-Difference Derivative Accuracy:
    absoluteTol: 1.7885e-05 +- 5.0385e-05 [0.0000e+00 - 1.6594e-04] (64#)
    relativeTol: 7.1536e-05 +- 2.4474e-05 [8.4638e-06 - 8.2963e-05] (8#)
    
```

### Performance
Code from [LayerTestBase.java:144](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.09 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 4.1333 +- 1.2256 [3.1604 - 11.2823]
    Learning performance: 1.0820 +- 0.1080 [0.9404 - 1.7954]
    
```

