# ImgCropLayer
## ImgCropLayerTest
### Json Serialization
Code from [LayerTestBase.java:84](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L84) executed in 0.08 seconds: 
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
      "class": "com.simiacryptus.mindseye.layers.java.ImgCropLayer",
      "id": "9e053691-b5ee-4f65-88c1-f87d00000001",
      "isFrozen": false,
      "name": "ImgCropLayer/9e053691-b5ee-4f65-88c1-f87d00000001",
      "sizeX": 2,
      "sizeY": 2
    }
```



### Example Input/Output Pair
Code from [LayerTestBase.java:121](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L121) executed in 0.02 seconds: 
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
    	[ [ -0.728, 0.392, -1.372 ], [ 1.616, -0.832, 1.104 ], [ -1.924, 1.188, 0.616 ], [ 1.812, 1.344, -1.216 ] ],
    	[ [ 0.132, 1.584, 1.604 ], [ 0.14, -1.192, 0.192 ], [ 1.836, -0.14, -1.972 ], [ 0.928, -1.868, 0.032 ] ],
    	[ [ -0.652, -1.864, 1.212 ], [ -0.792, -1.7, 1.32 ], [ 1.66, -1.416, -0.728 ], [ 0.492, -0.032, -1.664 ] ],
    	[ [ 1.496, -0.8, -0.812 ], [ -1.584, -0.54, 0.732 ], [ 1.012, 0.264, 0.592 ], [ -0.708, 0.124, -0.792 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 0.14, -1.192, 0.192 ], [ 1.836, -0.14, -1.972 ] ],
    	[ [ -0.792, -1.7, 1.32 ], [ 1.66, -1.416, -0.728 ] ]
    ]
```



### Differential Validation
Code from [LayerTestBase.java:139](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L139) executed in 0.07 seconds: 
```java
    getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Component: ImgCropLayer/9e053691-b5ee-4f65-88c1-f87d00000001
    Inputs: [
    	[ [ -0.728, 0.392, -1.372 ], [ 1.616, -0.832, 1.104 ], [ -1.924, 1.188, 0.616 ], [ 1.812, 1.344, -1.216 ] ],
    	[ [ 0.132, 1.584, 1.604 ], [ 0.14, -1.192, 0.192 ], [ 1.836, -0.14, -1.972 ], [ 0.928, -1.868, 0.032 ] ],
    	[ [ -0.652, -1.864, 1.212 ], [ -0.792, -1.7, 1.32 ], [ 1.66, -1.416, -0.728 ], [ 0.492, -0.032, -1.664 ] ],
    	[ [ 1.496, -0.8, -0.812 ], [ -1.584, -0.54, 0.732 ], [ 1.012, 0.264, 0.592 ], [ -0.708, 0.124, -0.792 ] ]
    ]
    output=[
    	[ [ 0.14, -1.192, 0.192 ], [ 1.836, -0.14, -1.972 ] ],
    	[ [ -0.792, -1.7, 1.32 ], [ 1.66, -1.416, -0.728 ] ]
    ]
    measured/actual: [ [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.9999999999998899, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.9999999999998899, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.9999999999998899, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ] ]
    implemented/expected: [ [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ] ]
    error: [ [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ -1.1013412404281553E-13, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, -1.1013412404281553E-13, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, -1.1013412404281553E-13, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ] ]
    Finite-Difference Derivative Accuracy:
    absoluteTol: 2.2945e-15 +- 1.5730e-14 [0.0000e+00 - 1.1013e-13] (576#)
    relativeTol: 5.5067e-14 +- 0.0000e+00 [5.5067e-14 - 5.5067e-14] (12#)
    
```

### Performance
Code from [LayerTestBase.java:144](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.03 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 0.9405 +- 0.9469 [0.6127 - 10.0968]
    Learning performance: 0.0583 +- 0.0347 [0.0342 - 0.2878]
    
```

