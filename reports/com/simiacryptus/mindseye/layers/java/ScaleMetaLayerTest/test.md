# ScaleMetaLayer
## ScaleMetaLayerTest
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
      "class": "com.simiacryptus.mindseye.layers.java.ScaleMetaLayer",
      "id": "e2d0bffa-47dc-4875-864f-3d3d00001554",
      "isFrozen": false,
      "name": "ScaleMetaLayer/e2d0bffa-47dc-4875-864f-3d3d00001554"
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
    [[ -0.58, 0.88, 1.62 ],
    [ -1.5, -0.504, -0.256 ]]
    --------------------
    Output: 
    [ 0.8699999999999999, -0.44352, -0.41472000000000003 ]
```



### Differential Validation
Code from [LayerTestBase.java:139](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L139) executed in 0.00 seconds: 
```java
    getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Feedback for input 0
    Inputs: [ -0.58, 0.88, 1.62 ],
    [ -1.5, -0.504, -0.256 ]
    Output: [ 0.8699999999999999, -0.44352, -0.41472000000000003 ]
    Measured: [ [ -1.4999999999987246, 0.0, 0.0 ], [ 0.0, -0.5039999999995048, 0.0 ], [ 0.0, 0.0, -0.256000000000145 ] ]
    Implemented: [ [ -1.5, 0.0, 0.0 ], [ 0.0, -0.504, 0.0 ], [ 0.0, 0.0, -0.256 ] ]
    Error: [ [ 1.2754242106893798E-12, 0.0, 0.0 ], [ 0.0, 4.951594689828198E-13, 0.0 ], [ 0.0, 0.0, -1.4499512701604544E-13 ] ]
    Feedback for input 1
    Inputs: [ -0.58, 0.88, 1.62 ],
    [ -1.5, -0.504, -0.256 ]
    Output: [ 0.8699999999999999, -0.44352, -0.41472000000000003 ]
    Measured: [ [ -0.5799999999989147, 0.0, 0.0 ], [ 0.0, 0.880000000000325, 0.0 ], [ 0.0, 0.0, 1.6199999999999548 ] ]
    Implemented: [ [ -0.58, 0.0, 0.0 ], [ 0.0, 0.88, 0.0 ], [ 0.0, 0.0, 1.62 ] ]
    Error: [ [ 1.0852430065710905E-12, 0.0, 0.0 ], [ 0.0, 3.249622793077833E-13, 0.0 ], [ 0.0, 0.0, -4.529709940470639E-14 ] ]
    Finite-Difference Derivative Accuracy:
    absoluteTol: 1.8728e-13 +- 3.7616e-13 [0.0000e+00 - 1.2754e-12] (18#)
    relativeTol: 3.8896e-13 +- 2.8985e-13 [1.3981e-14 - 9.3555e-13] (6#)
    
```

### Performance
Code from [LayerTestBase.java:144](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.00 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 0.1595 +- 0.0853 [0.1111 - 0.8863]
    Learning performance: 0.0021 +- 0.0015 [0.0000 - 0.0086]
    
```

