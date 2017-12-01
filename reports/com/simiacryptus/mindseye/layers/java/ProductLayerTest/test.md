# ProductLayer
## ProductLayerTest
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
      "class": "com.simiacryptus.mindseye.layers.java.ProductLayer",
      "id": "f4569375-56fe-4e46-925c-95f400000a64",
      "isFrozen": false,
      "name": "ProductLayer/f4569375-56fe-4e46-925c-95f400000a64"
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
    [[ -0.272, 0.464, -0.584 ]]
    --------------------
    Output: 
    [ 0.07370547200000001 ]
```



### Differential Validation
Code from [LayerTestBase.java:139](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L139) executed in 0.00 seconds: 
```java
    getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Feedback for input 0
    Inputs: [ -0.272, 0.464, -0.584 ]
    Output: [ 0.07370547200000001 ]
    Measured: [ [ -0.2709760000001171 ], [ 0.15884800000004473 ], [ -0.12620800000001653 ] ]
    Implemented: [ [ -0.270976 ], [ 0.15884800000000002 ], [ -0.12620800000000001 ] ]
    Error: [ [ -1.1712852909795402E-13 ], [ 4.471423231677818E-14 ], [ -1.6514567491299204E-14 ] ]
    Finite-Difference Derivative Accuracy:
    absoluteTol: 5.9452e-14 +- 4.2377e-14 [1.6515e-14 - 1.1713e-13] (3#)
    relativeTol: 1.4076e-13 +- 6.1522e-14 [6.5426e-14 - 2.1612e-13] (3#)
    
```

### Performance
Code from [LayerTestBase.java:144](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.00 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 0.1467 +- 0.0248 [0.1054 - 0.2479]
    Learning performance: 0.0035 +- 0.0037 [0.0000 - 0.0313]
    
```

