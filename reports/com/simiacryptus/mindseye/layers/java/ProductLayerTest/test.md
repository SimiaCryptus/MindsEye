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
      "id": "e2d0bffa-47dc-4875-864f-3d3d0000154d",
      "isFrozen": false,
      "name": "ProductLayer/e2d0bffa-47dc-4875-864f-3d3d0000154d"
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
    [[ 0.836, 0.22, 0.632 ]]
    --------------------
    Output: 
    [ 0.11623744 ]
```



### Differential Validation
Code from [LayerTestBase.java:139](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L139) executed in 0.00 seconds: 
```java
    getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Feedback for input 0
    Inputs: [ 0.836, 0.22, 0.632 ]
    Output: [ 0.11623744 ]
    Measured: [ [ 0.13903999999995142 ], [ 0.5283520000000375 ], [ 0.1839200000000596 ] ]
    Implemented: [ [ 0.13904 ], [ 0.5283519999999999 ], [ 0.18392 ] ]
    Error: [ [ -4.85722573273506E-14 ], [ 3.752553823233029E-14 ], [ 5.959122084675528E-14 ] ]
    Finite-Difference Derivative Accuracy:
    absoluteTol: 4.8563e-14 +- 9.0083e-15 [3.7526e-14 - 5.9591e-14] (3#)
    relativeTol: 1.2406e-13 +- 6.2827e-14 [3.5512e-14 - 1.7467e-13] (3#)
    
```

### Performance
Code from [LayerTestBase.java:144](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.00 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 0.1471 +- 0.0272 [0.1140 - 0.2479]
    Learning performance: 0.0023 +- 0.0013 [0.0000 - 0.0057]
    
```

