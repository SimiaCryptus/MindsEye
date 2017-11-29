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
      "id": "c88cbdf1-1c2a-4a5e-b964-890900000f95",
      "isFrozen": false,
      "name": "ProductLayer/c88cbdf1-1c2a-4a5e-b964-890900000f95"
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
    [[ 1.9, -1.056, -0.024 ]]
    --------------------
    Output: 
    [ 0.048153600000000005 ]
```



### Differential Validation
Code from [LayerTestBase.java:139](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L139) executed in 0.00 seconds: 
```java
    getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Component: ProductLayer/c88cbdf1-1c2a-4a5e-b964-890900000f95
    Inputs: [ 1.9, -1.056, -0.024 ]
    output=[ 0.048153600000000005 ]
    measured/actual: [ [ 0.025343999999968836 ], [ -0.045600000000076135 ], [ -2.00639999999995 ] ]
    implemented/expected: [ [ 0.025344000000000002 ], [ -0.0456 ], [ -2.0064 ] ]
    error: [ [ -3.1166041969399316E-14 ], [ -7.613354391367011E-14 ], [ 5.0182080713057076E-14 ] ]
    Finite-Difference Derivative Accuracy:
    absoluteTol: 5.2494e-14 +- 1.8431e-14 [3.1166e-14 - 7.6134e-14] (3#)
    relativeTol: 4.8739e-13 +- 3.4759e-13 [1.2506e-14 - 8.3480e-13] (3#)
    
```

### Performance
Code from [LayerTestBase.java:144](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.00 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 0.1537 +- 0.0396 [0.1054 - 0.3448]
    Learning performance: 0.0018 +- 0.0017 [0.0000 - 0.0086]
    
```

