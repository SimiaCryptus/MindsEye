# ProductInputsLayer
## N1Test
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
      "class": "com.simiacryptus.mindseye.layers.java.ProductInputsLayer",
      "id": "c88cbdf1-1c2a-4a5e-b964-890900000f93",
      "isFrozen": false,
      "name": "ProductInputsLayer/c88cbdf1-1c2a-4a5e-b964-890900000f93"
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
    [[ -1.812, 1.88, -0.768 ],
    [ 1.884 ]]
    --------------------
    Output: 
    [ -3.413808, 3.5419199999999997, -1.446912 ]
```



### Differential Validation
Code from [LayerTestBase.java:139](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L139) executed in 0.00 seconds: 
```java
    getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Component: ProductInputsLayer/c88cbdf1-1c2a-4a5e-b964-890900000f93
    Inputs: [ -1.812, 1.88, -0.768 ],
    [ 1.884 ]
    output=[ -3.413808, 3.5419199999999997, -1.446912 ]
    measured/actual: [ [ 1.8839999999986645, 0.0, 0.0 ], [ 0.0, 1.8839999999986645, 0.0 ], [ 0.0, 0.0, 1.884000000000885 ] ]
    implemented/expected: [ [ 1.884, 0.0, 0.0 ], [ 0.0, 1.884, 0.0 ], [ 0.0, 0.0, 1.884 ] ]
    error: [ [ -1.3353762540191383E-12, 0.0, 0.0 ], [ 0.0, -1.3353762540191383E-12, 0.0 ], [ 0.0, 0.0, 8.850697952311748E-13 ] ]
    Component: ProductInputsLayer/c88cbdf1-1c2a-4a5e-b964-890900000f93
    Inputs: [ -1.812, 1.88, -0.768 ],
    [ 1.884 ]
    output=[ -3.413808, 3.5419199999999997, -1.446912 ]
    measured/actual: [ [ -1.812000000001035, 1.879999999996329, -0.7679999999998799 ] ]
    implemented/expected: [ [ -1.812, 1.88, -0.768 ] ]
    error: [ [ -1.034949903555571E-12, -3.6708414086206176E-12, 1.2012613126444194E-13 ] ]
    Finite-Difference Derivative Accuracy:
    absoluteTol: 6.9848e-13 +- 1.0431e-12 [0.0000e+00 - 3.6708e-12] (12#)
    relativeTol: 3.8063e-13 +- 2.8223e-13 [7.8207e-14 - 9.7629e-13] (6#)
    
```

### Performance
Code from [LayerTestBase.java:144](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.01 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 0.2693 +- 0.0898 [0.1738 - 0.7210]
    Learning performance: 0.0203 +- 0.0113 [0.0114 - 0.0912]
    
```

