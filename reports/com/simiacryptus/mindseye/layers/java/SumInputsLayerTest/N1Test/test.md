# SumInputsLayer
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
      "class": "com.simiacryptus.mindseye.layers.java.SumInputsLayer",
      "id": "c88cbdf1-1c2a-4a5e-b964-890900000fa6",
      "isFrozen": false,
      "name": "SumInputsLayer/c88cbdf1-1c2a-4a5e-b964-890900000fa6"
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
    [[ -1.588, -0.144, -0.696 ],
    [ 1.336 ]]
    --------------------
    Output: 
    [ -0.252, 1.1920000000000002, 0.6400000000000001 ]
```



### Differential Validation
Code from [LayerTestBase.java:139](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L139) executed in 0.00 seconds: 
```java
    getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Component: SumInputsLayer/c88cbdf1-1c2a-4a5e-b964-890900000fa6
    Inputs: [ -1.588, -0.144, -0.696 ],
    [ 1.336 ]
    output=[ -0.252, 1.1920000000000002, 0.6400000000000001 ]
    measured/actual: [ [ 0.9999999999998899, 0.0, 0.0 ], [ 0.0, 0.9999999999998899, 0.0 ], [ 0.0, 0.0, 0.9999999999998899 ] ]
    implemented/expected: [ [ 1.0, 0.0, 0.0 ], [ 0.0, 1.0, 0.0 ], [ 0.0, 0.0, 1.0 ] ]
    error: [ [ -1.1013412404281553E-13, 0.0, 0.0 ], [ 0.0, -1.1013412404281553E-13, 0.0 ], [ 0.0, 0.0, -1.1013412404281553E-13 ] ]
    Component: SumInputsLayer/c88cbdf1-1c2a-4a5e-b964-890900000fa6
    Inputs: [ -1.588, -0.144, -0.696 ],
    [ 1.336 ]
    output=[ -0.252, 1.1920000000000002, 0.6400000000000001 ]
    measured/actual: [ [ 0.9999999999998899, 0.9999999999998899, 0.9999999999998899 ] ]
    implemented/expected: [ [ 1.0, 1.0, 1.0 ] ]
    error: [ [ -1.1013412404281553E-13, -1.1013412404281553E-13, -1.1013412404281553E-13 ] ]
    Finite-Difference Derivative Accuracy:
    absoluteTol: 5.5067e-14 +- 5.5067e-14 [0.0000e+00 - 1.1013e-13] (12#)
    relativeTol: 5.5067e-14 +- 0.0000e+00 [5.5067e-14 - 5.5067e-14] (6#)
    
```

### Performance
Code from [LayerTestBase.java:144](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.01 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 0.2370 +- 0.0574 [0.1624 - 0.5386]
    Learning performance: 0.0173 +- 0.0079 [0.0114 - 0.0655]
    
```

