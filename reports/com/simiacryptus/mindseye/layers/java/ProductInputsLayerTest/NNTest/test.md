# ProductInputsLayer
## NNTest
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
      "id": "c88cbdf1-1c2a-4a5e-b964-890900000f94",
      "isFrozen": false,
      "name": "ProductInputsLayer/c88cbdf1-1c2a-4a5e-b964-890900000f94"
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
    [[ 1.748, -0.768, 0.752 ],
    [ 1.2, -1.704, 1.148 ]]
    --------------------
    Output: 
    [ 2.0976, 1.308672, 0.863296 ]
```



### Differential Validation
Code from [LayerTestBase.java:139](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L139) executed in 0.00 seconds: 
```java
    getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Component: ProductInputsLayer/c88cbdf1-1c2a-4a5e-b964-890900000f94
    Inputs: [ 1.748, -0.768, 0.752 ],
    [ 1.2, -1.704, 1.148 ]
    output=[ 2.0976, 1.308672, 0.863296 ]
    measured/actual: [ [ 1.1999999999989797, 0.0, 0.0 ], [ 0.0, -1.7040000000001498, 0.0 ], [ 0.0, 0.0, 1.1479999999997048 ] ]
    implemented/expected: [ [ 1.2, 0.0, 0.0 ], [ -0.0, -1.704, -0.0 ], [ 0.0, 0.0, 1.148 ] ]
    error: [ [ -1.0202949596305189E-12, 0.0, 0.0 ], [ 0.0, -1.4988010832439613E-13, 0.0 ], [ 0.0, 0.0, -2.950972799453666E-13 ] ]
    Component: ProductInputsLayer/c88cbdf1-1c2a-4a5e-b964-890900000f94
    Inputs: [ 1.748, -0.768, 0.752 ],
    [ 1.2, -1.704, 1.148 ]
    output=[ 2.0976, 1.308672, 0.863296 ]
    measured/actual: [ [ 1.7479999999991946, 0.0, 0.0 ], [ 0.0, -0.7679999999998799, 0.0 ], [ 0.0, 0.0, 0.7519999999994198 ] ]
    implemented/expected: [ [ 1.748, 0.0, 0.0 ], [ -0.0, -0.768, -0.0 ], [ 0.0, 0.0, 0.752 ] ]
    error: [ [ -8.053557820630886E-13, 0.0, 0.0 ], [ 0.0, 1.2012613126444194E-13, 0.0 ], [ 0.0, 0.0, -5.802025526691068E-13 ] ]
    Finite-Difference Derivative Accuracy:
    absoluteTol: 1.6505e-13 +- 3.0367e-13 [0.0000e+00 - 1.0203e-12] (18#)
    relativeTol: 2.1533e-13 +- 1.4662e-13 [4.3979e-14 - 4.2512e-13] (6#)
    
```

### Performance
Code from [LayerTestBase.java:144](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.00 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 0.2102 +- 0.1008 [0.1624 - 0.9974]
    Learning performance: 0.0188 +- 0.0166 [0.0114 - 0.1368]
    
```

