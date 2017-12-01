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
      "id": "f4569375-56fe-4e46-925c-95f400000a5d",
      "isFrozen": false,
      "name": "ProductInputsLayer/f4569375-56fe-4e46-925c-95f400000a5d"
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
    [[ -1.188, -0.092, 1.372 ],
    [ -0.312, -0.02, 1.68 ]]
    --------------------
    Output: 
    [ 0.370656, 0.00184, 2.30496 ]
```



### Differential Validation
Code from [LayerTestBase.java:139](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L139) executed in 0.00 seconds: 
```java
    getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Feedback for input 0
    Inputs: [ -1.188, -0.092, 1.372 ],
    [ -0.312, -0.02, 1.68 ]
    Output: [ 0.370656, 0.00184, 2.30496 ]
    Measured: [ [ -0.31200000000009, 0.0, 0.0 ], [ 0.0, -0.020000000000000486, 0.0 ], [ 0.0, 0.0, 1.6800000000039006 ] ]
    Implemented: [ [ -0.312, 0.0, 0.0 ], [ 0.0, -0.02, 0.0 ], [ 0.0, 0.0, 1.6799999999999997 ] ]
    Error: [ [ -8.998357614586894E-14, 0.0, 0.0 ], [ 0.0, -4.85722573273506E-16, 0.0 ], [ 0.0, 0.0, 3.90087961932295E-12 ] ]
    Feedback for input 1
    Inputs: [ -1.188, -0.092, 1.372 ],
    [ -0.312, -0.02, 1.68 ]
    Output: [ 0.370656, 0.00184, 2.30496 ]
    Measured: [ [ -1.1879999999997448, 0.0, 0.0 ], [ 0.0, -0.09200000000000007, 0.0 ], [ 0.0, 0.0, 1.3720000000017052 ] ]
    Implemented: [ [ -1.188, 0.0, 0.0 ], [ 0.0, -0.092, 0.0 ], [ 0.0, 0.0, 1.3719999999999999 ] ]
    Error: [ [ 2.5512925105886097E-13, 0.0, 0.0 ], [ 0.0, -6.938893903907228E-17, 0.0 ], [ 0.0, 0.0, 1.7053025658242404E-12 ] ]
    Finite-Difference Derivative Accuracy:
    absoluteTol: 3.3066e-13 +- 9.4956e-13 [0.0000e+00 - 3.9009e-12] (18#)
    relativeTol: 3.4109e-13 +- 4.2200e-13 [3.7711e-16 - 1.1610e-12] (6#)
    
```

### Performance
Code from [LayerTestBase.java:144](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.00 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 0.2070 +- 0.0719 [0.1425 - 0.5928]
    Learning performance: 0.0238 +- 0.0145 [0.0142 - 0.1140]
    
```

