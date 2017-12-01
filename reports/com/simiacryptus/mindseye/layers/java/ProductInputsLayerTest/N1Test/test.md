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
      "id": "f4569375-56fe-4e46-925c-95f400000a56",
      "isFrozen": false,
      "name": "ProductInputsLayer/f4569375-56fe-4e46-925c-95f400000a56"
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
    [[ -0.84, 0.112, -1.956 ],
    [ -0.44 ]]
    --------------------
    Output: 
    [ 0.3696, -0.049280000000000004, 0.86064 ]
```



### Differential Validation
Code from [LayerTestBase.java:139](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L139) executed in 0.00 seconds: 
```java
    getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Feedback for input 0
    Inputs: [ -0.84, 0.112, -1.956 ],
    [ -0.44 ]
    Output: [ 0.3696, -0.049280000000000004, 0.86064 ]
    Measured: [ [ -0.4399999999998849, 0.0, 0.0 ], [ 0.0, -0.4399999999999543, 0.0 ], [ 0.0, 0.0, -0.4399999999993298 ] ]
    Implemented: [ [ -0.44, 0.0, 0.0 ], [ 0.0, -0.44, 0.0 ], [ 0.0, 0.0, -0.44 ] ]
    Error: [ [ 1.1507461650239748E-13, 0.0, 0.0 ], [ 0.0, 4.568567746332519E-14, 0.0 ], [ 0.0, 0.0, 6.701861288149757E-13 ] ]
    Feedback for input 1
    Inputs: [ -0.84, 0.112, -1.956 ],
    [ -0.44 ]
    Output: [ 0.3696, -0.049280000000000004, 0.86064 ]
    Measured: [ [ -0.8399999999997299, 0.11200000000002874, -1.9559999999996247 ] ]
    Implemented: [ [ -0.84, 0.11200000000000002, -1.956 ] ]
    Error: [ [ 2.701172618913006E-13, 2.8727020762175925E-14, 3.752553823233029E-13 ] ]
    Finite-Difference Derivative Accuracy:
    absoluteTol: 1.2542e-13 +- 2.0214e-13 [0.0000e+00 - 6.7019e-13] (12#)
    relativeTol: 2.2154e-13 +- 2.4387e-13 [5.1916e-14 - 7.6158e-13] (6#)
    
```

### Performance
Code from [LayerTestBase.java:144](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.01 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 0.3495 +- 0.2123 [0.2166 - 2.1744]
    Learning performance: 0.0311 +- 0.0168 [0.0199 - 0.1368]
    
```

