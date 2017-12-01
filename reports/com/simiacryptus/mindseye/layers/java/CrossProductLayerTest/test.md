# CrossProductLayer
## CrossProductLayerTest
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
      "class": "com.simiacryptus.mindseye.layers.java.CrossProductLayer",
      "id": "f4569375-56fe-4e46-925c-95f4000009a1",
      "isFrozen": false,
      "name": "CrossProductLayer/f4569375-56fe-4e46-925c-95f4000009a1"
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
    [[ 0.988, 0.416, -0.324, -1.804 ]]
    --------------------
    Output: 
    [ 0.411008, -0.320112, -1.782352, -0.134784, -0.750464, 0.584496 ]
```



### Differential Validation
Code from [LayerTestBase.java:139](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L139) executed in 0.00 seconds: 
```java
    getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Feedback for input 0
    Inputs: [ 0.988, 0.416, -0.324, -1.804 ]
    Output: [ 0.411008, -0.320112, -1.782352, -0.134784, -0.750464, 0.584496 ]
    Measured: [ [ 0.4159999999997499, -0.32399999999987994, -1.804000000000805, 0.0, 0.0, 0.0 ], [ 0.9879999999995448, 0.0, 0.0, -0.3240000000001575, -1.8039999999996947, 0.0 ], [ 0.0, 0.9880000000000999, 0.0, 0.4159999999997499, 0.0, -1.8039999999996947 ], [ 0.0, 0.0, 0.9879999999995448, 0.0, 0.4159999999997499, -0.32399999999932483 ] ]
    Implemented: [ [ 0.416, -0.324, -1.804, 0.0, 0.0, 0.0 ], [ 0.988, 0.0, 0.0, -0.324, -1.804, 0.0 ], [ 0.0, 0.988, 0.0, 0.416, 0.0, -1.804 ], [ 0.0, 0.0, 0.988, 0.0, 0.416, -0.324 ] ]
    Error: [ [ -2.500777362968165E-13, 1.2007062011321068E-13, -8.049116928532385E-13, 0.0, 0.0, 0.0 ], [ -4.551914400963142E-13, 0.0, 0.0, -1.5748513604307846E-13, 3.0531133177191805E-13, 0.0 ], [ 0.0, 9.992007221626409E-14, 0.0, -2.500777362968165E-13, 0.0, 3.0531133177191805E-13 ], [ 0.0, 0.0, -4.551914400963142E-13, 0.0, -2.500777362968165E-13, 6.75182132425789E-13 ] ]
    Finite-Difference Derivative Accuracy:
    absoluteTol: 1.7203e-13 +- 2.2649e-13 [0.0000e+00 - 8.0491e-13] (24#)
    relativeTol: 2.7297e-13 +- 2.4643e-13 [5.0567e-14 - 1.0419e-12] (12#)
    
```

### Performance
Code from [LayerTestBase.java:144](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.00 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 0.2217 +- 0.1361 [0.1510 - 1.4335]
    Learning performance: 0.0038 +- 0.0020 [0.0000 - 0.0142]
    
```

