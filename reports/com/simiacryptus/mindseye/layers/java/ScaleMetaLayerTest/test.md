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
      "id": "f4569375-56fe-4e46-925c-95f400000a6b",
      "isFrozen": false,
      "name": "ScaleMetaLayer/f4569375-56fe-4e46-925c-95f400000a6b"
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
    [[ -1.528, -1.692, -1.132 ],
    [ -1.336, -0.344, 0.484 ]]
    --------------------
    Output: 
    [ 2.041408, 0.5820479999999999, -0.5478879999999999 ]
```



### Differential Validation
Code from [LayerTestBase.java:139](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L139) executed in 0.00 seconds: 
```java
    getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Feedback for input 0
    Inputs: [ -1.528, -1.692, -1.132 ],
    [ -1.336, -0.344, 0.484 ]
    Output: [ 2.041408, 0.5820479999999999, -0.5478879999999999 ]
    Measured: [ [ -1.3359999999984495, 0.0, 0.0 ], [ 0.0, -0.34399999999989994, 0.0 ], [ 0.0, 0.0, 0.48399999999948484 ] ]
    Implemented: [ [ -1.336, 0.0, 0.0 ], [ 0.0, -0.344, 0.0 ], [ 0.0, 0.0, 0.484 ] ]
    Error: [ [ 1.5505374761914936E-12, 0.0, 0.0 ], [ 0.0, 1.000310945187266E-13, 0.0 ], [ 0.0, 0.0, -5.151434834260726E-13 ] ]
    Feedback for input 1
    Inputs: [ -1.528, -1.692, -1.132 ],
    [ -1.336, -0.344, 0.484 ]
    Output: [ 2.041408, 0.5820479999999999, -0.5478879999999999 ]
    Measured: [ [ -1.5279999999995297, 0.0, 0.0 ], [ 0.0, -1.6919999999998048, 0.0 ], [ 0.0, 0.0, -1.132000000000355 ] ]
    Implemented: [ [ -1.528, 0.0, 0.0 ], [ 0.0, -1.692, 0.0 ], [ 0.0, 0.0, -1.132 ] ]
    Error: [ [ 4.702904732312163E-13, 0.0, 0.0 ], [ 0.0, 1.9517720772910252E-13, 0.0 ], [ 0.0, 0.0, -3.5504932327512506E-13 ] ]
    Finite-Difference Derivative Accuracy:
    absoluteTol: 1.7701e-13 +- 3.7274e-13 [0.0000e+00 - 1.5505e-12] (18#)
    relativeTol: 2.7104e-13 +- 2.0490e-13 [5.7676e-14 - 5.8029e-13] (6#)
    
```

### Performance
Code from [LayerTestBase.java:144](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.00 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 0.1732 +- 0.0393 [0.1339 - 0.3106]
    Learning performance: 0.0036 +- 0.0019 [0.0000 - 0.0142]
    
```

