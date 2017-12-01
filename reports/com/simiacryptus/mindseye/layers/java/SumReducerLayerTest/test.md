# SumReducerLayer
## SumReducerLayerTest
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
      "class": "com.simiacryptus.mindseye.layers.java.SumReducerLayer",
      "id": "f4569375-56fe-4e46-925c-95f400000a9e",
      "isFrozen": false,
      "name": "SumReducerLayer/f4569375-56fe-4e46-925c-95f400000a9e"
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
    [[ -0.572, 1.872, 0.024 ]]
    --------------------
    Output: 
    [ 1.3240000000000003 ]
```



### Differential Validation
Code from [LayerTestBase.java:139](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L139) executed in 0.00 seconds: 
```java
    getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Feedback for input 0
    Inputs: [ -0.572, 1.872, 0.024 ]
    Output: [ 1.3240000000000003 ]
    Measured: [ [ 0.9999999999976694 ], [ 0.9999999999976694 ], [ 0.9999999999998899 ] ]
    Implemented: [ [ 1.0 ], [ 1.0 ], [ 1.0 ] ]
    Error: [ [ -2.3305801732931286E-12 ], [ -2.3305801732931286E-12 ], [ -1.1013412404281553E-13 ] ]
    Finite-Difference Derivative Accuracy:
    absoluteTol: 1.5904e-12 +- 1.0467e-12 [1.1013e-13 - 2.3306e-12] (3#)
    relativeTol: 7.9522e-13 +- 5.2336e-13 [5.5067e-14 - 1.1653e-12] (3#)
    
```

### Performance
Code from [LayerTestBase.java:144](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.00 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 0.1799 +- 0.0578 [0.1282 - 0.4759]
    Learning performance: 0.0027 +- 0.0039 [0.0000 - 0.0399]
    
```

