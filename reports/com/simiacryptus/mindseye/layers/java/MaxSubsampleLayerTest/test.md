# MaxSubsampleLayer
## MaxSubsampleLayerTest
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
      "class": "com.simiacryptus.mindseye.layers.java.MaxSubsampleLayer",
      "id": "0910987d-3688-428c-a892-e2c400000e5e",
      "isFrozen": false,
      "name": "MaxSubsampleLayer/0910987d-3688-428c-a892-e2c400000e5e",
      "inner": [
        2,
        2,
        1
      ]
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
    [[
    	[ [ 0.196, -0.352, -1.312 ], [ -0.096, 0.372, -1.756 ] ],
    	[ [ 1.736, -0.492, 1.552 ], [ -0.844, 0.872, 0.368 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 1.736, 0.872, 1.552 ] ]
    ]
```



### Differential Validation
Code from [LayerTestBase.java:139](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L139) executed in 0.00 seconds: 
```java
    getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Finite-Difference Derivative Accuracy:
    absoluteTol: 4.9036e-13 +- 1.6490e-12 [0.0000e+00 - 6.5510e-12] (36#)
    relativeTol: 2.9422e-12 +- 4.7137e-13 [2.2756e-12 - 3.2755e-12] (3#)
    
```

### Performance
Code from [LayerTestBase.java:144](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.01 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 0.3081 +- 0.0826 [0.2023 - 0.7865]
    Learning performance: 0.0043 +- 0.0026 [0.0028 - 0.0228]
    
```

