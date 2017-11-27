# SumInputsLayer
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
      "class": "com.simiacryptus.mindseye.layers.java.SumInputsLayer",
      "id": "0910987d-3688-428c-a892-e2c400000ebd",
      "isFrozen": false,
      "name": "SumInputsLayer/0910987d-3688-428c-a892-e2c400000ebd"
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
    [[ -1.764, 0.524, 0.272 ],
    [ 0.088, 0.664, 1.504 ]]
    --------------------
    Output: 
    [ -1.676, 1.1880000000000002, 1.776 ]
```



### Differential Validation
Code from [LayerTestBase.java:139](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L139) executed in 0.00 seconds: 
```java
    getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Finite-Difference Derivative Accuracy:
    absoluteTol: 3.7007e-12 +- 5.8562e-12 [0.0000e+00 - 1.5653e-11] (18#)
    relativeTol: 5.5511e-12 +- 2.2756e-12 [3.2755e-12 - 7.8267e-12] (6#)
    
```

### Performance
Code from [LayerTestBase.java:144](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.00 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 0.0834 +- 0.0684 [0.0513 - 0.6127]
    Learning performance: 0.0120 +- 0.0049 [0.0085 - 0.0427]
    
```

