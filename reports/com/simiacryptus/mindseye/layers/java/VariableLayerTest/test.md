# VariableLayer
## VariableLayerTest
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
      "class": "com.simiacryptus.mindseye.layers.java.VariableLayer",
      "id": "0910987d-3688-428c-a892-e2c400000ec3",
      "isFrozen": false,
      "name": "VariableLayer/0910987d-3688-428c-a892-e2c400000ec3",
      "inner": {
        "class": "com.simiacryptus.mindseye.layers.java.MonitoringSynapse",
        "id": "0910987d-3688-428c-a892-e2c400000ec2",
        "isFrozen": false,
        "name": "MonitoringSynapse/0910987d-3688-428c-a892-e2c400000ec2",
        "totalBatches": 0,
        "totalItems": 0
      }
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
    [[ 1.452, 1.372, -0.168 ]]
    --------------------
    Output: 
    [ 1.452, 1.372, -0.168 ]
```



### Differential Validation
Code from [LayerTestBase.java:139](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L139) executed in 0.00 seconds: 
```java
    getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Finite-Difference Derivative Accuracy:
    absoluteTol: 1.5669e-12 +- 2.6819e-12 [0.0000e+00 - 6.5510e-12] (9#)
    relativeTol: 2.3503e-12 +- 1.3084e-12 [4.9993e-13 - 3.2755e-12] (3#)
    
```

### Performance
Code from [LayerTestBase.java:144](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.00 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 0.0085 +- 0.0030 [0.0057 - 0.0314]
    Learning performance: 0.0113 +- 0.0094 [0.0085 - 0.1026]
    
```

