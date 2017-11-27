# MaxImageBandLayer
## MaxImageBandLayerTest
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
      "class": "com.simiacryptus.mindseye.layers.java.MaxImageBandLayer",
      "id": "0910987d-3688-428c-a892-e2c400000e56",
      "isFrozen": false,
      "name": "MaxImageBandLayer/0910987d-3688-428c-a892-e2c400000e56"
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
    	[ [ -0.956, -0.556, -0.112 ], [ -0.492, 1.756, 1.024 ] ],
    	[ [ -1.752, 0.504, 1.4 ], [ 1.864, 1.64, 0.124 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 1.864, 1.64, 0.124 ] ]
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
    absoluteTol: 3.7472e-13 +- 1.4993e-12 [0.0000e+00 - 6.5510e-12] (36#)
    relativeTol: 2.2483e-12 +- 1.4527e-12 [1.9390e-13 - 3.2755e-12] (3#)
    
```

### Performance
Code from [LayerTestBase.java:144](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.00 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 0.1356 +- 0.0381 [0.1083 - 0.3249]
    Learning performance: 0.0018 +- 0.0021 [0.0000 - 0.0171]
    
```

