# FullyConnectedLayer
## FullyConnectedLayerTest
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
      "class": "com.simiacryptus.mindseye.layers.java.FullyConnectedLayer",
      "id": "0910987d-3688-428c-a892-e2c400000e1b",
      "isFrozen": false,
      "name": "FullyConnectedLayer/0910987d-3688-428c-a892-e2c400000e1b",
      "outputDims": [
        3
      ],
      "inputDims": [
        3
      ],
      "weights": {
        "dimensions": [
          3,
          3
        ],
        "data": [
          -0.3540307359835975,
          -0.24930522184857762,
          -0.04432082401516988,
          0.25178163048484614,
          0.4011535462478614,
          -0.6296615779671495,
          0.3917854303892088,
          -0.795530401388429,
          -0.5829898113703589
        ]
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
    [[ 1.248, -0.424, 1.42 ]]
    --------------------
    Output: 
    [ 0.007749541319572062, -1.6108751904476872, -0.6161814114587703 ]
```



### Differential Validation
Code from [LayerTestBase.java:139](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L139) executed in 0.00 seconds: 
```java
    getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Finite-Difference Derivative Accuracy:
    absoluteTol: 2.0110e-12 +- 3.4774e-12 [0.0000e+00 - 1.4678e-11] (36#)
    relativeTol: 5.3713e-12 +- 1.0847e-11 [6.1440e-14 - 4.8778e-11] (18#)
    
```

### Performance
Code from [LayerTestBase.java:144](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.01 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 0.0795 +- 0.0253 [0.0684 - 0.2821]
    Learning performance: 0.6517 +- 0.3727 [0.3505 - 2.2143]
    
```

