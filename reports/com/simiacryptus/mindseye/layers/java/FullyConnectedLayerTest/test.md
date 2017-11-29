# FullyConnectedLayer
## FullyConnectedLayerTest
### Json Serialization
Code from [LayerTestBase.java:84](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L84) executed in 0.05 seconds: 
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
      "id": "8fc89c80-468c-4831-85de-f4d000000001",
      "isFrozen": false,
      "name": "FullyConnectedLayer/8fc89c80-468c-4831-85de-f4d000000001",
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
          -0.017553875264661307,
          -0.8319368982881744,
          -0.01034649768201477,
          -0.07177898370679524,
          -0.08155035113921981,
          0.22982435534566523,
          -0.744321197066051,
          0.39466296840640697,
          -0.13091579356429114
        ]
      }
    }
```



### Example Input/Output Pair
Code from [LayerTestBase.java:121](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L121) executed in 0.02 seconds: 
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
    [[ 0.94, 0.152, -0.136 ]]
    --------------------
    Output: 
    [ 0.07381663452876844, -0.8480905014673167, 0.043012142116190824 ]
```



### Differential Validation
Code from [LayerTestBase.java:139](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L139) executed in 0.08 seconds: 
```java
    getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Finite-Difference Derivative Accuracy:
    absoluteTol: 7.4937e-13 +- 1.5944e-12 [0.0000e+00 - 6.2764e-12] (36#)
    relativeTol: 4.4817e-12 +- 6.3210e-12 [6.8056e-14 - 2.3461e-11] (18#)
    
```

### Performance
Code from [LayerTestBase.java:144](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.02 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 0.5228 +- 0.7034 [0.3762 - 7.4921]
    Learning performance: 0.8069 +- 0.1695 [0.6099 - 2.0861]
    
```

