# ActivationLayer
## ActivationLayerReLuTest
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
      "class": "com.simiacryptus.mindseye.layers.cudnn.f32.ActivationLayer",
      "id": "e2d0bffa-47dc-4875-864f-3d3d0000001c",
      "isFrozen": false,
      "name": "ActivationLayer/e2d0bffa-47dc-4875-864f-3d3d0000001c",
      "mode": 1
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
    [[
    	[ [ 1.088, 0.132, 1.64 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 1.0880000591278076, 0.13199999928474426, 1.6399999856948853 ] ]
    ]
```



### Differential Validation
Code from [LayerTestBase.java:139](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L139) executed in 0.01 seconds: 
```java
    getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Feedback for input 0
    Inputs: [
    	[ [ 1.088, 0.132, 1.64 ] ]
    ]
    Output: [
    	[ [ 1.0880000591278076, 0.13199999928474426, 1.6399999856948853 ] ]
    ]
    Measured: [ [ 0.9989738464355469, 0.0, 0.0 ], [ 0.0, 1.0000169277191162, 0.0 ], [ 0.0, 0.0, 1.0001659393310547 ] ]
    Implemented: [ [ 1.0, 0.0, 0.0 ], [ 0.0, 1.0, 0.0 ], [ 0.0, 0.0, 1.0 ] ]
    Error: [ [ -0.001026153564453125, 0.0, 0.0 ], [ 0.0, 1.6927719116210938E-5, 0.0 ], [ 0.0, 0.0, 1.659393310546875E-4 ] ]
    Finite-Difference Derivative Accuracy:
    absoluteTol: 1.3434e-04 +- 3.1944e-04 [0.0000e+00 - 1.0262e-03] (9#)
    relativeTol: 2.0159e-04 +- 2.2253e-04 [8.4638e-06 - 5.1334e-04] (3#)
    
```

### Performance
Code from [LayerTestBase.java:144](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.08 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 2.8993 +- 0.3691 [2.4651 - 4.3060]
    Learning performance: 2.1072 +- 0.2132 [1.7669 - 2.7985]
    
```

