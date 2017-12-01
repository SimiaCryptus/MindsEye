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
      "id": "f4569375-56fe-4e46-925c-95f40000001c",
      "isFrozen": false,
      "name": "ActivationLayer/f4569375-56fe-4e46-925c-95f40000001c",
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
    	[ [ 1.076, 1.616, -0.52 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 1.0759999752044678, 1.6160000562667847, 0.0 ] ]
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
    	[ [ 1.076, 1.616, -0.52 ] ]
    ]
    Output: [
    	[ [ 1.0759999752044678, 1.6160000562667847, 0.0 ] ]
    ]
    Measured: [ [ 1.0001659393310547, 0.0, 0.0 ], [ 0.0, 0.9989738464355469, 0.0 ], [ 0.0, 0.0, 0.0 ] ]
    Implemented: [ [ 1.0, 0.0, 0.0 ], [ 0.0, 1.0, 0.0 ], [ 0.0, 0.0, 0.0 ] ]
    Error: [ [ 1.659393310546875E-4, 0.0, 0.0 ], [ 0.0, -0.001026153564453125, 0.0 ], [ 0.0, 0.0, 0.0 ] ]
    Finite-Difference Derivative Accuracy:
    absoluteTol: 1.3245e-04 +- 3.2018e-04 [0.0000e+00 - 1.0262e-03] (9#)
    relativeTol: 2.9815e-04 +- 2.1519e-04 [8.2963e-05 - 5.1334e-04] (2#)
    
```

### Performance
Code from [LayerTestBase.java:144](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.07 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 2.5196 +- 1.3589 [1.8296 - 13.5565]
    Learning performance: 1.7136 +- 0.2026 [1.4506 - 2.5591]
    
```

