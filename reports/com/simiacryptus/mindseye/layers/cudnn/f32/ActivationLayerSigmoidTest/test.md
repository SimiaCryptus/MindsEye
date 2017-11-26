# ActivationLayer
## ActivationLayerSigmoidTest
### Json Serialization
Code from [LayerTestBase.java:75](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L75) executed in 0.00 seconds: 
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
      "id": "b385277b-2d2d-42fe-8250-210c0000000d",
      "isFrozen": false,
      "name": "ActivationLayer/b385277b-2d2d-42fe-8250-210c0000000d",
      "mode": 0
    }
```



### Example Input/Output Pair
Code from [LayerTestBase.java:112](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L112) executed in 0.00 seconds: 
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
    	[ [ 1.044, -1.6 ], [ 1.024, 1.036 ], [ 0.904, 1.672 ] ],
    	[ [ 0.916, -0.236 ], [ 1.56, 0.488 ], [ -0.924, 1.688 ] ],
    	[ [ -0.992, -1.528 ], [ -1.34, -1.132 ], [ -1.98, 1.98 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 0.739621102809906, 0.1679816097021103 ], [ 0.7357510328292847, 0.7380775213241577 ], [ 0.7117708325386047, 0.8418422937393188 ] ],
    	[ [ 0.7142263650894165, 0.4412723183631897 ], [ 0.8263533711433411, 0.6196351647377014 ], [ 0.2841435670852661, 0.8439609408378601 ] ],
    	[ [ 0.2705172300338745, 0.17828649282455444 ], [ 0.2075100541114807, 0.24379220604896545 ], [ 0.12131884694099426, 0.8786811232566833 ] ]
    ]
```



### Differential Validation
Code from [LayerTestBase.java:130](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L130) executed in 0.03 seconds: 
```java
    getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Finite-Difference Derivative Accuracy:
    absoluteTol: 8.4982e-06 +- 4.6109e-05 [0.0000e+00 - 4.4233e-04] (324#)
    relativeTol: 4.5383e-04 +- 4.1126e-04 [3.2570e-05 - 1.5437e-03] (18#)
    
```

### Performance
Code from [LayerTestBase.java:135](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L135) executed in 4.32 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 1.5184 +- 0.3663 [1.1741 - 9.7634]
    Learning performance: 1.6583 +- 0.2858 [1.2055 - 6.3009]
    
```

