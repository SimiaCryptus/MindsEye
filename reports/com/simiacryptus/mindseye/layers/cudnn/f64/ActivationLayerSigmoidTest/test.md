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
      "class": "com.simiacryptus.mindseye.layers.cudnn.f64.ActivationLayer",
      "id": "b385277b-2d2d-42fe-8250-210c000000e5",
      "isFrozen": false,
      "name": "ActivationLayer/b385277b-2d2d-42fe-8250-210c000000e5",
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
    	[ [ -1.048, 1.1 ], [ -0.544, -0.488 ], [ 0.396, -0.1 ] ],
    	[ [ -1.092, 1.184 ], [ -1.012, -0.536 ], [ -0.684, 0.908 ] ],
    	[ [ -0.028, 1.96 ], [ 0.872, -1.436 ], [ 1.772, -1.06 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 0.25960934061621116, 0.7502601055951177 ], [ 0.36725757190544506, 0.38036482959282186 ], [ 0.5977262388897371, 0.47502081252106 ] ],
    	[ [ 0.25124185247230446, 0.7656662520457792 ], [ 0.26658862999609406, 0.3691185738336976 ], [ 0.3353691295623231, 0.7125907266074945 ] ],
    	[ [ 0.49300045729748126, 0.8765329524347759 ], [ 0.7051616859142585, 0.19216553584518478 ], [ 0.8547062143029203, 0.2573094546973142 ] ]
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
    absoluteTol: 2.0243e-09 +- 8.9422e-09 [0.0000e+00 - 4.7756e-08] (324#)
    relativeTol: 1.0040e-07 +- 4.9400e-08 [3.3123e-09 - 1.8776e-07] (18#)
    
```

### Performance
Code from [LayerTestBase.java:135](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L135) executed in 4.40 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 2.0135 +- 0.9136 [1.2026 - 11.9406]
    Learning performance: 1.1613 +- 0.2165 [0.8549 - 6.0102]
    
```

