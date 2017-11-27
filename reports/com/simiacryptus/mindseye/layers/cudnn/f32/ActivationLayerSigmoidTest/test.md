# ActivationLayer
## ActivationLayerSigmoidTest
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
      "id": "0910987d-3688-428c-a892-e2c400000011",
      "isFrozen": false,
      "name": "ActivationLayer/0910987d-3688-428c-a892-e2c400000011",
      "mode": 0
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
    	[ [ -1.916, 0.292 ], [ -0.228, -1.248 ], [ 0.708, 1.568 ] ],
    	[ [ 1.812, 0.42 ], [ 0.196, -0.244 ], [ -0.152, -0.1 ] ],
    	[ [ 0.504, 1.688 ], [ 1.768, -1.692 ], [ -1.764, -0.176 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 0.1283082813024521, 0.5724857449531555 ], [ 0.44324561953544617, 0.22304655611515045 ], [ 0.6699590682983398, 0.8274983167648315 ] ],
    	[ [ 0.8596034646034241, 0.603483259677887 ], [ 0.5488437414169312, 0.43930086493492126 ], [ 0.4620729982852936, 0.47502079606056213 ] ],
    	[ [ 0.6233989000320435, 0.8439609408378601 ], [ 0.854208767414093, 0.1555129885673523 ], [ 0.14629007875919342, 0.45611321926116943 ] ]
    ]
```



### Differential Validation
Code from [LayerTestBase.java:139](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L139) executed in 0.03 seconds: 
```java
    getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Finite-Difference Derivative Accuracy:
    absoluteTol: 1.0249e-05 +- 5.0595e-05 [0.0000e+00 - 3.6678e-04] (324#)
    relativeTol: 4.8646e-04 +- 3.3381e-04 [4.3668e-05 - 1.1778e-03] (18#)
    
```

### Performance
Code from [LayerTestBase.java:144](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.05 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 2.1638 +- 3.6188 [1.4078 - 37.4633]
    Learning performance: 1.7790 +- 0.2352 [1.3394 - 3.7332]
    
```

