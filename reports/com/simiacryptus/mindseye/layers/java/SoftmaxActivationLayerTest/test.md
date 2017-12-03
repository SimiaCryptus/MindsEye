# SoftmaxActivationLayer
## SoftmaxActivationLayerTest
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
      "class": "com.simiacryptus.mindseye.layers.java.SoftmaxActivationLayer",
      "id": "e2d0bffa-47dc-4875-864f-3d3d0000155f",
      "isFrozen": false,
      "name": "SoftmaxActivationLayer/e2d0bffa-47dc-4875-864f-3d3d0000155f"
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
    [[ -0.868, -0.284, 1.36, -0.628 ]]
    --------------------
    Output: 
    [ 0.07493032803791297, 0.13436483134293611, 0.6954497247676499, 0.09525511585150101 ]
```



### Differential Validation
Code from [LayerTestBase.java:139](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L139) executed in 0.00 seconds: 
```java
    getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Feedback for input 0
    Inputs: [ -0.868, -0.284, 1.36, -0.628 ]
    Output: [ 0.07493032803791297, 0.13436483134293611, 0.6954497247676499, 0.09525511585150101 ]
    Measured: [ [ 0.06931872044896314, -0.010068428859533185, -0.05211249111125582, -0.007137800478312917 ], [ -0.010068369015875378, 0.11631517623594823, -0.09344740166450372, -0.012799405556679355 ], [ -0.05210925749302153, -0.09344215857354232, 0.2117952653801769, -0.06624384931222527 ], [ -0.007137785970195987, -0.012799455616774313, -0.0662478253932175, 0.08618506698004902 ] ]
    Implemented: [ [ 0.06931577397804374, -0.010068000889285052, -0.0521102760107163, -0.007137497078042375 ], [ -0.010068000889285052, 0.11631092344112044, -0.09344398497589661, -0.012798937575938774 ], [ -0.0521102760107163, -0.09344398497589661, 0.21179940508824993, -0.06624514410163698 ], [ -0.007137497078042375, -0.012798937575938774, -0.06624514410163698, 0.08618157875561813 ] ]
    Error: [ [ 2.9464709194043648E-6, -4.2797024813259754E-7, -2.215100539521009E-6, -3.034002705416469E-7 ], [ -3.681265903256181E-7, 4.2527948277831795E-6, -3.416688607107976E-6, -4.679807405812836E-7 ], [ 1.0185176947674623E-6, 1.826402354285217E-6, -4.13970807303099E-6, 1.294789411715458E-6 ], [ -2.8889215361160153E-7, -5.1804083553951E-7, -2.6812915805141113E-6, 3.4882244308864774E-6 ] ]
    Finite-Difference Derivative Accuracy:
    absoluteTol: 1.8534e-06 +- 1.4123e-06 [2.8889e-07 - 4.2528e-06] (16#)
    relativeTol: 1.7386e-05 +- 4.5236e-06 [9.7728e-06 - 2.1254e-05] (16#)
    
```

### Performance
Code from [LayerTestBase.java:144](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.01 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 0.2677 +- 0.0900 [0.2023 - 0.8549]
    Learning performance: 0.0019 +- 0.0015 [0.0000 - 0.0057]
    
```

