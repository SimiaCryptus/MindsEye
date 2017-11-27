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
      "class": "com.simiacryptus.mindseye.layers.cudnn.f64.ActivationLayer",
      "id": "0910987d-3688-428c-a892-e2c400000429",
      "isFrozen": false,
      "name": "ActivationLayer/0910987d-3688-428c-a892-e2c400000429",
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
    	[ [ -1.692, 0.616 ], [ -0.924, -0.056 ], [ 1.156, -0.484 ] ],
    	[ [ -0.428, -1.492 ], [ -0.612, 1.428 ], [ -1.124, 0.896 ] ],
    	[ [ -1.596, 1.808 ], [ 1.796, 1.552 ], [ 1.08, 1.72 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 0.15551300170651194, 0.6493082652711343 ], [ 0.28414356827121023, 0.4860036575196728 ], [ 0.7606051344021186, 0.38130802939870706 ] ],
    	[ [ 0.39460401408967927, 0.18362172811482294 ], [ 0.3516031059824349, 0.8065895013134284 ], [ 0.2452700786423919, 0.7101268083047296 ] ],
    	[ [ 0.1685414127401554, 0.8591199824064242 ], [ 0.8576613198329636, 0.8252024062848793 ], [ 0.7464939833376621, 0.8481288363433407 ] ]
    ]
```



### Differential Validation
Code from [LayerTestBase.java:139](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L139) executed in 0.02 seconds: 
```java
    getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Finite-Difference Derivative Accuracy:
    absoluteTol: 2.2183e-08 +- 9.5151e-08 [0.0000e+00 - 4.7829e-07] (324#)
    relativeTol: 1.2257e-06 +- 4.9826e-07 [6.9967e-08 - 1.7956e-06] (18#)
    
```

### Performance
Code from [LayerTestBase.java:144](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.05 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 1.7459 +- 0.6421 [1.3195 - 5.9105]
    Learning performance: 1.3659 +- 0.6507 [0.9604 - 7.2784]
    
```

