# ImgConcatLayer
## ImgConcatLayerTest
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
      "class": "com.simiacryptus.mindseye.layers.cudnn.f32.ImgConcatLayer",
      "id": "0910987d-3688-428c-a892-e2c400000407",
      "isFrozen": false,
      "name": "ImgConcatLayer/0910987d-3688-428c-a892-e2c400000407",
      "maxBands": -1
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
    	[ [ -1.424 ], [ 1.556 ] ],
    	[ [ 1.96 ], [ -1.392 ] ]
    ],
    [
    	[ [ 0.524 ], [ -0.192 ] ],
    	[ [ 0.192 ], [ -0.404 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ -1.4240000247955322, 0.5239999890327454 ], [ 1.555999994277954, -0.19200000166893005 ] ],
    	[ [ 1.9600000381469727, 0.19200000166893005 ], [ -1.3919999599456787, -0.40400001406669617 ] ]
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
    absoluteTol: 1.6086e-05 +- 4.8288e-05 [0.0000e+00 - 1.6594e-04] (64#)
    relativeTol: 6.4338e-05 +- 3.2259e-05 [8.4638e-06 - 8.2963e-05] (8#)
    
```

### Performance
Code from [LayerTestBase.java:144](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.06 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 2.2805 +- 0.4031 [1.9606 - 5.5115]
    Learning performance: 1.1960 +- 0.2399 [1.0430 - 3.3912]
    
```

