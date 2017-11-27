# ImgBandBiasLayer
## ImgBandBiasLayerTest
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
      "class": "com.simiacryptus.mindseye.layers.cudnn.f32.ImgBandBiasLayer",
      "id": "0910987d-3688-428c-a892-e2c400000403",
      "isFrozen": false,
      "name": "ImgBandBiasLayer/0910987d-3688-428c-a892-e2c400000403",
      "bias": [
        0.0,
        0.0
      ]
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
    	[ [ -1.028, 1.188 ], [ 1.892, -0.408 ], [ -0.1, 0.808 ] ],
    	[ [ -0.18, 0.524 ], [ -0.084, -1.216 ], [ -0.92, -1.96 ] ],
    	[ [ -1.828, 0.568 ], [ -1.888, -1.04 ], [ 0.776, 1.224 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ -1.027999997138977, 1.187999963760376 ], [ 1.8919999599456787, -0.40799999237060547 ], [ -0.10000000149011612, 0.8080000281333923 ] ],
    	[ [ -0.18000000715255737, 0.5239999890327454 ], [ -0.08399999886751175, -1.215999960899353 ], [ -0.9200000166893005, -1.9600000381469727 ] ],
    	[ [ -1.8279999494552612, 0.5680000185966492 ], [ -1.8880000114440918, -1.0399999618530273 ], [ 0.7760000228881836, 1.2239999771118164 ] ]
    ]
```



### Differential Validation
Code from [LayerTestBase.java:139](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L139) executed in 0.04 seconds: 
```java
    getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Finite-Difference Derivative Accuracy:
    absoluteTol: 1.8513e-05 +- 7.7854e-05 [0.0000e+00 - 1.0262e-03] (360#)
    relativeTol: 9.2573e-05 +- 8.6306e-05 [8.4638e-06 - 5.1334e-04] (36#)
    
```

### Performance
Code from [LayerTestBase.java:144](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.08 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 2.3174 +- 0.2436 [2.1202 - 4.2747]
    Learning performance: 2.8091 +- 0.2238 [2.3596 - 3.5993]
    
```

