# ImgConcatLayer
## BandLimitTest
### Json Serialization
Code from [LayerTestBase.java:84](../../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L84) executed in 0.00 seconds: 
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
      "id": "0910987d-3688-428c-a892-e2c40000040b",
      "isFrozen": false,
      "name": "ImgConcatLayer/0910987d-3688-428c-a892-e2c40000040b",
      "maxBands": 3
    }
```



### Example Input/Output Pair
Code from [LayerTestBase.java:121](../../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L121) executed in 0.00 seconds: 
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
    	[ [ 0.12, 0.468 ], [ 0.896, -1.652 ] ],
    	[ [ 0.064, 0.244 ], [ 1.176, 0.612 ] ]
    ],
    [
    	[ [ 1.44, 1.684 ], [ -1.628, 0.16 ] ],
    	[ [ 1.06, 0.772 ], [ 0.46, -0.276 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 0.11999999731779099, 0.46799999475479126, 1.440000057220459 ], [ 0.8960000276565552, -1.6519999504089355, -1.628000020980835 ] ],
    	[ [ 0.06400000303983688, 0.24400000274181366, 1.059999942779541 ], [ 1.1759999990463257, 0.6119999885559082, 0.46000000834465027 ] ]
    ]
```



### Differential Validation
Code from [LayerTestBase.java:139](../../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L139) executed in 0.03 seconds: 
```java
    getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Finite-Difference Derivative Accuracy:
    absoluteTol: 1.8415e-05 +- 1.1145e-04 [0.0000e+00 - 1.0262e-03] (192#)
    relativeTol: 1.4736e-04 +- 1.7138e-04 [8.4638e-06 - 5.1334e-04] (12#)
    
```

### Performance
Code from [LayerTestBase.java:144](../../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.06 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 2.1642 +- 0.2463 [1.8210 - 3.5651]
    Learning performance: 1.1568 +- 0.1135 [1.0231 - 2.1088]
    
```

