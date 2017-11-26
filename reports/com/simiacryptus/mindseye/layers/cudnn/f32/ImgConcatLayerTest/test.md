# ImgConcatLayer
## ImgConcatLayerTest
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
      "class": "com.simiacryptus.mindseye.layers.cudnn.f32.ImgConcatLayer",
      "id": "b385277b-2d2d-42fe-8250-210c000000c7",
      "isFrozen": false,
      "name": "ImgConcatLayer/b385277b-2d2d-42fe-8250-210c000000c7"
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
    	[ [ -1.468, -1.98 ], [ -1.372, -0.792 ], [ -0.936, 1.624 ] ],
    	[ [ 0.128, -0.764 ], [ 0.832, 0.996 ], [ -1.396, -0.776 ] ],
    	[ [ 1.4, -0.644 ], [ 1.872, -0.544 ], [ 1.684, -0.248 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ -1.468000054359436, -1.9800000190734863 ], [ -1.371999979019165, -0.7919999957084656 ], [ -0.9359999895095825, 1.6239999532699585 ] ],
    	[ [ 0.12800000607967377, -0.7639999985694885 ], [ 0.8320000171661377, 0.9959999918937683 ], [ -1.3960000276565552, -0.7760000228881836 ] ],
    	[ [ 1.399999976158142, -0.6439999938011169 ], [ 1.871999979019165, -0.5440000295639038 ], [ 1.684000015258789, -0.24799999594688416 ] ]
    ]
```



### Differential Validation
Code from [LayerTestBase.java:130](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L130) executed in 0.02 seconds: 
```java
    getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Finite-Difference Derivative Accuracy:
    absoluteTol: 9.4698e-06 +- 4.2554e-05 [0.0000e+00 - 4.3011e-04] (324#)
    relativeTol: 8.5225e-05 +- 3.5904e-05 [8.4638e-06 - 2.1510e-04] (18#)
    
```

### Performance
Code from [LayerTestBase.java:135](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L135) executed in 3.79 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 1.0520 +- 0.2940 [0.8948 - 9.1051]
    Learning performance: 1.5831 +- 24.0423 [1.0715 - 2405.4542]
    
```

