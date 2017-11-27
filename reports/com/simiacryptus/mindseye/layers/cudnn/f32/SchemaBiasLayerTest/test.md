# SchemaBiasLayer
## SchemaBiasLayerTest
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
      "class": "com.simiacryptus.mindseye.layers.cudnn.f32.SchemaBiasLayer",
      "id": "0910987d-3688-428c-a892-e2c400000417",
      "isFrozen": false,
      "name": "SchemaBiasLayer/0910987d-3688-428c-a892-e2c400000417",
      "selected": [
        "test1",
        "test2"
      ],
      "features": {
        "test2": 0.0,
        "test1": 0.0
      }
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
    	[ [ 1.016, -0.932 ], [ -0.16, -0.996 ], [ 1.4, -1.536 ] ],
    	[ [ -0.72, 1.372 ], [ -0.248, -1.252 ], [ 0.88, 0.744 ] ],
    	[ [ -0.112, 1.348 ], [ -0.184, -0.244 ], [ -0.168, 1.916 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 1.0160000324249268, -0.9319999814033508 ], [ -0.1599999964237213, -0.9959999918937683 ], [ 1.399999976158142, -1.5360000133514404 ] ],
    	[ [ -0.7200000286102295, 1.371999979019165 ], [ -0.24799999594688416, -1.2519999742507935 ], [ 0.8799999952316284, 0.7440000176429749 ] ],
    	[ [ -0.1120000034570694, 1.3480000495910645 ], [ -0.18400000035762787, -0.24400000274181366 ], [ -0.1679999977350235, 1.9160000085830688 ] ]
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
    absoluteTol: 1.5484e-05 +- 7.2998e-05 [0.0000e+00 - 1.0262e-03] (360#)
    relativeTol: 7.7426e-05 +- 8.9076e-05 [8.4638e-06 - 5.1334e-04] (36#)
    
```

### Performance
Code from [LayerTestBase.java:144](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.08 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 2.4722 +- 0.7391 [2.1658 - 7.7714]
    Learning performance: 2.8596 +- 0.3770 [2.5278 - 4.8988]
    
```

