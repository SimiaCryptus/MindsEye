# ConvolutionLayer
## UpsizeTest
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
      "class": "com.simiacryptus.mindseye.layers.aparapi.ConvolutionLayer",
      "id": "0910987d-3688-428c-a892-e2c400000009",
      "isFrozen": false,
      "name": "ConvolutionLayer/0910987d-3688-428c-a892-e2c400000009",
      "filter": {
        "dimensions": [
          3,
          3,
          6
        ],
        "data": [
          -0.532,
          -1.908,
          -0.696,
          -1.968,
          -0.484,
          -1.724,
          -1.844,
          0.912,
          -0.46,
          1.164,
          1.668,
          -0.72,
          1.3,
          -1.792,
          -0.636,
          -0.424,
          1.808,
          0.236,
          -0.044,
          -1.8,
          -1.996,
          1.86,
          1.064,
          -1.448,
          -1.72,
          -0.876,
          1.636,
          1.592,
          -1.776,
          1.684,
          1.528,
          0.292,
          0.42,
          -0.024,
          -1.236,
          -1.752,
          -1.468,
          -1.28,
          -1.828,
          -0.248,
          -1.824,
          0.44,
          0.44,
          -0.232,
          -1.744,
          -1.036,
          -0.844,
          1.456,
          0.768,
          -1.904,
          0.888,
          -0.64,
          -0.284,
          0.4
        ]
      },
      "skip": {
        "dimensions": [
          1,
          1
        ]
      },
      "simple": false
    }
```



### Example Input/Output Pair
Code from [LayerTestBase.java:121](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L121) executed in 0.01 seconds: 
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
    	[ [ -1.236, 1.96 ], [ -0.768, 0.06 ], [ -1.3, -1.064 ] ],
    	[ [ -1.088, -0.232 ], [ 0.724, 0.94 ], [ 0.268, -1.192 ] ],
    	[ [ -0.428, 2.0 ], [ -1.624, -1.384 ], [ 1.292, 1.328 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 3.7778720000000003, -4.315984, -1.976176 ] ]
    ]
```



### Differential Validation
Code from [LayerTestBase.java:139](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L139) executed in 0.32 seconds: 
```java
    getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Finite-Difference Derivative Accuracy:
    absoluteTol: 1.3460e-12 +- 7.5001e-12 [0.0000e+00 - 7.8565e-11] (216#)
    relativeTol: 2.8210e-11 +- 6.4618e-11 [1.8250e-12 - 2.4147e-10] (12#)
    
```

### Performance
Code from [LayerTestBase.java:144](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.67 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 24.0327 +- 3.1374 [20.2848 - 35.9757]
    Learning performance: 18.9503 +- 1.5675 [17.0389 - 25.8931]
    
```

