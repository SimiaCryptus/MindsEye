# ConvolutionLayer
## UpsizeTest
### Json Serialization
Code from [LayerTestBase.java:75](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L75) executed in 0.06 seconds: 
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
      "id": "046908b5-49e8-4a74-9716-382300000001",
      "isFrozen": false,
      "name": "ConvolutionLayer/046908b5-49e8-4a74-9716-382300000001",
      "filter": {
        "dimensions": [
          3,
          3,
          6
        ],
        "data": [
          -1.52,
          -0.088,
          -1.02,
          -0.02,
          1.144,
          0.72,
          -1.856,
          -0.96,
          -0.464,
          0.736,
          1.384,
          -1.144,
          1.18,
          0.764,
          -0.792,
          -1.164,
          1.924,
          -0.68,
          0.236,
          -0.268,
          1.324,
          -1.128,
          -1.564,
          1.212,
          0.148,
          -1.972,
          -0.5,
          1.168,
          1.612,
          -0.008,
          -1.332,
          -1.488,
          -1.828,
          1.228,
          1.048,
          -1.112,
          1.8,
          1.836,
          1.092,
          -0.588,
          -1.28,
          -0.228,
          1.408,
          -1.46,
          1.156,
          -0.772,
          -0.612,
          -1.496,
          1.104,
          -0.656,
          1.404,
          1.252,
          0.588,
          1.748
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
Code from [LayerTestBase.java:112](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L112) executed in 0.29 seconds: 
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
    	[ [ 0.008, -0.196 ], [ -0.26, -1.304 ], [ -0.832, 0.108 ] ],
    	[ [ 0.264, -0.144 ], [ -0.572, 1.14 ], [ 0.072, 0.648 ] ],
    	[ [ -1.6, -1.964 ], [ -0.32, 1.344 ], [ 0.8, -0.392 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ -0.241088, -0.346912, 0.1532 ] ]
    ]
```



### Differential Validation
Code from [LayerTestBase.java:130](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L130) executed in 0.42 seconds: 
```java
    getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Finite-Difference Derivative Accuracy:
    absoluteTol: 4.3757e-13 +- 2.2722e-12 [0.0000e+00 - 1.9899e-11] (216#)
    relativeTol: 7.7329e-11 +- 1.0821e-10 [8.0174e-13 - 2.6318e-10] (12#)
    
```

### Performance
Code from [LayerTestBase.java:135](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L135) executed in 58.59 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 20.4815 +- 2.0904 [17.5062 - 179.7758]
    Learning performance: 17.0641 +- 1.0375 [14.7704 - 35.1977]
    
```

