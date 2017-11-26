# ConvolutionLayer
## ConvolutionLayerTest
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
      "class": "com.simiacryptus.mindseye.layers.cudnn.f64.ConvolutionLayer",
      "id": "b385277b-2d2d-42fe-8250-210c000000e9",
      "isFrozen": false,
      "name": "ConvolutionLayer/b385277b-2d2d-42fe-8250-210c000000e9",
      "filter": {
        "dimensions": [
          3,
          3,
          4
        ],
        "data": [
          1.332,
          0.628,
          0.744,
          1.524,
          0.068,
          -1.26,
          -0.564,
          1.948,
          -1.104,
          -1.96,
          -1.932,
          -1.996,
          -1.224,
          0.54,
          0.056,
          1.588,
          -0.248,
          1.088,
          0.464,
          1.92,
          0.668,
          0.724,
          -1.2,
          1.388,
          -0.308,
          -0.752,
          1.48,
          1.04,
          -1.4,
          0.272,
          1.932,
          -0.312,
          -0.924,
          1.88,
          -1.124,
          -0.144
        ]
      },
      "strideX": 1,
      "strideY": 1
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
    	[ [ 0.492, -1.24 ], [ 0.2, 0.0 ], [ 0.084, -0.824 ] ],
    	[ [ -0.484, -0.328 ], [ -0.164, -1.348 ], [ 1.944, 0.6 ] ],
    	[ [ 1.588, -1.044 ], [ 0.536, -1.448 ], [ 1.904, -1.176 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ -0.17195199999999994, -0.8556000000000002 ], [ 2.3910880000000003, -4.6040160000000006 ], [ 5.288848000000001, -3.7620799999999996 ] ],
    	[ [ -2.8163679999999998, -3.6980639999999987 ], [ 1.3496960000000011, -11.291279999999997 ], [ 0.8300159999999996, -3.0715359999999987 ] ],
    	[ [ -1.950672, 2.411552 ], [ 4.822, -3.471296 ], [ 0.243072, 2.4598400000000002 ] ]
    ]
```



### Reference Implementation
Code from [LayerTestBase.java:123](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L123) executed in 0.01 seconds: 
```java
    System.out.println(new GsonBuilder().setPrettyPrinting().create().toJson(referenceLayer.getJson()));
    getEquivalencyTester().test(referenceLayer, layer, inputPrototype);
```
Logging: 
```
    {
      "class": "com.simiacryptus.mindseye.layers.aparapi.ConvolutionLayer",
      "id": "b385277b-2d2d-42fe-8250-210c000000f1",
      "isFrozen": false,
      "name": "ConvolutionLayer/b385277b-2d2d-42fe-8250-210c000000f1",
      "filter": {
        "dimensions": [
          3,
          3,
          4
        ],
        "data": [
          1.332,
          0.628,
          0.744,
          1.524,
          0.068,
          -1.26,
          -0.564,
          1.948,
          -1.104,
          -1.96,
          -1.932,
          -1.996,
          -1.224,
          0.54,
          0.056,
          1.588,
          -0.248,
          1.088,
          0.464,
          1.92,
          0.668,
          0.724,
          -1.2,
          1.388,
          -0.308,
          -0.752,
          1.48,
          1.04,
          -1.4,
          0.272,
          1.932,
          -0.312,
          -0.924,
          1.88,
          -1.124,
          -0.144
        ]
      },
      "skip": {
        "dimensions": [
          1,
          1
        ]
      },
      "simple": true
    }
    Inputs: [
    	[ [ 0.492, -1.24 ], [ 0.2, 0.0 ], [ 0.084, -0.824 ] ],
    	[ [ -0.484, -0.328 ], [ -0.164, -1.348 ], [ 1.944, 0.6 ] ],
    	[ [ 1.588, -1.044 ], [ 0.536, -1.448 ], [ 1.904, -1.176 ] ]
    ]
    Error: [
    	[ [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ] ],
    	[ [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ] ],
    	[ [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ] ]
    ]
    Accuracy:
    absoluteTol: 0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (18#)
    relativeTol: 0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (18#)
    
```

### Differential Validation
Code from [LayerTestBase.java:130](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L130) executed in 0.17 seconds: 
```java
    getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Finite-Difference Derivative Accuracy:
    absoluteTol: 9.2045e-11 +- 1.8835e-10 [0.0000e+00 - 1.5055e-09] (972#)
    relativeTol: 3.2987e-10 +- 6.8262e-10 [6.2518e-13 - 6.2108e-09] (380#)
    
```

### Performance
Code from [LayerTestBase.java:135](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L135) executed in 26.14 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 11.9689 +- 19.6714 [5.7167 - 1852.9266]
    Learning performance: 5.0976 +- 1.0667 [4.0609 - 17.1728]
    
```

