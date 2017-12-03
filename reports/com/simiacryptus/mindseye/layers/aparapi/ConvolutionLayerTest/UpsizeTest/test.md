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
      "id": "e2d0bffa-47dc-4875-864f-3d3d00000018",
      "isFrozen": false,
      "name": "ConvolutionLayer/e2d0bffa-47dc-4875-864f-3d3d00000018",
      "filter": {
        "dimensions": [
          3,
          3,
          6
        ],
        "data": [
          1.52,
          1.728,
          1.188,
          1.088,
          0.656,
          1.752,
          1.692,
          -1.904,
          1.54,
          0.232,
          -0.272,
          0.968,
          0.388,
          0.004,
          1.7,
          -0.76,
          -1.884,
          1.712,
          -0.948,
          1.74,
          -0.832,
          1.564,
          1.124,
          -0.176,
          -0.156,
          1.42,
          -1.264,
          -1.856,
          -0.616,
          0.42,
          1.94,
          1.804,
          0.952,
          -1.636,
          -1.108,
          -0.72,
          1.192,
          1.9,
          1.052,
          -0.3,
          -1.972,
          -1.588,
          -0.972,
          0.448,
          0.012,
          0.832,
          -1.508,
          -0.924,
          -0.396,
          0.64,
          1.496,
          -1.328,
          -0.768,
          1.436
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
    	[ [ 1.184, 1.88 ], [ 1.988, 2.0 ], [ 0.812, 0.064 ] ],
    	[ [ -0.3, -1.392 ], [ -0.976, 0.92 ], [ -1.108, 0.2 ] ],
    	[ [ -1.388, 0.968 ], [ -0.364, 0.388 ], [ -1.0, 0.368 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ -1.6896, 2.5156479999999997, 0.44172799999999995 ] ]
    ]
```



### Differential Validation
Code from [LayerTestBase.java:139](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L139) executed in 0.32 seconds: 
```java
    getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Feedback for input 0
    Inputs: [
    	[ [ 1.184, 1.88 ], [ 1.988, 2.0 ], [ 0.812, 0.064 ] ],
    	[ [ -0.3, -1.392 ], [ -0.976, 0.92 ], [ -1.108, 0.2 ] ],
    	[ [ -1.388, 0.968 ], [ -0.364, 0.388 ], [ -1.0, 0.368 ] ]
    ]
    Output: [
    	[ [ -1.6896, 2.5156479999999997, 0.44172799999999995 ] ]
    ]
    Measured: [ [ 1.5199999999992997, 0.23200000000223042, -0.948000000000615 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ -1.8560000000000798, 1.1919999999987496, 0.8320000000000549 ] ]
    Implemented: [ [ 1.52, 0.232, -0.948 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ -1.856, 1.192, 0.832 ] ]
    Error: [ [ -7.003286839335487E-13, 2.230410300896324E-12, -6.150635556423367E-13 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ -7.971401316808624E-14, -1.2503331703328513E-12, 5.495603971894525E-14 ] ]
    Learning Gradient for weight set 0
    Inputs: [
    	[ [ 1.184, 1.88 ], [ 1.988, 2.0 ], [ 0.812, 0.064 ] ],
    	[ [ -0.3, -1.392 ], [ -0.976, 0.92 ], [ -1.108, 0.2 ] ],
    	[ [ -1.388, 0.968 ], [ -0.364, 0.388 ], [ -1.0, 0.368 ] ]
    ]
    Outputs: [
    	[ [ -1.6896, 2.5156479999999997, 0.44172799999999995 ] ]
    ]
    Measured Gradient: [ [ 1.18400000000074, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 1.1840000000029605, 0.0 ] ]
    Implemented Gradient: [ [ 1.184, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 1.184, 0.0 ] ]
    Error: [ [ 7.400746682151293E-13, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 2.9605207174654424E-12, 0.0 ] ]
    Finite-Difference Derivative Accuracy:
    absoluteTol: 5.5241e-14 +- 2.9992e-13 [0.0000e+00 - 2.9605e-12] (216#)
    relativeTol: 7.0808e-13 +- 1.2742e-12 [2.1475e-14 - 4.8069e-12] (12#)
    
```

### Performance
Code from [LayerTestBase.java:144](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.75 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 27.3360 +- 3.5857 [22.7071 - 43.0803]
    Learning performance: 20.4140 +- 1.2983 [18.1589 - 24.4426]
    
```

