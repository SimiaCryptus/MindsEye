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
      "id": "c88cbdf1-1c2a-4a5e-b964-890900000003",
      "isFrozen": false,
      "name": "ConvolutionLayer/c88cbdf1-1c2a-4a5e-b964-890900000003",
      "filter": {
        "dimensions": [
          3,
          3,
          6
        ],
        "data": [
          -0.172,
          -1.92,
          -0.392,
          -1.536,
          -1.296,
          1.444,
          -1.92,
          0.08,
          -0.08,
          1.616,
          -0.016,
          -0.576,
          -0.884,
          1.36,
          0.396,
          1.976,
          -1.784,
          -0.636,
          -0.48,
          -1.552,
          -1.972,
          -1.892,
          -0.34,
          0.54,
          -0.148,
          1.024,
          1.6,
          1.088,
          -0.512,
          1.828,
          -0.088,
          1.704,
          1.368,
          1.672,
          -0.836,
          -0.664,
          0.916,
          1.892,
          -0.028,
          0.328,
          0.732,
          1.756,
          1.256,
          -0.552,
          1.772,
          0.908,
          1.596,
          0.228,
          -1.884,
          -1.312,
          -1.268,
          0.7,
          -1.084,
          -1.832
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
    	[ [ 1.292, -1.94 ], [ -0.604, -0.02 ], [ 0.968, -1.86 ] ],
    	[ [ 0.268, 0.536 ], [ -1.796, -1.096 ], [ 0.176, 0.396 ] ],
    	[ [ 1.4, -0.132 ], [ 1.244, 1.092 ], [ -1.2, 1.348 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ -2.332944, 0.3108320000000004, -2.3816800000000002 ] ]
    ]
```



### Differential Validation
Code from [LayerTestBase.java:139](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L139) executed in 0.29 seconds: 
```java
    getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Component: ConvolutionLayer/c88cbdf1-1c2a-4a5e-b964-890900000003
    Inputs: [
    	[ [ 1.292, -1.94 ], [ -0.604, -0.02 ], [ 0.968, -1.86 ] ],
    	[ [ 0.268, 0.536 ], [ -1.796, -1.096 ], [ 0.176, 0.396 ] ],
    	[ [ 1.4, -0.132 ], [ 1.244, 1.092 ], [ -1.2, 1.348 ] ]
    ]
    output=[
    	[ [ -2.332944, 0.3108320000000004, -2.3816800000000002 ] ]
    ]
    measured/actual: [ [ -0.17200000000272553, 1.6159999999976193, -0.47999999999603915 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 1.0880000000002, 0.9159999999996948, 0.9080000000016852 ] ]
    implemented/expected: [ [ -0.172, 1.616, -0.48 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 1.088, 0.916, 0.908 ] ]
    error: [ [ -2.725542014303528E-12, -2.3807622540061857E-12, 3.9608316626527085E-12 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 1.9984014443252818E-13, -3.0520030946945553E-13, 1.6852075290785251E-12 ] ]
    Component: ConvolutionLayer/c88cbdf1-1c2a-4a5e-b964-890900000003
    Inputs: [
    	[ [ 1.292, -1.94 ], [ -0.604, -0.02 ], [ 0.968, -1.86 ] ],
    	[ [ 0.268, 0.536 ], [ -1.796, -1.096 ], [ 0.176, 0.396 ] ],
    	[ [ 1.4, -0.132 ], [ 1.244, 1.092 ], [ -1.2, 1.348 ] ]
    ]
    Outputs: [
    	[ [ -2.332944, 0.3108320000000004, -2.3816800000000002 ] ]
    ]
    Measured Gradient: [ [ 1.2919999999994047, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 1.2919999999994047, 0.0 ] ]
    Implemented Gradient: [ [ 1.292, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 1.292, 0.0 ] ]
    Error: [ [ -5.953015858040089E-13, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, -5.953015858040089E-13, 0.0 ] ]
    Finite-Difference Derivative Accuracy:
    absoluteTol: 9.7290e-14 +- 5.3886e-13 [0.0000e+00 - 4.1660e-12] (216#)
    relativeTol: 1.4281e-12 +- 2.2407e-12 [7.0848e-14 - 7.9231e-12] (12#)
    
```

### Performance
Code from [LayerTestBase.java:144](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.72 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 26.1371 +- 3.3844 [22.5105 - 39.2131]
    Learning performance: 19.6615 +- 1.8567 [17.3552 - 30.1764]
    
```

