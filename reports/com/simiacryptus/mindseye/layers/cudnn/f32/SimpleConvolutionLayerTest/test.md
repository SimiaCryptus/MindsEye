# SimpleConvolutionLayer
## SimpleConvolutionLayerTest
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
      "class": "com.simiacryptus.mindseye.layers.cudnn.f32.SimpleConvolutionLayer",
      "id": "b385277b-2d2d-42fe-8250-210c000000da",
      "isFrozen": false,
      "name": "SimpleConvolutionLayer/b385277b-2d2d-42fe-8250-210c000000da",
      "filter": {
        "dimensions": [
          3,
          3,
          1
        ],
        "data": [
          -0.704,
          -0.276,
          -1.9,
          0.328,
          -0.468,
          -0.112,
          -0.824,
          -1.988,
          1.936
        ]
      },
      "simple": false,
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
    	[ [ 0.42 ], [ 1.064 ], [ 1.72 ] ],
    	[ [ -0.584 ], [ 1.768 ], [ 1.736 ] ],
    	[ [ 1.456 ], [ 1.288 ], [ -1.46 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ -1.926448106765747 ], [ -1.968656063079834 ], [ -3.8076162338256836 ] ],
    	[ [ -2.712480068206787 ], [ -2.4690561294555664 ], [ -4.000160217285156 ] ],
    	[ [ -4.3306884765625 ], [ -7.721392631530762 ], [ 1.3511523008346558 ] ]
    ]
```



### Reference Implementation
Code from [LayerTestBase.java:123](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L123) executed in 0.02 seconds: 
```java
    System.out.println(new GsonBuilder().setPrettyPrinting().create().toJson(referenceLayer.getJson()));
    getEquivalencyTester().test(referenceLayer, layer, inputPrototype);
```
Logging: 
```
    {
      "class": "com.simiacryptus.mindseye.layers.aparapi.ConvolutionLayer",
      "id": "b385277b-2d2d-42fe-8250-210c000000dc",
      "isFrozen": false,
      "name": "ConvolutionLayer/b385277b-2d2d-42fe-8250-210c000000dc",
      "filter": {
        "dimensions": [
          3,
          3,
          1
        ],
        "data": [
          -0.704,
          -0.276,
          -1.9,
          0.328,
          -0.468,
          -0.112,
          -0.824,
          -1.988,
          1.936
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
    	[ [ 0.42 ], [ 1.064 ], [ 1.72 ] ],
    	[ [ -0.584 ], [ 1.768 ], [ 1.736 ] ],
    	[ [ 1.456 ], [ 1.288 ], [ -1.46 ] ]
    ]
    Error: [
    	[ [ -1.0676574691004248E-7 ], [ -6.30798340228722E-8 ], [ -2.3382568370422518E-7 ] ],
    	[ [ -6.820678732921692E-8 ], [ -1.2945556626675625E-7 ], [ -2.172851569781642E-7 ] ],
    	[ [ -4.7656249968497377E-7 ], [ -6.315307619075838E-7 ], [ 3.0083465607511073E-7 ] ]
    ]
    Accuracy:
    absoluteTol: 2.4751e-07 +- 1.8372e-07 [6.3080e-08 - 6.3153e-07] (9#)
    relativeTol: 3.8625e-08 +- 2.8316e-08 [1.2573e-08 - 1.1133e-07] (9#)
    
```

### Differential Validation
Code from [LayerTestBase.java:130](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L130) executed in 0.04 seconds: 
```java
    getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Finite-Difference Derivative Accuracy:
    absoluteTol: 1.8859e-03 +- 2.0889e-03 [0.0000e+00 - 9.5367e-03] (162#)
    relativeTol: 1.4971e-01 +- 3.5415e-01 [1.0591e-05 - 1.0000e+00] (115#)
    
```

### Performance
Code from [LayerTestBase.java:135](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L135) executed in 14.07 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 4.4366 +- 1.7675 [2.3169 - 12.7471]
    Learning performance: 5.1712 +- 1.9886 [2.9581 - 13.6049]
    
```

