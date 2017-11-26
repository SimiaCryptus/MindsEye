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
      "class": "com.simiacryptus.mindseye.layers.cudnn.f64.SimpleConvolutionLayer",
      "id": "b385277b-2d2d-42fe-8250-210c0000eca4",
      "isFrozen": false,
      "name": "SimpleConvolutionLayer/b385277b-2d2d-42fe-8250-210c0000eca4",
      "filter": {
        "dimensions": [
          3,
          3,
          1
        ],
        "data": [
          1.044,
          -0.676,
          1.024,
          -1.268,
          -1.204,
          -0.336,
          0.68,
          0.66,
          1.784
        ]
      },
      "strideX": 1,
      "strideY": 1,
      "simple": false
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
    	[ [ -1.66 ], [ -1.324 ], [ -1.796 ] ],
    	[ [ 1.872 ], [ -0.028 ], [ 0.652 ] ],
    	[ [ 0.668 ], [ 0.556 ], [ -0.684 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 0.49073599999999973 ], [ 3.7017440000000006 ], [ 0.44276799999999994 ] ],
    	[ [ -3.299536 ], [ -4.492064 ], [ -1.316656 ] ],
    	[ [ -1.837792 ], [ 4.2505440000000005 ], [ 0.9214720000000001 ] ]
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
      "id": "b385277b-2d2d-42fe-8250-210c0000eca6",
      "isFrozen": false,
      "name": "ConvolutionLayer/b385277b-2d2d-42fe-8250-210c0000eca6",
      "filter": {
        "dimensions": [
          3,
          3,
          1
        ],
        "data": [
          1.044,
          -0.676,
          1.024,
          -1.268,
          -1.204,
          -0.336,
          0.68,
          0.66,
          1.784
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
    	[ [ -1.66 ], [ -1.324 ], [ -1.796 ] ],
    	[ [ 1.872 ], [ -0.028 ], [ 0.652 ] ],
    	[ [ 0.668 ], [ 0.556 ], [ -0.684 ] ]
    ]
    Error: [
    	[ [ 0.0 ], [ 0.0 ], [ 0.0 ] ],
    	[ [ 0.0 ], [ 0.0 ], [ 0.0 ] ],
    	[ [ 0.0 ], [ 0.0 ], [ 0.0 ] ]
    ]
    Accuracy:
    absoluteTol: 0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (9#)
    relativeTol: 0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (9#)
    
```

### Differential Validation
Code from [LayerTestBase.java:130](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L130) executed in 0.03 seconds: 
```java
    getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Finite-Difference Derivative Accuracy:
    absoluteTol: 1.1315e-10 +- 1.6205e-10 [0.0000e+00 - 7.1697e-10] (162#)
    relativeTol: 4.5877e-10 +- 1.5177e-09 [2.6801e-12 - 9.6495e-09] (98#)
    
```

### Performance
Code from [LayerTestBase.java:135](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L135) executed in 13.53 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 3.5728 +- 0.4532 [3.1604 - 12.8041]
    Learning performance: 4.3646 +- 35.6876 [2.3026 - 3570.8818]
    
```

