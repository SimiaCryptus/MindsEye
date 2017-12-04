# ConvolutionLayer
## ConvolutionLayerTest
### Json Serialization
Code from [LayerTestBase.java:83](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L83) executed in 0.00 seconds: 
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
      "id": "370a9587-74a1-4959-b406-fa450000045e",
      "isFrozen": false,
      "name": "ConvolutionLayer/370a9587-74a1-4959-b406-fa450000045e",
      "filter": {
        "dimensions": [
          1,
          1,
          4
        ],
        "data": [
          -1.44,
          -0.96,
          1.624,
          1.76
        ]
      },
      "strideX": 1,
      "strideY": 1
    }
```



### Example Input/Output Pair
Code from [LayerTestBase.java:120](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L120) executed in 0.00 seconds: 
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
    	[ [ 1.412, -0.528 ], [ -1.712, 1.224 ], [ 1.0, 0.664 ], [ -1.368, -1.336 ], [ -1.744, 1.352 ] ],
    	[ [ 1.156, -1.52 ], [ -0.316, -0.58 ], [ -1.876, 1.728 ], [ -0.772, 0.524 ], [ 0.624, -0.388 ] ],
    	[ [ -0.644, 1.264 ], [ -0.044, -0.544 ], [ -0.072, 1.068 ], [ -0.584, 0.2 ], [ 1.872, -0.396 ] ],
    	[ [ 1.592, 0.18 ], [ 0.38, 1.952 ], [ -1.928, 1.676 ], [ 0.196, -0.972 ], [ 1.564, -0.952 ] ],
    	[ [ -1.556, 0.816 ], [ -0.128, -0.964 ], [ -1.244, 1.716 ], [ -1.752, -1.784 ], [ 0.128, -0.832 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ -2.890752, -2.2847999999999997 ], [ 4.453056, 3.79776 ], [ -0.3616639999999998, 0.2086400000000001 ], [ -0.19974400000000014, -1.0380800000000001 ], [ 4.707008, 4.0537600000000005 ] ],
    	[ [ -4.13312, -3.78496 ], [ -0.48688, -0.71744 ], [ 5.507712, 4.842239999999999 ], [ 1.9626560000000002, 1.66336 ], [ -1.528672, -1.28192 ] ],
    	[ [ 2.980096, 2.84288 ], [ -0.8200960000000002, -0.9152000000000001 ], [ 1.8381120000000002, 1.9488 ], [ 1.16576, 0.9126399999999999 ], [ -3.338784, -2.4940800000000003 ] ],
    	[ [ -2.0001599999999997, -1.2115200000000002 ], [ 2.6228480000000003, 3.07072 ], [ 5.498144, 4.80064 ], [ -1.860768, -1.89888 ], [ -3.798208, -3.1769600000000002 ] ],
    	[ [ 3.565824, 2.92992 ], [ -1.381216, -1.57376 ], [ 4.578144, 4.2144 ], [ -0.37433600000000045, -1.45792 ], [ -1.535488, -1.5872 ] ]
    ]
```



### Reference Implementation
Code from [LayerTestBase.java:131](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L131) executed in 0.01 seconds: 
```java
    System.out.println(new GsonBuilder().setPrettyPrinting().create().toJson(referenceLayer.getJson()));
    getEquivalencyTester().test(referenceLayer, layer, inputPrototype);
```
Logging: 
```
    {
      "class": "com.simiacryptus.mindseye.layers.aparapi.ConvolutionLayer",
      "id": "370a9587-74a1-4959-b406-fa4500000465",
      "isFrozen": false,
      "name": "ConvolutionLayer/370a9587-74a1-4959-b406-fa4500000465",
      "filter": {
        "dimensions": [
          1,
          1,
          4
        ],
        "data": [
          -1.44,
          -0.96,
          1.624,
          1.76
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
    	[ [ 1.412, -0.528 ], [ -1.712, 1.224 ], [ 1.0, 0.664 ], [ -1.368, -1.336 ], [ -1.744, 1.352 ] ],
    	[ [ 1.156, -1.52 ], [ -0.316, -0.58 ], [ -1.876, 1.728 ], [ -0.772, 0.524 ], [ 0.624, -0.388 ] ],
    	[ [ -0.644, 1.264 ], [ -0.044, -0.544 ], [ -0.072, 1.068 ], [ -0.584, 0.2 ], [ 1.872, -0.396 ] ],
    	[ [ 1.592, 0.18 ], [ 0.38, 1.952 ], [ -1.928, 1.676 ], [ 0.196, -0.972 ], [ 1.564, -0.952 ] ],
    	[ [ -1.556, 0.816 ], [ -0.128, -0.964 ], [ -1.244, 1.716 ], [ -1.752, -1.784 ], [ 0.128, -0.832 ] ]
    ]
    Error: [
    	[ [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ] ],
    	[ [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ] ],
    	[ [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ] ],
    	[ [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ] ],
    	[ [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ] ]
    ]
    Accuracy:
    absoluteTol: 0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (50#)
    relativeTol: 0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (50#)
    
```

### Batch Execution
Code from [LayerTestBase.java:138](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L138) executed in 0.02 seconds: 
```java
    BatchingTester batchingTester = getBatchingTester();
    return batchingTester==null?null:batchingTester.test(layer, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (1000#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (1000#)}
```



### Differential Validation
Code from [LayerTestBase.java:144](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.28 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ 1.412, -0.528 ], [ -1.712, 1.224 ], [ 1.0, 0.664 ], [ -1.368, -1.336 ], [ -1.744, 1.352 ] ],
    	[ [ 1.156, -1.52 ], [ -0.316, -0.58 ], [ -1.876, 1.728 ], [ -0.772, 0.524 ], [ 0.624, -0.388 ] ],
    	[ [ -0.644, 1.264 ], [ -0.044, -0.544 ], [ -0.072, 1.068 ], [ -0.584, 0.2 ], [ 1.872, -0.396 ] ],
    	[ [ 1.592, 0.18 ], [ 0.38, 1.952 ], [ -1.928, 1.676 ], [ 0.196, -0.972 ], [ 1.564, -0.952 ] ],
    	[ [ -1.556, 0.816 ], [ -0.128, -0.964 ], [ -1.244, 1.716 ], [ -1.752, -1.784 ], [ 0.128, -0.832 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.11781459120907824, negative=27, min=-0.832, max=-0.832, mean=-0.04496000000000003, count=50.0, positive=23, stdDev=1.174166886945804, zeros=0}
    Output: [
    	[ [ -2.890752, -2.2847999999999997 ], [ 4.453056, 3.79776 ], [ -0.3616639999999998, 0.2086400000000001 ], [ -0.19974400000000014, -1.0380800000000001 ], [ 4.707008, 4.0537600000000005 ] ],
    	[ [ -4.13312, -3.78496 ], [ -0.48688, -0.71744 ], [ 5.507712, 4.842239999999999 ], [ 1.9626560000000002, 1.66336 ], [ -1.528672, -1.28192 ] ],
    	[ [ 2.980096, 2.84288 ], [ -0.8200960000000002, -0.9152000000000001 ], [ 1.8381120000000002, 1.9488 ], [ 1.16576, 0.9126399999999999 ], [ -3.338784, -2.4940800000000003 ] ],
    	[ [ -2.0001599999999997, -1.2115200000000002 ], [ 2.6228480000000003, 3.07072 ], [ 5.498144, 4.80064 ], [ -1.860768, -1.89888 ], [ -3.798208, -3.1769600000000002 ] ],
    	[ [ 3.565824, 2.92992 ], [ -1.381216, -1.57376 ], [ 4.578144, 4.2144 ], [ -0.37433600000000045, -1.45792 ], [ -1.535488, -1.5872 ] ]
    ]
    Outputs Statistics: {meanExponent=0.27809519412985917, negative=27, min=-1.5872, max=-1.5872, mean=0.5206502399999999, count=50.0, positive=23, stdDev=2.8074716849527266, zeros=0}
    Feedback for input 0
    Inputs Values: [
    	[ [ 1.412, -0.528 ], [ -1.712, 1.224 ], [ 1.0, 0.664 ], [ -1.368, -1.336 ], [ -1.744, 1.352 ] ],
    	[ [ 1.156, -1.52 ], [ -0.316, -0.58 ], [ -1.876, 1.728 ], [ -0.772, 0.524 ], [ 0.624, -0.388 ] ],
    	[ [ -0.644, 1.264 ], [ -0.044, -0.544 ], [ -0.072, 1.068 ], [ -0.584, 0.2 ], [ 1.872, -0.396 ] ],
    	[ [ 1.592, 0.18 ], [ 
```
...[skipping 2600 bytes](etc/1.txt)...
```
    Implemented Gradient: [ [ 1.412, 1.156, -0.644, 1.592, -1.556, -1.712, -0.316, -0.044, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ -0.528, -1.52, 1.264, 0.18, 0.816, 1.224, -0.58, -0.544, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ] ]
    Implemented Statistics: {meanExponent=-0.11781459120907818, negative=54, min=-0.832, max=-0.832, mean=-0.022479999999999993, count=200.0, positive=46, stdDev=0.8305656443653325, zeros=100}
    Measured Gradient: [ [ 1.4120000000028554, 1.1559999999999349, -0.6439999999985346, 1.5919999999969292, -1.5559999999981144, -1.7120000000048208, -0.316000000000205, -0.044000000000155026, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ -0.5280000000018603, -1.5199999999992997, 1.26400000000082, 0.17999999999851468, 0.8159999999968193, 1.223999999995229, -0.5800000000000249, -0.5440000000001, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ] ]
    Measured Statistics: {meanExponent=-0.11781459120931613, negative=54, min=-0.8319999999994998, max=-0.8319999999994998, mean=-0.0224800000000136, count=200.0, positive=46, stdDev=0.8305656443652419, zeros=100}
    Gradient Error: [ [ 2.8554936193359026E-12, -6.505906924303417E-14, 1.4653833702027441E-12, -3.070876886113183E-12, 1.885602785023366E-12, -4.820810417527355E-12, -2.0500268149703516E-13, -1.550287676010953E-13, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ -1.8602897000619123E-12, 7.003286839335487E-13, 8.200107259881406E-13, -1.4853118734947657E-12, -3.180677943248611E-12, -4.771072426024148E-12, -2.4980018054066022E-14, -9.992007221626409E-14, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ] ]
    Error Statistics: {meanExponent=-12.152395307062552, negative=53, min=5.00155472593633E-13, max=5.00155472593633E-13, mean=-1.3601342274682793E-14, count=200.0, positive=47, stdDev=1.4634561128815414E-12, zeros=100}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 1.0941e-13 +- 5.4229e-13 [0.0000e+00 - 6.9063e-12] (2700#)
    relativeTol: 7.8487e-13 +- 1.1062e-12 [3.3077e-15 - 1.0854e-11] (200#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=1.0941e-13 +- 5.4229e-13 [0.0000e+00 - 6.9063e-12] (2700#), relativeTol=7.8487e-13 +- 1.1062e-12 [3.3077e-15 - 1.0854e-11] (200#)}
```



### Performance
Code from [LayerTestBase.java:149](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L149) executed in 0.27 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 10.0398 +- 1.9934 [8.5722 - 18.5635]
    Learning performance: 6.5677 +- 1.3538 [5.5913 - 15.1010]
    
```

