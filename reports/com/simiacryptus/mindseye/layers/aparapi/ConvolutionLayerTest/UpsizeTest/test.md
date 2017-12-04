# ConvolutionLayer
## UpsizeTest
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
      "class": "com.simiacryptus.mindseye.layers.aparapi.ConvolutionLayer",
      "id": "370a9587-74a1-4959-b406-fa4500000018",
      "isFrozen": false,
      "name": "ConvolutionLayer/370a9587-74a1-4959-b406-fa4500000018",
      "filter": {
        "dimensions": [
          3,
          3,
          6
        ],
        "data": [
          1.228,
          1.504,
          -1.844,
          1.356,
          -0.64,
          -0.196,
          1.456,
          -0.112,
          1.56,
          0.232,
          1.676,
          -0.492,
          -0.172,
          1.8,
          -1.972,
          0.988,
          -1.96,
          1.948,
          -0.908,
          -0.976,
          -0.184,
          0.356,
          -0.936,
          0.548,
          0.032,
          0.416,
          -0.32,
          1.1,
          -0.864,
          -0.336,
          -0.568,
          0.448,
          1.348,
          1.628,
          0.924,
          1.3,
          -0.264,
          -0.82,
          -0.264,
          -0.372,
          1.22,
          -1.668,
          -0.952,
          1.576,
          0.536,
          1.992,
          1.872,
          -0.4,
          -0.56,
          1.588,
          0.272,
          0.172,
          -1.68,
          -0.72
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
Code from [LayerTestBase.java:120](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L120) executed in 0.01 seconds: 
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
    	[ [ 0.032, -0.964 ], [ 1.564, 1.992 ], [ 0.9, -1.484 ] ],
    	[ [ -1.128, -1.232 ], [ -0.064, 0.648 ], [ 0.836, 1.844 ] ],
    	[ [ -0.668, 0.616 ], [ -0.724, -0.52 ], [ -0.208, 0.74 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ -1.021104, 0.26192, -1.949344 ] ]
    ]
```



### Batch Execution
Code from [LayerTestBase.java:138](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L138) executed in 0.08 seconds: 
```java
    BatchingTester batchingTester = getBatchingTester();
    return batchingTester==null?null:batchingTester.test(layer, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (210#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (50#)}
```



### Differential Validation
Code from [LayerTestBase.java:144](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.33 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ 0.032, -0.964 ], [ 1.564, 1.992 ], [ 0.9, -1.484 ] ],
    	[ [ -1.128, -1.232 ], [ -0.064, 0.648 ], [ 0.836, 1.844 ] ],
    	[ [ -0.668, 0.616 ], [ -0.724, -0.52 ], [ -0.208, 0.74 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.1980852429961627, negative=9, min=0.74, max=0.74, mean=0.12111111111111114, count=18.0, positive=9, stdDev=1.0439581137238585, zeros=0}
    Output: [
    	[ [ -1.021104, 0.26192, -1.949344 ] ]
    ]
    Outputs Statistics: {meanExponent=-0.09429095826610458, negative=2, min=-1.949344, max=-1.949344, mean=-0.9028426666666668, count=3.0, positive=1, stdDev=0.9066095913329445, zeros=0}
    Feedback for input 0
    Inputs Values: [
    	[ [ 0.032, -0.964 ], [ 1.564, 1.992 ], [ 0.9, -1.484 ] ],
    	[ [ -1.128, -1.232 ], [ -0.064, 0.648 ], [ 0.836, 1.844 ] ],
    	[ [ -0.668, 0.616 ], [ -0.724, -0.52 ], [ -0.208, 0.74 ] ]
    ]
    Value Statistics: {meanExponent=-0.1980852429961627, negative=9, min=0.74, max=0.74, mean=0.12111111111111114, count=18.0, positive=9, stdDev=1.0439581137238585, zeros=0}
    Implemented Feedback: [ [ 1.228, 0.232, -0.908 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], ... ]
    Implemented Statistics: {meanExponent=-0.13749030894452174, negative=2, min=0.0, max=0.0, mean=0.0625925925925926, count=54.0, positive=4, stdDev=0.3707467198985052, zeros=48}
    Measured Feedback: [ [ 1.2279999999997848, 0.23200000000000998, -0.9079999999994648 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], ... ]
    Measured Statistics: {meanExponent=-0.13749030894460382, negative=2, min=0.0, max=0.0, mean=0.06259259259260379, count=54.0, positive=4, stdDev=0.37074671989844876, zeros=48}
    Feedback Error: [ [ -2.1516122217235534E-13, 9.96425164601078E-15, 5.35238520171788E-13 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], ... ]
    Error Statistics: {meanExponent=-12.715826651573755, negative=2, min=0.0, max=0.0, mean=1.1198346776624207E-14, count=54.0, positive=4, stdDev=1.2606725491519978E-13, zeros=48}
    Learning Gradient for weight set 0
    Weights: [ 1.228, 1.504, -1.844, 1.356, -0.64, -0.196, 1.456, -0.112, ... ]
    Implemented Gradient: [ [ 0.032, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], ... ]
    Implemented Statistics: {meanExponent=-0.7553864938886315, negative=3, min=0.0, max=0.0, mean=-0.01725925925925926, count=162.0, positive=3, stdDev=0.1301163560739764, zeros=156}
    Measured Gradient: [ [ 0.031999999998699735, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], ... ]
    Measured Statistics: {meanExponent=-0.7553864938886735, negative=3, min=0.0, max=0.0, mean=-0.0172592592592587, count=162.0, positive=3, stdDev=0.13011635607397165, zeros=156}
    Gradient Error: [ [ -1.3002654508653677E-12, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], ... ]
    Error Statistics: {meanExponent=-12.287372659979313, negative=2, min=0.0, max=0.0, mean=5.569961501630222E-16, count=162.0, positive=4, stdDev=1.781165269818175E-13, zeros=156}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 3.1374e-14 +- 1.6375e-13 [0.0000e+00 - 1.3003e-12] (216#)
    relativeTol: 3.5562e-12 +- 6.4628e-12 [1.8197e-14 - 2.0317e-11] (12#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=3.1374e-14 +- 1.6375e-13 [0.0000e+00 - 1.3003e-12] (216#), relativeTol=3.5562e-12 +- 6.4628e-12 [1.8197e-14 - 2.0317e-11] (12#)}
```



### Performance
Code from [LayerTestBase.java:149](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L149) executed in 0.74 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 25.6302 +- 3.0505 [22.6416 - 38.7372]
    Learning performance: 20.8960 +- 2.4846 [18.4951 - 35.0838]
    
```

