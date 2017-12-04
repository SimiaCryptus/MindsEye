# FullyConnectedLayer
## FullyConnectedLayerTest
### Json Serialization
Code from [LayerTestBase.java:83](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L83) executed in 0.00 seconds: 
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
      "class": "com.simiacryptus.mindseye.layers.java.FullyConnectedLayer",
      "id": "370a9587-74a1-4959-b406-fa4500002bb0",
      "isFrozen": false,
      "name": "FullyConnectedLayer/370a9587-74a1-4959-b406-fa4500002bb0",
      "outputDims": [
        3
      ],
      "inputDims": [
        3
      ],
      "weights": {
        "dimensions": [
          3,
          3
        ],
        "data": [
          0.3561117587737334,
          0.5669290558917098,
          -0.8442368057845226,
          0.12500369115587426,
          -0.8451387064131721,
          -0.8259117912038715,
          0.13611208229729108,
          -0.7280327443857864,
          0.12975261469795427
        ]
      }
    }
```



### Example Input/Output Pair
Code from [LayerTestBase.java:120](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L120) executed in 0.00 seconds: 
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
    [[ -0.472, -1.072, 1.044 ]]
    --------------------
    Output: 
    [ -0.1599876931419275, -0.1216680062447274, 1.4193189422455093 ]
```



### Batch Execution
Code from [LayerTestBase.java:138](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L138) executed in 0.00 seconds: 
```java
    BatchingTester batchingTester = getBatchingTester();
    return batchingTester==null?null:batchingTester.test(layer, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (60#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (60#)}
```



### Differential Validation
Code from [LayerTestBase.java:144](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.00 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [ -0.472, -1.072, 1.044 ]
    Inputs Statistics: {meanExponent=-0.09238757244763922, negative=2, min=1.044, max=1.044, mean=-0.16666666666666666, count=3.0, positive=1, stdDev=0.890425116946332, zeros=0}
    Output: [ -0.1599876931419275, -0.1216680062447274, 1.4193189422455093 ]
    Outputs Statistics: {meanExponent=-0.5195523444445939, negative=2, min=1.4193189422455093, max=1.4193189422455093, mean=0.37922108095295143, count=3.0, positive=1, stdDev=0.7356266128954232, zeros=0}
    Feedback for input 0
    Inputs Values: [ -0.472, -1.072, 1.044 ]
    Value Statistics: {meanExponent=-0.09238757244763922, negative=2, min=1.044, max=1.044, mean=-0.16666666666666666, count=3.0, positive=1, stdDev=0.890425116946332, zeros=0}
    Implemented Feedback: [ [ 0.3561117587737334, 0.5669290558917098, -0.8442368057845226 ], [ 0.12500369115587426, -0.8451387064131721, -0.8259117912038715 ], [ 0.13611208229729108, -0.7280327443857864, 0.12975261469795427 ] ]
    Implemented Statistics: {meanExponent=-0.4131636092150222, negative=4, min=0.12975261469795427, max=0.12975261469795427, mean=-0.21437898277453218, count=9.0, positive=5, stdDev=0.5502266551854915, zeros=0}
    Measured Feedback: [ [ 0.3561117587741469, 0.5669290558918405, -0.8442368057837157 ], [ 0.1250036911559782, -0.8451387064134508, -0.8259117912046143 ], [ 0.13611208229735716, -0.7280327443859314, 0.12975261469705401 ] ]
    Measured Statistics: {meanExponent=-0.4131636092152034, negative=4, min=0.12975261469705401, max=0.12975261469705401, mean=-0.21437898277459283, count=9.0, positive=5, stdDev=0.5502266551855485, zeros=0}
    Feedback Error: [ [ 4.1350256552163955E-13, 1.3067324999838092E-13, 8.069100942975638E-13 ], [ 1.0394463068053028E-13, -2.786659791809143E-13, -7.428502257766922E-13 ], [ 6.608602554081244E-14, -1.4499512701604544E-13, -9.002520950929238E-13 ] ]
    Error Statistics: {meanExponent=-12.56576616640079, negative=4, min=-9.002520950929238E-13, max=-9.002520950929238E-13, mean=-6.06274290030721E-14, count=9.0, positive=5, stdDev=5.035665089467934E-13, zeros=0}
    Learning Gradient for weight set 0
    Weights: [ 0.3561117587737334, 0.5669290558917098, -0.8442368057845226, 0.12500369115587426, -0.8451387064131721, -0.8259117912038715, 0.13611208229729108, -0.7280327443857864, 0.12975261469795427 ]
    Implemented Gradient: [ [ -0.472, 0.0, 0.0 ], [ 0.0, -0.472, 0.0 ], [ 0.0, 0.0, -0.472 ], [ -1.072, 0.0, 0.0 ], [ 0.0, -1.072, 0.0 ], [ 0.0, 0.0, -1.072 ], [ 1.044, 0.0, 0.0 ], [ 0.0, 1.044, 0.0 ], [ 0.0, 0.0, 1.044 ] ]
    Implemented Statistics: {meanExponent=-0.09238757244763922, negative=6, min=1.044, max=1.044, mean=-0.05555555555555555, count=27.0, positive=3, stdDev=0.5200562172840572, zeros=18}
    Measured Gradient: [ [ -0.4719999999996949, 0.0, 0.0 ], [ 0.0, -0.47200000000025, 0.0 ], [ 0.0, 0.0, -0.47200000000025 ], [ -1.0719999999997398, 0.0, 0.0 ], [ 0.0, -1.0719999999997398, 0.0 ], [ 0.0, 0.0, -1.0719999999997398 ], [ 1.044000000000045, 0.0, 0.0 ], [ 0.0, 1.044000000000045, 0.0 ], [ 0.0, 0.0, 1.0439999999989347 ] ]
    Measured Statistics: {meanExponent=-0.09238757244769953, negative=6, min=1.0439999999989347, max=1.0439999999989347, mean=-0.05555555555557, count=27.0, positive=3, stdDev=0.52005621728393, zeros=18}
    Gradient Error: [ [ 3.05089287166993E-13, 0.0, 0.0 ], [ 0.0, -2.5002222514558525E-13, 0.0 ], [ 0.0, 0.0, -2.5002222514558525E-13 ], [ 2.602362769721367E-13, 0.0, 0.0 ], [ 0.0, 2.602362769721367E-13, 0.0 ], [ 0.0, 0.0, 2.602362769721367E-13 ], [ 4.4853010194856324E-14, 0.0, 0.0 ], [ 0.0, 4.4853010194856324E-14, 0.0 ], [ 0.0, 0.0, -1.0653700144303002E-12 ] ]
    Error Statistics: {meanExponent=-12.682492071046415, negative=3, min=-1.0653700144303002E-12, max=-1.0653700144303002E-12, mean=-1.4441123194383518E-14, count=27.0, positive=6, stdDev=2.399596083659438E-13, zeros=18}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 1.7580e-13 +- 2.7704e-13 [0.0000e+00 - 1.0654e-12] (36#)
    relativeTol: 4.3254e-13 +- 7.5525e-13 [2.1481e-14 - 3.4691e-12] (18#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=1.7580e-13 +- 2.7704e-13 [0.0000e+00 - 1.0654e-12] (36#), relativeTol=4.3254e-13 +- 7.5525e-13 [2.1481e-14 - 3.4691e-12] (18#)}
```



### Performance
Code from [LayerTestBase.java:149](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L149) executed in 0.01 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 0.2067 +- 0.0517 [0.1567 - 0.5016]
    Learning performance: 0.4253 +- 0.2508 [0.2964 - 2.1972]
    
```

