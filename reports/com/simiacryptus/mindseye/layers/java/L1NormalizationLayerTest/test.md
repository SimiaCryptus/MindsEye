# L1NormalizationLayer
## L1NormalizationLayerTest
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
      "class": "com.simiacryptus.mindseye.layers.java.L1NormalizationLayer",
      "id": "370a9587-74a1-4959-b406-fa4500002c06",
      "isFrozen": false,
      "name": "L1NormalizationLayer/370a9587-74a1-4959-b406-fa4500002c06"
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
    [[ 0.68, -0.224, 1.216, 0.136 ]]
    --------------------
    Output: 
    [ 0.3761061946902655, -0.12389380530973451, 0.6725663716814159, 0.0752212389380531 ]
```



### Batch Execution
Code from [LayerTestBase.java:138](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L138) executed in 0.00 seconds: 
```java
    BatchingTester batchingTester = getBatchingTester();
    return batchingTester==null?null:batchingTester.test(layer, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (80#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (40#)}
```



### Differential Validation
Code from [LayerTestBase.java:144](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.00 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [ 0.68, -0.224, 1.216, 0.136 ]
    Inputs Statistics: {meanExponent=-0.3996926464131668, negative=1, min=0.136, max=0.136, mean=0.45200000000000007, count=4.0, positive=3, stdDev=0.5460109889004066, zeros=0}
    Output: [ 0.3761061946902655, -0.12389380530973451, 0.6725663716814159, 0.0752212389380531 ]
    Outputs Statistics: {meanExponent=-0.6568910725525112, negative=1, min=0.0752212389380531, max=0.0752212389380531, mean=0.25, count=4.0, positive=3, stdDev=0.30199722837411874, zeros=0}
    Feedback for input 0
    Inputs Values: [ 0.68, -0.224, 1.216, 0.136 ]
    Value Statistics: {meanExponent=-0.3996926464131668, negative=1, min=0.136, max=0.136, mean=0.45200000000000007, count=4.0, positive=3, stdDev=0.5460109889004066, zeros=0}
    Implemented Feedback: [ [ 0.34507400736157884, 0.06852533479520716, -0.37199467460255314, -0.041604667554232914 ], [ -0.2080233377711646, 0.6216226799279505, -0.37199467460255314, -0.041604667554232914 ], [ -0.2080233377711646, 0.06852533479520716, 0.1811026705301903, -0.041604667554232914 ], [ -0.2080233377711646, 0.06852533479520716, -0.37199467460255314, 0.5114926775785105 ] ]
    Implemented Statistics: {meanExponent=-0.7919294131457439, negative=9, min=0.5114926775785105, max=0.5114926775785105, mean=-2.0816681711721685E-17, count=16.0, positive=7, stdDev=0.2919926169690141, zeros=0}
    Measured Feedback: [ [ 0.34505492246539315, 0.0685215448867349, -0.3719741008134658, -0.04160236653838467 ], [ -0.20801183269192336, 0.6215883000440514, -0.3719741008134658, -0.04160236653838467 ], [ -0.20801183269192336, 0.0685215448867349, 0.1810926543444058, -0.04160236653838467 ], [ -0.20801183269192336, 0.0685215448867349, -0.3719741008134658, 0.5114643886189318 ] ]
    Measured Statistics: {meanExponent=-0.7919534331939423, negative=9, min=0.5114643886189318, max=0.5114643886189318, mean=1.0408340855860843E-13, count=16.0, positive=7, stdDev=0.2919764678280369, zeros=0}
    Feedback Error: [ [ -1.9084896185683764E-5, -3.7899084722597953E-6, 2.0573789087319216E-5, 2.301015848242549E-6 ], [ 1.150507924124744E-5, -3.4379883899093855E-5, 2.0573789087319216E-5, 2.301015848242549E-6 ], [ 1.150507924124744E-5, -3.7899084722597953E-6, -1.0016185784500475E-5, 2.301015848242549E-6 ], [ 1.150507924124744E-5, -3.7899084722597953E-6, 2.0573789087319216E-5, -2.8288959578626205E-5 ] ]
    Error Statistics: {meanExponent=-5.049151859896785, negative=7, min=-2.8288959578626205E-5, max=-2.8288959578626205E-5, mean=1.0410899572987908E-13, count=16.0, positive=9, stdDev=1.614914097715576E-5, zeros=0}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 1.2892e-05 +- 9.7252e-06 [2.3010e-06 - 3.4380e-05] (16#)
    relativeTol: 2.7654e-05 +- 6.4311e-13 [2.7654e-05 - 2.7654e-05] (16#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=1.2892e-05 +- 9.7252e-06 [2.3010e-06 - 3.4380e-05] (16#), relativeTol=2.7654e-05 +- 6.4311e-13 [2.7654e-05 - 2.7654e-05] (16#)}
```



### Performance
Code from [LayerTestBase.java:149](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L149) executed in 0.00 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 0.1602 +- 0.0356 [0.1140 - 0.3762]
    Learning performance: 0.0030 +- 0.0020 [0.0000 - 0.0142]
    
```

