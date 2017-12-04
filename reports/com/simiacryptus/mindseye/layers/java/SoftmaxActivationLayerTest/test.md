# SoftmaxActivationLayer
## SoftmaxActivationLayerTest
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
      "class": "com.simiacryptus.mindseye.layers.java.SoftmaxActivationLayer",
      "id": "370a9587-74a1-4959-b406-fa4500002c94",
      "isFrozen": false,
      "name": "SoftmaxActivationLayer/370a9587-74a1-4959-b406-fa4500002c94"
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
    [[ -0.856, 1.376, -0.312, -0.136 ]]
    --------------------
    Output: 
    [ 0.07094310638120944, 0.6610822317879198, 0.1222267880150717, 0.14574787381579898 ]
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
    Inputs: [ -0.856, 1.376, -0.312, -0.136 ]
    Inputs Statistics: {meanExponent=-0.3253035747586735, negative=3, min=-0.136, max=-0.136, mean=0.017999999999999974, count=4.0, positive=1, stdDev=0.8277463379562605, zeros=0}
    Output: [ 0.07094310638120944, 0.6610822317879198, 0.1222267880150717, 0.14574787381579898 ]
    Outputs Statistics: {meanExponent=-0.769516421839904, negative=0, min=0.14574787381579898, max=0.14574787381579898, mean=0.25, count=4.0, positive=4, stdDev=0.2388746909778325, zeros=0}
    Feedback for input 0
    Inputs Values: [ -0.856, 1.376, -0.312, -0.136 ]
    Value Statistics: {meanExponent=-0.3253035747586735, negative=3, min=-0.136, max=-0.136, mean=0.017999999999999974, count=4.0, positive=1, stdDev=0.8277463379562605, zeros=0}
    Implemented Feedback: [ [ 0.06591018203819385, -0.04689922709645777, -0.008671148024786768, -0.010339806916949319 ], [ -0.04689922709645777, 0.2240525146022229, -0.08080195780527258, -0.09635132970049252 ], [ -0.008671148024786768, -0.08080195780527258, 0.10728740030659042, -0.01781429447653108 ], [ -0.010339806916949319, -0.09635132970049252, -0.01781429447653108, 0.12450543109397293 ] ]
    Implemented Statistics: {meanExponent=-1.3858346732636926, negative=12, min=0.12450543109397293, max=0.12450543109397293, mean=8.673617379884035E-19, count=16.0, positive=4, stdDev=0.08608822301966759, zeros=0}
    Measured Feedback: [ [ 0.06591301002639716, -0.04690123938777546, -0.008671520075193007, -0.010340250563845022 ], [ -0.04689847160635474, 0.22404890538618716, -0.08080065618309984, -0.09634977759603869 ], [ -0.008671475602572976, -0.08080501033402854, 0.10729145340074298, -0.017814967463447573 ], [ -0.010340173211137449, -0.09635474300773161, -0.01781492555932851, 0.12450984177792002 ] ]
    Measured Statistics: {meanExponent=-1.3858238160040095, negative=12, min=0.12450984177792002, max=0.12450984177792002, mean=4.336808689942018E-14, count=16.0, positive=4, stdDev=0.08608878871996002, zeros=0}
    Feedback Error: [ [ 2.827988203307763E-6, -2.0122913176948143E-6, -3.7205040623843055E-7, -4.4364689570294824E-7 ], [ 7.554901030257244E-7, -3.6092160357326097E-6, 1.301622172739103E-6, 1.552104453836356E-6 ], [ -3.275777862073864E-7, -3.052528755959072E-6, 4.053094152550729E-6, -6.729869164931457E-7 ], [ -3.6629418813036163E-7, -3.4133072390890584E-6, -6.310827974295719E-7, 4.410683947089766E-6 ] ]
    Error Statistics: {meanExponent=-5.894298723146299, negative=10, min=4.410683947089766E-6, max=4.410683947089766E-6, mean=4.336700269724769E-14, count=16.0, positive=6, stdDev=2.3455475191426386E-6, zeros=0}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 1.8626e-06 +- 1.4256e-06 [3.2758e-07 - 4.4107e-06] (16#)
    relativeTol: 1.6527e-05 +- 5.0752e-06 [8.0545e-06 - 2.1453e-05] (16#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=1.8626e-06 +- 1.4256e-06 [3.2758e-07 - 4.4107e-06] (16#), relativeTol=1.6527e-05 +- 5.0752e-06 [8.0545e-06 - 2.1453e-05] (16#)}
```



### Performance
Code from [LayerTestBase.java:149](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L149) executed in 0.01 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 0.2490 +- 0.0891 [0.1881 - 0.8150]
    Learning performance: 0.0014 +- 0.0014 [0.0000 - 0.0029]
    
```

