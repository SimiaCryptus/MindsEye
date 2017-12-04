# SimpleConvolutionLayer
## SimpleConvolutionLayerTest
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
      "class": "com.simiacryptus.mindseye.layers.cudnn.f32.SimpleConvolutionLayer",
      "id": "370a9587-74a1-4959-b406-fa4500000402",
      "isFrozen": false,
      "name": "SimpleConvolutionLayer/370a9587-74a1-4959-b406-fa4500000402",
      "filter": {
        "dimensions": [
          1,
          1,
          1
        ],
        "data": [
          1.612
        ]
      },
      "strideX": 1,
      "strideY": 1,
      "simple": false
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
    	[ [ 1.656 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 2.6694719791412354 ] ]
    ]
```



### Reference Implementation
Code from [LayerTestBase.java:131](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L131) executed in 0.02 seconds: 
```java
    System.out.println(new GsonBuilder().setPrettyPrinting().create().toJson(referenceLayer.getJson()));
    getEquivalencyTester().test(referenceLayer, layer, inputPrototype);
```
Logging: 
```
    {
      "class": "com.simiacryptus.mindseye.layers.aparapi.ConvolutionLayer",
      "id": "370a9587-74a1-4959-b406-fa4500000403",
      "isFrozen": false,
      "name": "ConvolutionLayer/370a9587-74a1-4959-b406-fa4500000403",
      "filter": {
        "dimensions": [
          1,
          1,
          1
        ],
        "data": [
          1.612
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
    	[ [ 1.656 ] ]
    ]
    Error: [
    	[ [ -2.0858764493425497E-8 ] ]
    ]
    Accuracy:
    absoluteTol: 2.0859e-08 +- 0.0000e+00 [2.0859e-08 - 2.0859e-08] (1#)
    relativeTol: 3.9069e-09 +- 0.0000e+00 [3.9069e-09 - 3.9069e-09] (1#)
    
```

### Batch Execution
Code from [LayerTestBase.java:138](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L138) executed in 0.02 seconds: 
```java
    BatchingTester batchingTester = getBatchingTester();
    return batchingTester==null?null:batchingTester.test(layer, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (20#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (20#)}
```



### Differential Validation
Code from [LayerTestBase.java:144](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.01 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ 1.656 ] ]
    ]
    Inputs Statistics: {meanExponent=0.2190603324488613, negative=0, min=1.656, max=1.656, mean=1.656, count=1.0, positive=1, stdDev=0.0, zeros=0}
    Output: [
    	[ [ 2.6694719791412354 ] ]
    ]
    Outputs Statistics: {meanExponent=0.42642536652443574, negative=0, min=2.6694719791412354, max=2.6694719791412354, mean=2.6694719791412354, count=1.0, positive=1, stdDev=0.0, zeros=0}
    Feedback for input 0
    Inputs Values: [
    	[ [ 1.656 ] ]
    ]
    Value Statistics: {meanExponent=0.2190603324488613, negative=0, min=1.656, max=1.656, mean=1.656, count=1.0, positive=1, stdDev=0.0, zeros=0}
    Implemented Feedback: [ [ 1.6119999885559082 ] ]
    Implemented Statistics: {meanExponent=0.20736503438587958, negative=0, min=1.6119999885559082, max=1.6119999885559082, mean=1.6119999885559082, count=1.0, positive=1, stdDev=0.0, zeros=0}
    Measured Feedback: [ [ 1.611948013305664 ] ]
    Measured Statistics: {meanExponent=0.20735103132853155, negative=0, min=1.611948013305664, max=1.611948013305664, mean=1.611948013305664, count=1.0, positive=1, stdDev=0.0, zeros=0}
    Feedback Error: [ [ -5.1975250244140625E-5 ] ]
    Error Statistics: {meanExponent=-4.284203411002982, negative=1, min=-5.1975250244140625E-5, max=-5.1975250244140625E-5, mean=-5.1975250244140625E-5, count=1.0, positive=0, stdDev=0.0, zeros=0}
    Learning Gradient for weight set 0
    Weights: [ 1.612 ]
    Implemented Gradient: [ [ 1.656000018119812 ] ]
    Implemented Statistics: {meanExponent=0.2190603372008748, negative=0, min=1.656000018119812, max=1.656000018119812, mean=1.656000018119812, count=1.0, positive=1, stdDev=0.0, zeros=0}
    Measured Gradient: [ [ 1.6560554504394531 ] ]
    Measured Statistics: {meanExponent=0.21907487436715722, negative=0, min=1.6560554504394531, max=1.6560554504394531, mean=1.6560554504394531, count=1.0, positive=1, stdDev=0.0, zeros=0}
    Gradient Error: [ [ 5.543231964111328E-5 ] ]
    Error Statistics: {meanExponent=-4.2562369473816135, negative=0, min=5.543231964111328E-5, max=5.543231964111328E-5, mean=5.543231964111328E-5, count=1.0, positive=1, stdDev=0.0, zeros=0}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 5.3704e-05 +- 1.7285e-06 [5.1975e-05 - 5.5432e-05] (2#)
    relativeTol: 1.6429e-05 +- 3.0746e-07 [1.6122e-05 - 1.6737e-05] (2#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=5.3704e-05 +- 1.7285e-06 [5.1975e-05 - 5.5432e-05] (2#), relativeTol=1.6429e-05 +- 3.0746e-07 [1.6122e-05 - 1.6737e-05] (2#)}
```



### Performance
Code from [LayerTestBase.java:149](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L149) executed in 0.15 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 5.4537 +- 0.7293 [4.9159 - 11.1028]
    Learning performance: 3.9625 +- 0.4903 [3.5565 - 6.4776]
    
```

