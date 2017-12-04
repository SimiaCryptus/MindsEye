# SimpleConvolutionLayer
## SimpleConvolutionLayerTest
### Json Serialization
Code from [LayerTestBase.java:83](../../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L83) executed in 0.00 seconds: 
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
      "id": "a864e734-2f23-44db-97c1-504000000402",
      "isFrozen": false,
      "name": "SimpleConvolutionLayer/a864e734-2f23-44db-97c1-504000000402",
      "filter": {
        "dimensions": [
          1,
          1,
          1
        ],
        "data": [
          0.308
        ]
      },
      "strideX": 1,
      "strideY": 1,
      "simple": false
    }
```



### Example Input/Output Pair
Code from [LayerTestBase.java:120](../../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L120) executed in 0.00 seconds: 
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
    	[ [ 0.276 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 0.08500799536705017 ] ]
    ]
```



### Reference Implementation
Code from [LayerTestBase.java:131](../../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L131) executed in 0.02 seconds: 
```java
    System.out.println(new GsonBuilder().setPrettyPrinting().create().toJson(referenceLayer.getJson()));
    getEquivalencyTester().test(referenceLayer, layer, inputPrototype);
```
Logging: 
```
    {
      "class": "com.simiacryptus.mindseye.layers.aparapi.ConvolutionLayer",
      "id": "a864e734-2f23-44db-97c1-504000000403",
      "isFrozen": false,
      "name": "ConvolutionLayer/a864e734-2f23-44db-97c1-504000000403",
      "filter": {
        "dimensions": [
          1,
          1,
          1
        ],
        "data": [
          0.308
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
    	[ [ 0.276 ] ]
    ]
    Error: [
    	[ [ -4.6329498293307125E-9 ] ]
    ]
    Accuracy:
    absoluteTol: 4.6329e-09 +- 0.0000e+00 [4.6329e-09 - 4.6329e-09] (1#)
    relativeTol: 2.7250e-08 +- 0.0000e+00 [2.7250e-08 - 2.7250e-08] (1#)
    
```

### Batch Execution
Code from [LayerTestBase.java:138](../../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L138) executed in 0.02 seconds: 
```java
    BatchingTester batchingTester = getBatchingTester();
    return batchingTester==null?null:batchingTester.test(layer, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (20#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (20#)}
```



### Differential Validation
Code from [LayerTestBase.java:144](../../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.01 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ 0.276 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.5590909179347823, negative=0, min=0.276, max=0.276, mean=0.276, count=1.0, positive=1, stdDev=0.0, zeros=0}
    Output: [
    	[ [ 0.08500799536705017 ] ]
    ]
    Outputs Statistics: {meanExponent=-1.0705402251034586, negative=0, min=0.08500799536705017, max=0.08500799536705017, mean=0.08500799536705017, count=1.0, positive=1, stdDev=0.0, zeros=0}
    Feedback for input 0
    Inputs Values: [
    	[ [ 0.276 ] ]
    ]
    Value Statistics: {meanExponent=-0.5590909179347823, negative=0, min=0.276, max=0.276, mean=0.276, count=1.0, positive=1, stdDev=0.0, zeros=0}
    Implemented Feedback: [ [ 0.30799999833106995 ] ]
    Implemented Statistics: {meanExponent=-0.5114492858528256, negative=0, min=0.30799999833106995, max=0.30799999833106995, mean=0.30799999833106995, count=1.0, positive=1, stdDev=0.0, zeros=0}
    Measured Feedback: [ [ 0.30800700187683105 ] ]
    Measured Statistics: {meanExponent=-0.5114394106362228, negative=0, min=0.30800700187683105, max=0.30800700187683105, mean=0.30800700187683105, count=1.0, positive=1, stdDev=0.0, zeros=0}
    Feedback Error: [ [ 7.0035457611083984E-6 ] ]
    Error Statistics: {meanExponent=-5.154682029327794, negative=0, min=7.0035457611083984E-6, max=7.0035457611083984E-6, mean=7.0035457611083984E-6, count=1.0, positive=1, stdDev=0.0, zeros=0}
    Learning Gradient for weight set 0
    Weights: [ 0.308 ]
    Implemented Gradient: [ [ 0.2759999930858612 ] ]
    Implemented Statistics: {meanExponent=-0.5590909288143923, negative=0, min=0.2759999930858612, max=0.2759999930858612, mean=0.2759999930858612, count=1.0, positive=1, stdDev=0.0, zeros=0}
    Measured Gradient: [ [ 0.2759993076324463 ] ]
    Measured Statistics: {meanExponent=-0.5590920073977721, negative=0, min=0.2759993076324463, max=0.2759993076324463, mean=0.2759993076324463, count=1.0, positive=1, stdDev=0.0, zeros=0}
    Gradient Error: [ [ -6.854534149169922E-7 ] ]
    Error Statistics: {meanExponent=-6.164022055581937, negative=1, min=-6.854534149169922E-7, max=-6.854534149169922E-7, mean=-6.854534149169922E-7, count=1.0, positive=0, stdDev=0.0, zeros=0}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 3.8445e-06 +- 3.1590e-06 [6.8545e-07 - 7.0035e-06] (2#)
    relativeTol: 6.3055e-06 +- 5.0637e-06 [1.2418e-06 - 1.1369e-05] (2#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=3.8445e-06 +- 3.1590e-06 [6.8545e-07 - 7.0035e-06] (2#), relativeTol=6.3055e-06 +- 5.0637e-06 [1.2418e-06 - 1.1369e-05] (2#)}
```



### Performance
Code from [LayerTestBase.java:149](../../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L149) executed in 0.16 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 5.8527 +- 1.3981 [4.9985 - 14.8189]
    Learning performance: 4.1277 +- 0.5388 [3.3998 - 7.6802]
    
```

