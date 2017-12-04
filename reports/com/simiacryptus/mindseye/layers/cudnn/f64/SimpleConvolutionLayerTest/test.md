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
      "class": "com.simiacryptus.mindseye.layers.cudnn.f64.SimpleConvolutionLayer",
      "id": "370a9587-74a1-4959-b406-fa4500002b44",
      "isFrozen": false,
      "name": "SimpleConvolutionLayer/370a9587-74a1-4959-b406-fa4500002b44",
      "filter": {
        "dimensions": [
          1,
          1,
          1
        ],
        "data": [
          0.1
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
    	[ [ 0.644 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 0.0644 ] ]
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
      "id": "370a9587-74a1-4959-b406-fa4500002b45",
      "isFrozen": false,
      "name": "ConvolutionLayer/370a9587-74a1-4959-b406-fa4500002b45",
      "filter": {
        "dimensions": [
          1,
          1,
          1
        ],
        "data": [
          0.1
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
    	[ [ 0.644 ] ]
    ]
    Error: [
    	[ [ 0.0 ] ]
    ]
    Accuracy:
    absoluteTol: 0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (1#)
    relativeTol: 0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (1#)
    
```

### Batch Execution
Code from [LayerTestBase.java:138](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L138) executed in 0.01 seconds: 
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
    	[ [ 0.644 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.1911141326401879, negative=0, min=0.644, max=0.644, mean=0.644, count=1.0, positive=1, stdDev=0.0, zeros=0}
    Output: [
    	[ [ 0.0644 ] ]
    ]
    Outputs Statistics: {meanExponent=-1.191114132640188, negative=0, min=0.0644, max=0.0644, mean=0.0644, count=1.0, positive=1, stdDev=0.0, zeros=0}
    Feedback for input 0
    Inputs Values: [
    	[ [ 0.644 ] ]
    ]
    Value Statistics: {meanExponent=-0.1911141326401879, negative=0, min=0.644, max=0.644, mean=0.644, count=1.0, positive=1, stdDev=0.0, zeros=0}
    Implemented Feedback: [ [ 0.1 ] ]
    Implemented Statistics: {meanExponent=-1.0, negative=0, min=0.1, max=0.1, mean=0.1, count=1.0, positive=1, stdDev=0.0, zeros=0}
    Measured Feedback: [ [ 0.10000000000010001 ] ]
    Measured Statistics: {meanExponent=-0.9999999999995657, negative=0, min=0.10000000000010001, max=0.10000000000010001, mean=0.10000000000010001, count=1.0, positive=1, stdDev=0.0, zeros=0}
    Feedback Error: [ [ 1.0000333894311098E-13 ] ]
    Error Statistics: {meanExponent=-12.999985499396397, negative=0, min=1.0000333894311098E-13, max=1.0000333894311098E-13, mean=1.0000333894311098E-13, count=1.0, positive=1, stdDev=0.0, zeros=0}
    Learning Gradient for weight set 0
    Weights: [ 0.1 ]
    Implemented Gradient: [ [ 0.644 ] ]
    Implemented Statistics: {meanExponent=-0.1911141326401879, negative=0, min=0.644, max=0.644, mean=0.644, count=1.0, positive=1, stdDev=0.0, zeros=0}
    Measured Gradient: [ [ 0.6440000000000612 ] ]
    Measured Statistics: {meanExponent=-0.19111413264014662, negative=0, min=0.6440000000000612, max=0.6440000000000612, mean=0.6440000000000612, count=1.0, positive=1, stdDev=0.0, zeros=0}
    Gradient Error: [ [ 6.117328865684613E-14 ] ]
    Error Statistics: {meanExponent=-13.213438171339218, negative=0, min=6.117328865684613E-14, max=6.117328865684613E-14, mean=6.117328865684613E-14, count=1.0, positive=1, stdDev=0.0, zeros=0}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 8.0588e-14 +- 1.9415e-14 [6.1173e-14 - 1.0000e-13] (2#)
    relativeTol: 2.7376e-13 +- 2.2626e-13 [4.7495e-14 - 5.0002e-13] (2#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=8.0588e-14 +- 1.9415e-14 [6.1173e-14 - 1.0000e-13] (2#), relativeTol=2.7376e-13 +- 2.2626e-13 [4.7495e-14 - 5.0002e-13] (2#)}
```



### Performance
Code from [LayerTestBase.java:149](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L149) executed in 0.15 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 5.8960 +- 0.8911 [5.0327 - 10.7551]
    Learning performance: 3.5986 +- 0.4393 [3.0749 - 5.3291]
    
```

