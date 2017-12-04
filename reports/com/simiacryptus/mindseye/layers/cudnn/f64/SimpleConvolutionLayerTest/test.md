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
      "class": "com.simiacryptus.mindseye.layers.cudnn.f64.SimpleConvolutionLayer",
      "id": "a864e734-2f23-44db-97c1-504000002b44",
      "isFrozen": false,
      "name": "SimpleConvolutionLayer/a864e734-2f23-44db-97c1-504000002b44",
      "filter": {
        "dimensions": [
          1,
          1,
          1
        ],
        "data": [
          0.904
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
    	[ [ -0.284 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ -0.25673599999999996 ] ]
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
      "id": "a864e734-2f23-44db-97c1-504000002b45",
      "isFrozen": false,
      "name": "ConvolutionLayer/a864e734-2f23-44db-97c1-504000002b45",
      "filter": {
        "dimensions": [
          1,
          1,
          1
        ],
        "data": [
          0.904
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
    	[ [ -0.284 ] ]
    ]
    Error: [
    	[ [ 0.0 ] ]
    ]
    Accuracy:
    absoluteTol: 0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (1#)
    relativeTol: 0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (1#)
    
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
    	[ [ -0.284 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.5466816599529624, negative=1, min=-0.284, max=-0.284, mean=-0.284, count=1.0, positive=0, stdDev=0.0, zeros=0}
    Output: [
    	[ [ -0.25673599999999996 ] ]
    ]
    Outputs Statistics: {meanExponent=-0.5905132294775991, negative=1, min=-0.25673599999999996, max=-0.25673599999999996, mean=-0.25673599999999996, count=1.0, positive=0, stdDev=0.0, zeros=0}
    Feedback for input 0
    Inputs Values: [
    	[ [ -0.284 ] ]
    ]
    Value Statistics: {meanExponent=-0.5466816599529624, negative=1, min=-0.284, max=-0.284, mean=-0.284, count=1.0, positive=0, stdDev=0.0, zeros=0}
    Implemented Feedback: [ [ 0.904 ] ]
    Implemented Statistics: {meanExponent=-0.04383156952463668, negative=0, min=0.904, max=0.904, mean=0.904, count=1.0, positive=1, stdDev=0.0, zeros=0}
    Measured Feedback: [ [ 0.9039999999999049 ] ]
    Measured Statistics: {meanExponent=-0.04383156952468239, negative=0, min=0.9039999999999049, max=0.9039999999999049, mean=0.9039999999999049, count=1.0, positive=1, stdDev=0.0, zeros=0}
    Feedback Error: [ [ -9.514611321037592E-14 ] ]
    Error Statistics: {meanExponent=-13.021608948267804, negative=1, min=-9.514611321037592E-14, max=-9.514611321037592E-14, mean=-9.514611321037592E-14, count=1.0, positive=0, stdDev=0.0, zeros=0}
    Learning Gradient for weight set 0
    Weights: [ 0.904 ]
    Implemented Gradient: [ [ -0.284 ] ]
    Implemented Statistics: {meanExponent=-0.5466816599529624, negative=1, min=-0.284, max=-0.284, mean=-0.284, count=1.0, positive=0, stdDev=0.0, zeros=0}
    Measured Gradient: [ [ -0.28400000000039505 ] ]
    Measured Statistics: {meanExponent=-0.5466816599523582, negative=1, min=-0.28400000000039505, max=-0.28400000000039505, mean=-0.28400000000039505, count=1.0, positive=0, stdDev=0.0, zeros=0}
    Gradient Error: [ [ -3.9507286331286195E-13 ] ]
    Error Statistics: {meanExponent=-12.403322800028059, negative=1, min=-3.9507286331286195E-13, max=-3.9507286331286195E-13, mean=-3.9507286331286195E-13, count=1.0, positive=0, stdDev=0.0, zeros=0}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 2.4511e-13 +- 1.4996e-13 [9.5146e-14 - 3.9507e-13] (2#)
    relativeTol: 3.7409e-13 +- 3.2146e-13 [5.2625e-14 - 6.9555e-13] (2#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=2.4511e-13 +- 1.4996e-13 [9.5146e-14 - 3.9507e-13] (2#), relativeTol=3.7409e-13 +- 3.2146e-13 [5.2625e-14 - 6.9555e-13] (2#)}
```



### Performance
Code from [LayerTestBase.java:149](../../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L149) executed in 0.16 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 5.8769 +- 1.0483 [5.0099 - 10.7466]
    Learning performance: 3.7792 +- 0.4937 [3.2915 - 6.2923]
    
```

