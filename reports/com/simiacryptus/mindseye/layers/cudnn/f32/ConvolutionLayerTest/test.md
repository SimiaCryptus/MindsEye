# ConvolutionLayer
## ConvolutionLayerTest
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
      "class": "com.simiacryptus.mindseye.layers.cudnn.f32.ConvolutionLayer",
      "id": "a864e734-2f23-44db-97c1-504000000036",
      "isFrozen": false,
      "name": "ConvolutionLayer/a864e734-2f23-44db-97c1-504000000036",
      "filter": {
        "dimensions": [
          1,
          1,
          4
        ],
        "data": [
          0.988,
          1.044,
          1.932,
          -0.456
        ]
      },
      "strideX": 1,
      "strideY": 1
    }
```



### Example Input/Output Pair
Code from [LayerTestBase.java:120](../../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L120) executed in 0.01 seconds: 
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
    	[ [ -0.752, 0.6 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 0.4162241220474243, -1.0586880445480347 ] ]
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
      "id": "a864e734-2f23-44db-97c1-50400000003d",
      "isFrozen": false,
      "name": "ConvolutionLayer/a864e734-2f23-44db-97c1-50400000003d",
      "filter": {
        "dimensions": [
          1,
          1,
          4
        ],
        "data": [
          0.988,
          1.044,
          1.932,
          -0.456
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
    	[ [ -0.752, 0.6 ] ]
    ]
    Error: [
    	[ [ 1.2204742438903793E-7, -4.454803459452705E-8 ] ]
    ]
    Accuracy:
    absoluteTol: 8.3298e-08 +- 3.8750e-08 [4.4548e-08 - 1.2205e-07] (2#)
    relativeTol: 8.3826e-08 +- 6.2787e-08 [2.1039e-08 - 1.4661e-07] (2#)
    
```

### Batch Execution
Code from [LayerTestBase.java:138](../../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L138) executed in 0.03 seconds: 
```java
    BatchingTester batchingTester = getBatchingTester();
    return batchingTester==null?null:batchingTester.test(layer, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (40#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (40#)}
```



### Differential Validation
Code from [LayerTestBase.java:144](../../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.03 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ -0.752, 0.6 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.17281545451235708, negative=1, min=0.6, max=0.6, mean=-0.07600000000000001, count=2.0, positive=1, stdDev=0.676, zeros=0}
    Output: [
    	[ [ 0.4162241220474243, -1.0586880445480347 ] ]
    ]
    Outputs Statistics: {meanExponent=-0.17795237267086952, negative=1, min=-1.0586880445480347, max=-1.0586880445480347, mean=-0.3212319612503052, count=2.0, positive=1, stdDev=0.7374560832977295, zeros=0}
    Feedback for input 0
    Inputs Values: [
    	[ [ -0.752, 0.6 ] ]
    ]
    Value Statistics: {meanExponent=-0.17281545451235708, negative=1, min=0.6, max=0.6, mean=-0.07600000000000001, count=2.0, positive=1, stdDev=0.676, zeros=0}
    Implemented Feedback: [ [ 0.9879999756813049, 1.0440000295639038 ], [ 1.9320000410079956, -0.4560000002384186 ] ]
    Implemented Statistics: {meanExponent=-0.010392645237098888, negative=1, min=-0.4560000002384186, max=-0.4560000002384186, mean=0.8770000115036964, count=4.0, positive=3, stdDev=0.8558802620165527, zeros=0}
    Measured Feedback: [ [ 0.9876489639282227, 1.0442733764648438 ], [ 1.9311904907226562, -0.4553794860839844 ] ]
    Measured Statistics: {meanExponent=-0.010596151177123508, negative=1, min=-0.4553794860839844, max=-0.4553794860839844, mean=0.8769333362579346, count=4.0, positive=3, stdDev=0.8553911741980417, zeros=0}
    Feedback Error: [ [ -3.510117530822754E-4, 2.733469009399414E-4 ], [ -8.095502853393555E-4, 6.205141544342041E-4 ] ]
    Error Statistics: {meanExponent=-3.3292421661007685, negative=2, min=6.205141544342041E-4, max=6.205141544342041E-4, mean=-6.667524576187134E-5, count=4.0, positive=2, stdDev=5.5239363361012E-4, zeros=0}
    Learning Gradient for weight set 0
    Weights: [ 0.988, 1.044, 1.932, -0.456 ]
    Implemented Gradient: [ [ -0.7519999742507935, 0.0 ], [ 0.0, -0.7519999742507935 ], [ 0.6000000238418579, 0.0 ], [ 0.0, 0.6000000238418579 ] ]
    Implemented Statistics: {meanExponent=-0.17281545331903259, negative=2, min=0.6000000238418579, max=0.6000000238418579, mean=-0.03799998760223389, count=8.0, positive=2, stdDev=0.4795122505349432, zeros=4}
    Measured Gradient: [ [ -0.7522106170654297, 0.0 ], [ 0.0, -0.7522106170654297 ], [ 0.6002187728881836, 0.0 ], [ 0.0, 0.6008148193359375 ] ]
    Measured Statistics: {meanExponent=-0.17256771781570981, negative=2, min=0.6008148193359375, max=0.6008148193359375, mean=-0.037923455238342285, count=8.0, positive=2, stdDev=0.4797625896804669, zeros=4}
    Gradient Error: [ [ -2.1064281463623047E-4, 0.0 ], [ 0.0, -2.1064281463623047E-4 ], [ 2.187490463256836E-4, 0.0 ], [ 0.0, 8.147954940795898E-4 ] ]
    Error Statistics: {meanExponent=-3.525477978645198, negative=2, min=8.147954940795898E-4, max=8.147954940795898E-4, mean=7.653236389160156E-5, count=8.0, positive=2, stdDev=3.069254818709774E-4, zeros=4}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 2.9244e-04 +- 2.9049e-04 [0.0000e+00 - 8.1480e-04] (12#)
    relativeTol: 2.9248e-04 +- 2.2491e-04 [1.3090e-04 - 6.8085e-04] (8#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=2.9244e-04 +- 2.9049e-04 [0.0000e+00 - 8.1480e-04] (12#), relativeTol=2.9248e-04 +- 2.2491e-04 [1.3090e-04 - 6.8085e-04] (8#)}
```



### Performance
Code from [LayerTestBase.java:149](../../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L149) executed in 0.30 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 10.6198 +- 1.4697 [8.8144 - 15.6995]
    Learning performance: 7.5119 +- 1.2977 [5.9789 - 14.1378]
    
```

