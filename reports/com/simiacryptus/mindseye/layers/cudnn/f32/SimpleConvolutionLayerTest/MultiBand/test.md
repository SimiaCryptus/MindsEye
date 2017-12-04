# SimpleConvolutionLayer
## MultiBand
### Json Serialization
Code from [LayerTestBase.java:83](../../../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L83) executed in 0.00 seconds: 
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
      "id": "a864e734-2f23-44db-97c1-504000000442",
      "isFrozen": false,
      "name": "SimpleConvolutionLayer/a864e734-2f23-44db-97c1-504000000442",
      "filter": {
        "dimensions": [
          1,
          1,
          9
        ],
        "data": [
          -0.836,
          -1.168,
          -1.204,
          1.508,
          0.024,
          1.336,
          1.56,
          0.396,
          0.032
        ]
      },
      "strideX": 1,
      "strideY": 1,
      "simple": false
    }
```



### Example Input/Output Pair
Code from [LayerTestBase.java:120](../../../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L120) executed in 0.00 seconds: 
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
    	[ [ -1.444, 1.64, 0.352 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ -1.1321438550949097, -1.6679199934005737, -1.5919359922409058 ] ]
    ]
```



### Reference Implementation
Code from [LayerTestBase.java:131](../../../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L131) executed in 0.01 seconds: 
```java
    System.out.println(new GsonBuilder().setPrettyPrinting().create().toJson(referenceLayer.getJson()));
    getEquivalencyTester().test(referenceLayer, layer, inputPrototype);
```
Logging: 
```
    {
      "class": "com.simiacryptus.mindseye.layers.aparapi.ConvolutionLayer",
      "id": "a864e734-2f23-44db-97c1-504000000443",
      "isFrozen": false,
      "name": "ConvolutionLayer/a864e734-2f23-44db-97c1-504000000443",
      "filter": {
        "dimensions": [
          1,
          1,
          9
        ],
        "data": [
          -0.836,
          1.508,
          1.56,
          -1.168,
          0.024,
          0.396,
          -1.204,
          1.336,
          0.032
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
    	[ [ -1.444, 1.64, 0.352 ] ]
    ]
    Error: [
    	[ [ 1.449050901491944E-7, 6.5994263387381125E-9, 7.759094255987975E-9 ] ]
    ]
    Accuracy:
    absoluteTol: 5.3088e-08 +- 6.4926e-08 [6.5994e-09 - 1.4491e-07] (3#)
    relativeTol: 2.2804e-08 +- 2.9128e-08 [1.9783e-09 - 6.3996e-08] (3#)
    
```

### Batch Execution
Code from [LayerTestBase.java:138](../../../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L138) executed in 0.02 seconds: 
```java
    BatchingTester batchingTester = getBatchingTester();
    return batchingTester==null?null:batchingTester.test(layer, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (60#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (60#)}
```



### Differential Validation
Code from [LayerTestBase.java:144](../../../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.02 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ -1.444, 1.64, 0.352 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.026348765080183616, negative=1, min=0.352, max=0.352, mean=0.18266666666666664, count=3.0, positive=2, stdDev=1.2647185018370248, zeros=0}
    Output: [
    	[ [ -1.1321438550949097, -1.6679199934005737, -1.5919359922409058 ] ]
    ]
    Outputs Statistics: {meanExponent=0.15933414338843457, negative=3, min=-1.5919359922409058, max=-1.5919359922409058, mean=-1.4639999469121296, count=3.0, positive=0, stdDev=0.23669916401092772, zeros=0}
    Feedback for input 0
    Inputs Values: [
    	[ [ -1.444, 1.64, 0.352 ] ]
    ]
    Value Statistics: {meanExponent=-0.026348765080183616, negative=1, min=0.352, max=0.352, mean=0.18266666666666664, count=3.0, positive=2, stdDev=1.2647185018370248, zeros=0}
    Implemented Feedback: [ [ -0.8360000252723694, 1.5080000162124634, 1.559999942779541 ], [ -1.1679999828338623, 0.024000000208616257, 0.3959999978542328 ], [ -1.2039999961853027, 1.3359999656677246, 0.03200000151991844 ] ]
    Implemented Statistics: {meanExponent=-0.32770395376440115, negative=3, min=0.03200000151991844, max=0.03200000151991844, mean=0.18311110221677357, count=9.0, positive=6, stdDev=1.0467911119339428, zeros=0}
    Measured Feedback: [ [ -0.8360147476196289, 1.5079975128173828, 1.560211181640625 ], [ -1.1681318283081055, 0.02384185791015625, 0.3960132598876953 ], [ -1.2040138244628906, 1.3359785079956055, 0.032067298889160156 ] ]
    Measured Statistics: {meanExponent=-0.3279074493933781, negative=3, min=0.032067298889160156, max=0.032067298889160156, mean=0.18310546875, count=9.0, positive=6, stdDev=1.0468434380197895, zeros=0}
    Feedback Error: [ [ -1.4722347259521484E-5, -2.5033950805664062E-6, 2.1123886108398438E-4 ], [ -1.3184547424316406E-4, -1.5814229846000671E-4, 1.3262033462524414E-5 ], [ -1.3828277587890625E-5, -2.1457672119140625E-5, 6.729736924171448E-5 ] ]
    Error Statistics: {meanExponent=-4.485183061906451, negative=6, min=6.729736924171448E-5, max=6.729736924171448E-5, mean=-5.633466773562961E-6, count=9.0, positive=3, stdDev=1.0127406897988054E-4, zeros=0}
    Learning Gradient for weight set 0
    Weights: [ -0.836, -1.168, -1.204, 1.508, 0.024, 1.336, 1.56, 0.396, 0.032 ]
    Implemented Gradient: [ [ -1.444000005722046, 0.0, 0.0 ], [ 1.6399999856948853, 0.0, 0.0 ], [ 0.35199999809265137, 0.0, 0.0 ], [ 0.0, -1.444000005722046, 0.0 ], [ 0.0, 1.6399999856948853, 0.0 ], [ 0.0, 0.35199999809265137, 0.0 ], [ 0.0, 0.0, -1.444000005722046 ], [ 0.0, 0.0, 1.6399999856948853 ], [ 0.0, 0.0, 0.35199999809265137 ] ]
    Implemented Statistics: {meanExponent=-0.026348766553686783, negative=3, min=0.35199999809265137, max=0.35199999809265137, mean=0.06088888645172119, count=27.0, positive=6, stdDev=0.7352454510661912, zeros=18}
    Measured Gradient: [ [ -1.444101333618164, 0.0, 0.0 ], [ 1.6398429870605469, 0.0, 0.0 ], [ 0.35202503204345703, 0.0, 0.0 ], [ 0.0, -1.4438629150390625, 0.0 ], [ 0.0, 1.6398429870605469, 0.0 ], [ 0.0, 0.35202503204345703, 0.0 ], [ 0.0, 0.0, -1.444101333618164 ], [ 0.0, 0.0, 1.6399621963500977 ], [ 0.0, 0.0, 0.35202503204345703 ] ]
    Measured Statistics: {meanExponent=-0.026346632128799893, negative=3, min=0.35202503204345703, max=0.35202503204345703, mean=0.06087621053059896, count=27.0, positive=6, stdDev=0.7352235414445626, zeros=18}
    Gradient Error: [ [ -1.0132789611816406E-4, 0.0, 0.0 ], [ -1.569986343383789E-4, 0.0, 0.0 ], [ 2.5033950805664062E-5, 0.0, 0.0 ], [ 0.0, 1.3709068298339844E-4, 0.0 ], [ 0.0, -1.569986343383789E-4, 0.0 ], [ 0.0, 2.5033950805664062E-5, 0.0 ], [ 0.0, 0.0, -1.0132789611816406E-4 ], [ 0.0, 0.0, -3.7789344787597656E-5 ], [ 0.0, 0.0, 2.5033950805664062E-5 ] ]
    Error Statistics: {meanExponent=-4.187420523813205, negative=5, min=2.5033950805664062E-5, max=2.5033950805664062E-5, mean=-1.2675921122233072E-5, count=27.0, positive=4, stdDev=5.6958555787854534E-5, zeros=18}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 3.8915e-05 +- 6.0095e-05 [0.0000e+00 - 2.1124e-04] (36#)
    relativeTol: 2.6954e-04 +- 7.7257e-04 [8.3004e-07 - 3.3055e-03] (18#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=3.8915e-05 +- 6.0095e-05 [0.0000e+00 - 2.1124e-04] (36#), relativeTol=2.6954e-04 +- 7.7257e-04 [8.3004e-07 - 3.3055e-03] (18#)}
```



### Performance
Code from [LayerTestBase.java:149](../../../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L149) executed in 0.17 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 6.0690 +- 1.3416 [4.9444 - 15.3091]
    Learning performance: 4.0811 +- 0.5126 [3.6078 - 6.0387]
    
```

