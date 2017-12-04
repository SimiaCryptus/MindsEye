# SimpleConvolutionLayer
## MultiBand
### Json Serialization
Code from [LayerTestBase.java:83](../../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L83) executed in 0.00 seconds: 
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
      "id": "370a9587-74a1-4959-b406-fa4500000442",
      "isFrozen": false,
      "name": "SimpleConvolutionLayer/370a9587-74a1-4959-b406-fa4500000442",
      "filter": {
        "dimensions": [
          1,
          1,
          9
        ],
        "data": [
          -0.596,
          0.956,
          -0.724,
          1.184,
          0.624,
          -1.064,
          -1.52,
          -1.868,
          0.024
        ]
      },
      "strideX": 1,
      "strideY": 1,
      "simple": false
    }
```



### Example Input/Output Pair
Code from [LayerTestBase.java:120](../../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L120) executed in 0.00 seconds: 
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
    	[ [ -1.248, -1.884, 0.352 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ -1.3121439218521118, -3.027776002883911, 5.424720287322998 ] ]
    ]
```



### Reference Implementation
Code from [LayerTestBase.java:131](../../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L131) executed in 0.01 seconds: 
```java
    System.out.println(new GsonBuilder().setPrettyPrinting().create().toJson(referenceLayer.getJson()));
    getEquivalencyTester().test(referenceLayer, layer, inputPrototype);
```
Logging: 
```
    {
      "class": "com.simiacryptus.mindseye.layers.aparapi.ConvolutionLayer",
      "id": "370a9587-74a1-4959-b406-fa4500000443",
      "isFrozen": false,
      "name": "ConvolutionLayer/370a9587-74a1-4959-b406-fa4500000443",
      "filter": {
        "dimensions": [
          1,
          1,
          9
        ],
        "data": [
          -0.596,
          1.184,
          -1.52,
          0.956,
          0.624,
          -1.868,
          -0.724,
          -1.064,
          0.024
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
    	[ [ -1.248, -1.884, 0.352 ] ]
    ]
    Error: [
    	[ [ 7.814788816062901E-8, -2.883911331963418E-9, 2.8732299828249097E-7 ] ]
    ]
    Accuracy:
    absoluteTol: 1.2278e-07 +- 1.2033e-07 [2.8839e-09 - 2.8732e-07] (3#)
    relativeTol: 1.8913e-08 +- 1.3106e-08 [4.7624e-10 - 2.9779e-08] (3#)
    
```

### Batch Execution
Code from [LayerTestBase.java:138](../../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L138) executed in 0.01 seconds: 
```java
    BatchingTester batchingTester = getBatchingTester();
    return batchingTester==null?null:batchingTester.test(layer, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (60#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (60#)}
```



### Differential Validation
Code from [LayerTestBase.java:144](../../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.02 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ -1.248, -1.884, 0.352 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.027387284239535088, negative=2, min=0.352, max=0.352, mean=-0.9266666666666666, count=3.0, positive=1, stdDev=0.9406966685506132, zeros=0}
    Output: [
    	[ [ -1.3121439218521118, -3.027776002883911, 5.424720287322998 ] ]
    ]
    Outputs Statistics: {meanExponent=0.4444941884533457, negative=2, min=5.424720287322998, max=5.424720287322998, mean=0.361600120862325, count=3.0, positive=1, stdDev=3.648034879619614, zeros=0}
    Feedback for input 0
    Inputs Values: [
    	[ [ -1.248, -1.884, 0.352 ] ]
    ]
    Value Statistics: {meanExponent=-0.027387284239535088, negative=2, min=0.352, max=0.352, mean=-0.9266666666666666, count=3.0, positive=1, stdDev=0.9406966685506132, zeros=0}
    Implemented Feedback: [ [ -0.5960000157356262, 1.184000015258789, -1.5199999809265137 ], [ 0.9559999704360962, 0.6240000128746033, -1.8680000305175781 ], [ -0.7239999771118164, -1.0640000104904175, 0.024000000208616257 ] ]
    Implemented Statistics: {meanExponent=-0.18396085018239638, negative=5, min=0.024000000208616257, max=0.024000000208616257, mean=-0.3315555573337608, count=9.0, positive=4, stdDev=1.0288987786983552, zeros=0}
    Measured Feedback: [ [ -0.5960464477539062, 1.184225082397461, -1.5201568603515625 ], [ 0.9559392929077148, 0.6239414215087891, -1.8682479858398438 ], [ -0.7239580154418945, -1.0640621185302734, 0.02384185791015625 ] ]
    Measured Statistics: {meanExponent=-0.18426312492656108, negative=5, min=0.02384185791015625, max=0.02384185791015625, mean=-0.33161375257703996, count=9.0, positive=4, stdDev=1.0289807991862334, zeros=0}
    Feedback Error: [ [ -4.64320182800293E-5, 2.25067138671875E-4, -1.5687942504882812E-4 ], [ -6.0677528381347656E-5, -5.8591365814208984E-5, -2.47955322265625E-4 ], [ 4.1961669921875E-5, -6.210803985595703E-5, -1.5814229846000671E-4 ] ]
    Error Statistics: {meanExponent=-4.025002309786755, negative=7, min=-1.5814229846000671E-4, max=-1.5814229846000671E-4, mean=-5.81952432791392E-5, count=9.0, positive=2, stdDev=1.2743881125647805E-4, zeros=0}
    Learning Gradient for weight set 0
    Weights: [ -0.596, 0.956, -0.724, 1.184, 0.624, -1.064, -1.52, -1.868, 0.024 ]
    Implemented Gradient: [ [ -1.2480000257492065, 0.0, 0.0 ], [ -1.8839999437332153, 0.0, 0.0 ], [ 0.35199999809265137, 0.0, 0.0 ], [ 0.0, -1.2480000257492065, 0.0 ], [ 0.0, -1.8839999437332153, 0.0 ], [ 0.0, 0.35199999809265137, 0.0 ], [ 0.0, 0.0, -1.2480000257492065 ], [ 0.0, 0.0, -1.8839999437332153 ], [ 0.0, 0.0, 0.35199999809265137 ] ]
    Implemented Statistics: {meanExponent=-0.027387286360603946, negative=6, min=0.35199999809265137, max=0.35199999809265137, mean=-0.3088888857099745, count=27.0, positive=3, stdDev=0.6969897781394293, zeros=18}
    Measured Gradient: [ [ -1.248002052307129, 0.0, 0.0 ], [ -1.8839836120605469, 0.0, 0.0 ], [ 0.35202503204345703, 0.0, 0.0 ], [ 0.0, -1.2478828430175781, 0.0 ], [ 0.0, -1.8839836120605469, 0.0 ], [ 0.0, 0.35190582275390625, 0.0 ], [ 0.0, 0.0, -1.2478828430175781 ], [ 0.0, 0.0, -1.88446044921875 ], [ 0.0, 0.0, 0.35190582275390625 ] ]
    Measured Statistics: {meanExponent=-0.027407705797150017, negative=6, min=0.35190582275390625, max=0.35190582275390625, mean=-0.30890217533818, count=27.0, positive=3, stdDev=0.6970082647722124, zeros=18}
    Gradient Error: [ [ -2.0265579223632812E-6, 0.0, 0.0 ], [ 1.633167266845703E-5, 0.0, 0.0 ], [ 2.5033950805664062E-5, 0.0, 0.0 ], [ 0.0, 1.1718273162841797E-4, 0.0 ], [ 0.0, 1.633167266845703E-5, 0.0 ], [ 0.0, -9.417533874511719E-5, 0.0 ], [ 0.0, 0.0, 1.1718273162841797E-4 ], [ 0.0, 0.0, -4.6050548553466797E-4 ], [ 0.0, 0.0, -9.417533874511719E-5 ] ]
    Error Statistics: {meanExponent=-4.346645980625458, negative=4, min=-9.417533874511719E-5, max=-9.417533874511719E-5, mean=-1.3289628205475984E-5, count=27.0, positive=5, stdDev=9.692733915812617E-5, zeros=18}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 5.5577e-05 +- 9.4851e-05 [0.0000e+00 - 4.6051e-04] (36#)
    relativeTol: 2.3461e-04 +- 7.4590e-04 [8.1192e-07 - 3.3055e-03] (18#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=5.5577e-05 +- 9.4851e-05 [0.0000e+00 - 4.6051e-04] (36#), relativeTol=2.3461e-04 +- 7.4590e-04 [8.1192e-07 - 3.3055e-03] (18#)}
```



### Performance
Code from [LayerTestBase.java:149](../../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L149) executed in 0.15 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 5.4912 +- 1.1066 [4.7449 - 12.8469]
    Learning performance: 3.9393 +- 0.8697 [3.4511 - 11.6442]
    
```

