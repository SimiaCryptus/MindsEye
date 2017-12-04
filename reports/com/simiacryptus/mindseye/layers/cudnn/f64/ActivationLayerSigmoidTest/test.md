# ActivationLayer
## ActivationLayerSigmoidTest
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
      "class": "com.simiacryptus.mindseye.layers.cudnn.f64.ActivationLayer",
      "id": "a864e734-2f23-44db-97c1-50400000044b",
      "isFrozen": false,
      "name": "ActivationLayer/a864e734-2f23-44db-97c1-50400000044b",
      "mode": 0
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
    	[ [ -1.08, 1.656 ], [ -1.432, 1.516 ], [ 0.248, -0.76 ] ],
    	[ [ -1.836, -1.96 ], [ 0.344, 1.56 ], [ 0.472, -0.844 ] ],
    	[ [ -1.252, 1.204 ], [ -1.588, -1.188 ], [ -1.848, -1.364 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 0.2535060166623378, 0.8397003196067644 ], [ 0.19278725232998756, 0.819948704865942 ], [ 0.5616841716618158, 0.31864626621097447 ] ],
    	[ [ 0.137525052689379, 0.12346704756522399 ], [ 0.5851618423601453, 0.826353352980995 ], [ 0.6158570201970384, 0.3006930071413841 ] ],
    	[ [ 0.22235412126790433, 0.769235596829871 ], [ 0.1696654690094729, 0.23361682498897077 ], [ 0.1361078909718396, 0.2035909665344247 ] ]
    ]
```



### Batch Execution
Code from [LayerTestBase.java:138](../../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L138) executed in 0.01 seconds: 
```java
    BatchingTester batchingTester = getBatchingTester();
    return batchingTester==null?null:batchingTester.test(layer, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (360#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (360#)}
```



### Differential Validation
Code from [LayerTestBase.java:144](../../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.03 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ -1.08, 1.656 ], [ -1.432, 1.516 ], [ 0.248, -0.76 ] ],
    	[ [ -1.836, -1.96 ], [ 0.344, 1.56 ], [ 0.472, -0.844 ] ],
    	[ [ -1.252, 1.204 ], [ -1.588, -1.188 ], [ -1.848, -1.364 ] ]
    ]
    Inputs Statistics: {meanExponent=0.03366121702528563, negative=11, min=-1.364, max=-1.364, mean=-0.45288888888888895, count=18.0, positive=7, stdDev=1.2501139898642528, zeros=0}
    Output: [
    	[ [ 0.2535060166623378, 0.8397003196067644 ], [ 0.19278725232998756, 0.819948704865942 ], [ 0.5616841716618158, 0.31864626621097447 ] ],
    	[ [ 0.137525052689379, 0.12346704756522399 ], [ 0.5851618423601453, 0.826353352980995 ], [ 0.6158570201970384, 0.3006930071413841 ] ],
    	[ [ 0.22235412126790433, 0.769235596829871 ], [ 0.1696654690094729, 0.23361682498897077 ], [ 0.1361078909718396, 0.2035909665344247 ] ]
    ]
    Outputs Statistics: {meanExponent=-0.4869110860153207, negative=0, min=0.2035909665344247, max=0.2035909665344247, mean=0.40610560688191505, count=18.0, positive=18, stdDev=0.2625629843377186, zeros=0}
    Feedback for input 0
    Inputs Values: [
    	[ [ -1.08, 1.656 ], [ -1.432, 1.516 ], [ 0.248, -0.76 ] ],
    	[ [ -1.836, -1.96 ], [ 0.344, 1.56 ], [ 0.472, -0.844 ] ],
    	[ [ -1.252, 1.204 ], [ -1.588, -1.188 ], [ -1.848, -1.364 ] ]
    ]
    Value Statistics: {meanExponent=0.03366121702528563, negative=11, min=-1.364, max=-1.364, mean=-0.45288888888888895, count=18.0, positive=7, stdDev=1.2501139898642528, zeros=0}
    Implemented Feedback: [ [ 0.18924071617833232, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.11861191257216254, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.17291276602308242, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.15562032766904124, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.24274746060582575, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.1408790976352685, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.24619506296639565, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.23657715087106304, ... ], ... ]
    Implemented Statistics: {meanExponent=-0.7771350266709018, negative=0, min=0.16214168488000347, max=0.16214168488000347, mean=0.009569140122036527, count=324.0, positive=18, stdDev=0.04070877667443358, zeros=306}
    Measured Feedback: [ [ 0.1892453808055583, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.1186162120137002, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.1729175668638283, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.15562510854133516, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.24274539313973165, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.14088375139470477, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.24619354413668582, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.23657440979318878, ... ], ... ]
    Measured Statistics: {meanExponent=-0.777130949082409, negative=0, min=0.16214649091295108, max=0.16214649091295108, mean=0.00956921472492962, count=324.0, positive=18, stdDev=0.04070903044994623, zeros=306}
    Feedback Error: [ [ 4.664627225980933E-6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 4.299441537666815E-6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 4.800840745888824E-6, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 4.780872293913552E-6, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, -2.067466094102244E-6, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 4.653759436257232E-6, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.5188297098345238E-6, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -2.7410778742631425E-6, ... ], ... ]
    Error Statistics: {meanExponent=-5.401710696570754, negative=7, min=4.806032947612948E-6, max=4.806032947612948E-6, mean=7.460289309114503E-8, count=324.0, positive=11, stdDev=9.967710242218287E-7, zeros=306}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 2.2945e-07 +- 9.7287e-07 [0.0000e+00 - 4.8060e-06] (324#)
    relativeTol: 1.3128e-05 +- 4.6958e-06 [3.0846e-06 - 1.8827e-05] (18#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=2.2945e-07 +- 9.7287e-07 [0.0000e+00 - 4.8060e-06] (324#), relativeTol=1.3128e-05 +- 4.6958e-06 [3.0846e-06 - 1.8827e-05] (18#)}
```



### Performance
Code from [LayerTestBase.java:149](../../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L149) executed in 0.07 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 2.8146 +- 0.4029 [2.3083 - 4.7449]
    Learning performance: 1.5020 +- 0.3725 [1.1029 - 3.9612]
    
```

