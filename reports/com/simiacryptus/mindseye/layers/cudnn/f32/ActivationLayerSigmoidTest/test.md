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
      "class": "com.simiacryptus.mindseye.layers.cudnn.f32.ActivationLayer",
      "id": "a864e734-2f23-44db-97c1-504000000020",
      "isFrozen": false,
      "name": "ActivationLayer/a864e734-2f23-44db-97c1-504000000020",
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
    	[ [ -1.8, -0.276 ], [ -0.136, 0.984 ], [ -1.78, -0.252 ] ],
    	[ [ 1.636, -1.98 ], [ 0.572, 0.312 ], [ 0.984, -0.296 ] ],
    	[ [ 1.436, 1.592 ], [ 0.784, 0.612 ], [ 1.496, -0.672 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 0.14185106754302979, 0.431434690952301 ], [ 0.4660523235797882, 0.7279012203216553 ], [ 0.14430314302444458, 0.43733128905296326 ] ],
    	[ [ 0.8369899392127991, 0.12131884694099426 ], [ 0.6392245888710022, 0.5773733854293823 ], [ 0.7279012203216553, 0.42653557658195496 ] ],
    	[ [ 0.8078344464302063, 0.830897331237793 ], [ 0.6865415573120117, 0.648396909236908 ], [ 0.8169771432876587, 0.33804914355278015 ] ]
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
Code from [LayerTestBase.java:144](../../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.05 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ -1.8, -0.276 ], [ -0.136, 0.984 ], [ -1.78, -0.252 ] ],
    	[ [ 1.636, -1.98 ], [ 0.572, 0.312 ], [ 0.984, -0.296 ] ],
    	[ [ 1.436, 1.592 ], [ 0.784, 0.612 ], [ 1.496, -0.672 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.1253728936844832, negative=8, min=-0.672, max=-0.672, mean=0.17866666666666667, count=18.0, positive=10, stdDev=1.138051355998001, zeros=0}
    Output: [
    	[ [ 0.14185106754302979, 0.431434690952301 ], [ 0.4660523235797882, 0.7279012203216553 ], [ 0.14430314302444458, 0.43733128905296326 ] ],
    	[ [ 0.8369899392127991, 0.12131884694099426 ], [ 0.6392245888710022, 0.5773733854293823 ], [ 0.7279012203216553, 0.42653557658195496 ] ],
    	[ [ 0.8078344464302063, 0.830897331237793 ], [ 0.6865415573120117, 0.648396909236908 ], [ 0.8169771432876587, 0.33804914355278015 ] ]
    ]
    Outputs Statistics: {meanExponent=-0.32779473135963444, negative=0, min=0.33804914355278015, max=0.33804914355278015, mean=0.5448285457160738, count=18.0, positive=18, stdDev=0.23661771122995706, zeros=0}
    Feedback for input 0
    Inputs Values: [
    	[ [ -1.8, -0.276 ], [ -0.136, 0.984 ], [ -1.78, -0.252 ] ],
    	[ [ 1.636, -1.98 ], [ 0.572, 0.312 ], [ 0.984, -0.296 ] ],
    	[ [ 1.436, 1.592 ], [ 0.784, 0.612 ], [ 1.496, -0.672 ] ]
    ]
    Value Statistics: {meanExponent=-0.1253728936844832, negative=8, min=-0.672, max=-0.672, mean=0.17866666666666667, count=18.0, positive=10, stdDev=1.138051355998001, zeros=0}
    Implemented Feedback: [ [ 0.12172934412956238, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.1364377737045288, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.15523795783519745, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.2488475739955902, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.2306165099143982, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.2152022421360016, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.12347974628210068, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1980610340833664, ... ], ... ]
    Implemented Statistics: {meanExponent=-0.7331834361945043, negative=0, min=0.22377191483974457, max=0.22377191483974457, mean=0.010666803307371376, count=324.0, positive=18, stdDev=0.04552166754684779, zeros=306}
    Measured Feedback: [ [ 0.12174248695373535, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.13649463653564453, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.15556812286376953, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.2485513687133789, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.2300739288330078, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.21576881408691406, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1233816146850586, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.19788742065429688, ... ], ... ]
    Measured Statistics: {meanExponent=-0.7332476235436727, negative=0, min=0.2238154411315918, max=0.2238154411315918, mean=0.010665368150781703, count=324.0, positive=18, stdDev=0.04551602719938921, zeros=306}
    Feedback Error: [ [ 1.3142824172973633E-5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 5.6862831115722656E-5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 3.301650285720825E-4, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, -2.962052822113037E-4, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, -5.425810813903809E-4, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 5.665719509124756E-4, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -9.813159704208374E-5, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.7361342906951904E-4, ... ], ... ]
    Error Statistics: {meanExponent=-3.8239592941815417, negative=10, min=4.3526291847229004E-5, max=4.3526291847229004E-5, mean=-1.4351565896728893E-6, count=324.0, positive=8, stdDev=7.401031952372363E-5, zeros=306}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 1.3380e-05 +- 7.2805e-05 [0.0000e+00 - 6.7015e-04] (324#)
    relativeTol: 6.1453e-04 +- 4.8839e-04 [5.2367e-05 - 1.7213e-03] (18#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=1.3380e-05 +- 7.2805e-05 [0.0000e+00 - 6.7015e-04] (324#), relativeTol=6.1453e-04 +- 4.8839e-04 [5.2367e-05 - 1.7213e-03] (18#)}
```



### Performance
Code from [LayerTestBase.java:149](../../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L149) executed in 0.08 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 2.5695 +- 0.5310 [2.1829 - 5.5001]
    Learning performance: 2.2713 +- 0.4095 [1.6671 - 3.9783]
    
```

