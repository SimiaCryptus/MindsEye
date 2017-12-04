# ActivationLayer
## ActivationLayerSigmoidTest
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
      "class": "com.simiacryptus.mindseye.layers.cudnn.f32.ActivationLayer",
      "id": "370a9587-74a1-4959-b406-fa4500000020",
      "isFrozen": false,
      "name": "ActivationLayer/370a9587-74a1-4959-b406-fa4500000020",
      "mode": 0
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
    	[ [ -1.708, -1.692 ], [ -0.016, -0.456 ], [ -1.848, 1.444 ] ],
    	[ [ -0.864, 1.412 ], [ -0.816, 1.588 ], [ -0.896, 0.508 ] ],
    	[ [ 1.468, -1.212 ], [ 1.176, 1.692 ], [ 0.932, 0.232 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 0.15342330932617188, 0.1555129885673523 ], [ 0.49600011110305786, 0.38793519139289856 ], [ 0.13610787689685822, 0.8090733289718628 ] ],
    	[ [ 0.29650431871414185, 0.8040812015533447 ], [ 0.30661341547966003, 0.830334484577179 ], [ 0.2898731827735901, 0.6243375539779663 ] ],
    	[ [ 0.8127532005310059, 0.2293473780155182 ], [ 0.7642278671264648, 0.8444870114326477 ], [ 0.7174808382987976, 0.5577412247657776 ] ]
    ]
```



### Batch Execution
Code from [LayerTestBase.java:138](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L138) executed in 0.01 seconds: 
```java
    BatchingTester batchingTester = getBatchingTester();
    return batchingTester==null?null:batchingTester.test(layer, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (360#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (360#)}
```



### Differential Validation
Code from [LayerTestBase.java:144](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.04 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ -1.708, -1.692 ], [ -0.016, -0.456 ], [ -1.848, 1.444 ] ],
    	[ [ -0.864, 1.412 ], [ -0.816, 1.588 ], [ -0.896, 0.508 ] ],
    	[ [ 1.468, -1.212 ], [ 1.176, 1.692 ], [ 0.932, 0.232 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.08381981769010863, negative=9, min=0.232, max=0.232, mean=0.05244444444444442, count=18.0, positive=9, stdDev=1.2297250922336813, zeros=0}
    Output: [
    	[ [ 0.15342330932617188, 0.1555129885673523 ], [ 0.49600011110305786, 0.38793519139289856 ], [ 0.13610787689685822, 0.8090733289718628 ] ],
    	[ [ 0.29650431871414185, 0.8040812015533447 ], [ 0.30661341547966003, 0.830334484577179 ], [ 0.2898731827735901, 0.6243375539779663 ] ],
    	[ [ 0.8127532005310059, 0.2293473780155182 ], [ 0.7642278671264648, 0.8444870114326477 ], [ 0.7174808382987976, 0.5577412247657776 ] ]
    ]
    Outputs Statistics: {meanExponent=-0.36516199259927634, negative=0, min=0.5577412247657776, max=0.5577412247657776, mean=0.5119908046391275, count=18.0, positive=18, stdDev=0.2610781333434865, zeros=0}
    Feedback for input 0
    Inputs Values: [
    	[ [ -1.708, -1.692 ], [ -0.016, -0.456 ], [ -1.848, 1.444 ] ],
    	[ [ -0.864, 1.412 ], [ -0.816, 1.588 ], [ -0.896, 0.508 ] ],
    	[ [ 1.468, -1.212 ], [ 1.176, 1.692 ], [ 0.932, 0.232 ] ]
    ]
    Value Statistics: {meanExponent=-0.08381981769010863, negative=9, min=0.232, max=0.232, mean=0.05244444444444442, count=18.0, positive=9, stdDev=1.2297250922336813, zeros=0}
    Implemented Feedback: [ [ 0.12988460063934326, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.20858950912952423, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.15218544006347656, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.2499839961528778, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.21260161697864532, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.18018363416194916, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1175825223326683, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.20584672689437866, ... ], ... ]
    Implemented Statistics: {meanExponent=-0.7531003168956348, negative=0, min=0.24666595458984375, max=0.24666595458984375, mean=0.010094135056859181, count=324.0, positive=18, stdDev=0.04283413512817612, zeros=306}
    Measured Feedback: [ [ 0.12978911399841309, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.2086162567138672, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.1519918441772461, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.25004148483276367, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.21278858184814453, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.1800060272216797, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.11771917343139648, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.20563602447509766, ... ], ... ]
    Measured Statistics: {meanExponent=-0.7530058709136694, negative=0, min=0.2467632293701172, max=0.2467632293701172, mean=0.010095536708831787, count=324.0, positive=18, stdDev=0.04283715717543187, zeros=306}
    Feedback Error: [ [ -9.548664093017578E-5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 2.6747584342956543E-5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, -1.9359588623046875E-4, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 5.748867988586426E-5, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 1.8696486949920654E-4, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, -1.7760694026947021E-4, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.3665109872817993E-4, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -2.1070241928100586E-4, ... ], ... ]
    Error Statistics: {meanExponent=-3.771404822055364, negative=10, min=9.72747802734375E-5, max=9.72747802734375E-5, mean=1.4016519726058582E-6, count=324.0, positive=8, stdDev=7.359802931102992E-5, zeros=306}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 1.2693e-05 +- 7.2509e-05 [0.0000e+00 - 9.7993e-04] (324#)
    relativeTol: 6.8704e-04 +- 7.3386e-04 [6.4111e-05 - 3.4659e-03] (18#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=1.2693e-05 +- 7.2509e-05 [0.0000e+00 - 9.7993e-04] (324#), relativeTol=6.8704e-04 +- 7.3386e-04 [6.4111e-05 - 3.4659e-03] (18#)}
```



### Performance
Code from [LayerTestBase.java:149](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L149) executed in 0.07 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 2.5757 +- 0.3383 [2.1687 - 4.1265]
    Learning performance: 2.0805 +- 0.3488 [1.7355 - 4.4742]
    
```

