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
      "class": "com.simiacryptus.mindseye.layers.cudnn.f64.ActivationLayer",
      "id": "370a9587-74a1-4959-b406-fa450000044b",
      "isFrozen": false,
      "name": "ActivationLayer/370a9587-74a1-4959-b406-fa450000044b",
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
    	[ [ -1.976, -1.188 ], [ -0.54, 1.416 ], [ -0.02, -0.804 ] ],
    	[ [ 0.58, -0.272 ], [ -0.548, 0.212 ], [ 1.972, 1.776 ] ],
    	[ [ -0.856, -1.124 ], [ 1.744, -1.732 ], [ -0.796, 1.844 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 0.121745886493239, 0.23361682498897077 ], [ 0.3681875822638983, 0.8047105766252601 ], [ 0.4950001666600003, 0.309170530929853 ] ],
    	[ [ 0.6410674063348171, 0.4324161639899626 ], [ 0.3663285486362262, 0.5528023854446937 ], [ 0.8778257706856875, 0.8552022438686094 ] ],
    	[ [ 0.29817573719142637, 0.2452700786423919 ], [ 0.8511944274200296, 0.15033193655972285 ], [ 0.31088180720848063, 0.8634210939295217 ] ]
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
Code from [LayerTestBase.java:144](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.03 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ -1.976, -1.188 ], [ -0.54, 1.416 ], [ -0.02, -0.804 ] ],
    	[ [ 0.58, -0.272 ], [ -0.548, 0.212 ], [ 1.972, 1.776 ] ],
    	[ [ -0.856, -1.124 ], [ 1.744, -1.732 ], [ -0.796, 1.844 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.1167878857673712, negative=11, min=1.844, max=1.844, mean=-0.01733333333333336, count=18.0, positive=7, stdDev=1.2499777775802434, zeros=0}
    Output: [
    	[ [ 0.121745886493239, 0.23361682498897077 ], [ 0.3681875822638983, 0.8047105766252601 ], [ 0.4950001666600003, 0.309170530929853 ] ],
    	[ [ 0.6410674063348171, 0.4324161639899626 ], [ 0.3663285486362262, 0.5528023854446937 ], [ 0.8778257706856875, 0.8552022438686094 ] ],
    	[ [ 0.29817573719142637, 0.2452700786423919 ], [ 0.8511944274200296, 0.15033193655972285 ], [ 0.31088180720848063, 0.8634210939295217 ] ]
    ]
    Outputs Statistics: {meanExponent=-0.3813772775724827, negative=0, min=0.8634210939295217, max=0.8634210939295217, mean=0.4876305093262662, count=18.0, positive=18, stdDev=0.2572349703061641, zeros=0}
    Feedback for input 0
    Inputs Values: [
    	[ [ -1.976, -1.188 ], [ -0.54, 1.416 ], [ -0.02, -0.804 ] ],
    	[ [ 0.58, -0.272 ], [ -0.548, 0.212 ], [ 1.972, 1.776 ] ],
    	[ [ -0.856, -1.124 ], [ 1.744, -1.732 ], [ -0.796, 1.844 ] ]
    ]
    Value Statistics: {meanExponent=-0.1167878857673712, negative=11, min=1.844, max=1.844, mean=-0.01733333333333336, count=18.0, positive=7, stdDev=1.2499777775802434, zeros=0}
    Implemented Feedback: [ [ 0.10692382561521437, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.2300999868699676, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.2092669669417758, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.23262548653056345, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.23213194309030227, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.12666247414911758, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2499750016665722, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.10724768700576631, ... ], ... ]
    Implemented Statistics: {meanExponent=-0.7552267841253092, negative=0, min=0.11792510848706976, max=0.11792510848706976, mean=0.010204286986225513, count=324.0, positive=18, stdDev=0.04381170649482272, zeros=306}
    Measured Feedback: [ [ 0.10692787011656879, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.23009674076202025, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.20927119036784347, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.23262855267047744, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.23213504587948908, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.12665802588407438, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.24997512644120423, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.107243634976939, ... ], ... ]
    Measured Statistics: {meanExponent=-0.7552262471430027, negative=0, min=0.11792082289852068, max=0.11792082289852068, mean=0.010204323325318105, count=324.0, positive=18, stdDev=0.04381193428763514, zeros=306}
    Feedback Error: [ [ 4.044501354424912E-6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, -3.2461079473578014E-6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 4.223426067678782E-6, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 3.066139913993071E-6, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 3.102789186804067E-6, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, -4.448265043199839E-6, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.2477463204318973E-7, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -4.052028827306042E-6, ... ], ... ]
    Error Statistics: {meanExponent=-5.523055517570843, negative=7, min=-4.285588549077235E-6, max=-4.285588549077235E-6, mean=3.633909259267987E-8, count=324.0, positive=11, stdDev=9.002888228039684E-7, zeros=306}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 2.0007e-07 +- 8.7853e-07 [0.0000e+00 - 4.7886e-06] (324#)
    relativeTol: 1.1431e-05 +- 5.9283e-06 [2.4957e-07 - 1.8913e-05] (18#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=2.0007e-07 +- 8.7853e-07 [0.0000e+00 - 4.7886e-06] (324#), relativeTol=1.1431e-05 +- 5.9283e-06 [2.4957e-07 - 1.8913e-05] (18#)}
```



### Performance
Code from [LayerTestBase.java:149](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L149) executed in 0.07 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 2.8145 +- 0.6786 [2.3254 - 7.6859]
    Learning performance: 1.4364 +- 0.3509 [1.0630 - 3.2887]
    
```

