# SchemaBiasLayer
## SchemaBiasLayerTest
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
      "class": "com.simiacryptus.mindseye.layers.cudnn.f32.SchemaBiasLayer",
      "id": "370a9587-74a1-4959-b406-fa45000003ec",
      "isFrozen": false,
      "name": "SchemaBiasLayer/370a9587-74a1-4959-b406-fa45000003ec",
      "selected": [
        "test1",
        "test2"
      ],
      "features": {
        "test2": 0.0,
        "test1": 0.0
      }
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
    	[ [ -1.456, -1.964 ], [ 0.64, -1.516 ], [ 0.504, -1.232 ] ],
    	[ [ -1.568, 0.028 ], [ 1.748, 0.816 ], [ -1.44, 1.552 ] ],
    	[ [ -0.36, 0.168 ], [ 0.776, -0.444 ], [ -0.476, -0.352 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ -1.4559999704360962, -1.9639999866485596 ], [ 0.6399999856948853, -1.5160000324249268 ], [ 0.5040000081062317, -1.2319999933242798 ] ],
    	[ [ -1.5679999589920044, 0.02800000086426735 ], [ 1.7480000257492065, 0.8159999847412109 ], [ -1.440000057220459, 1.5520000457763672 ] ],
    	[ [ -0.36000001430511475, 0.1679999977350235 ], [ 0.7760000228881836, -0.4440000057220459 ], [ -0.47600001096725464, -0.35199999809265137 ] ]
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
Code from [LayerTestBase.java:144](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.06 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ -1.456, -1.964 ], [ 0.64, -1.516 ], [ 0.504, -1.232 ] ],
    	[ [ -1.568, 0.028 ], [ 1.748, 0.816 ], [ -1.44, 1.552 ] ],
    	[ [ -0.36, 0.168 ], [ 0.776, -0.444 ], [ -0.476, -0.352 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.17082115256629501, negative=10, min=-0.352, max=-0.352, mean=-0.25422222222222224, count=18.0, positive=8, stdDev=1.0868991344163734, zeros=0}
    Output: [
    	[ [ -1.4559999704360962, -1.9639999866485596 ], [ 0.6399999856948853, -1.5160000324249268 ], [ 0.5040000081062317, -1.2319999933242798 ] ],
    	[ [ -1.5679999589920044, 0.02800000086426735 ], [ 1.7480000257492065, 0.8159999847412109 ], [ -1.440000057220459, 1.5520000457763672 ] ],
    	[ [ -0.36000001430511475, 0.1679999977350235 ], [ 0.7760000228881836, -0.4440000057220459 ], [ -0.47600001096725464, -0.35199999809265137 ] ]
    ]
    Outputs Statistics: {meanExponent=-0.17082114921657895, negative=10, min=-0.35199999809265137, max=-0.35199999809265137, mean=-0.2542222198098898, count=18.0, positive=8, stdDev=1.0868991410089401, zeros=0}
    Feedback for input 0
    Inputs Values: [
    	[ [ -1.456, -1.964 ], [ 0.64, -1.516 ], [ 0.504, -1.232 ] ],
    	[ [ -1.568, 0.028 ], [ 1.748, 0.816 ], [ -1.44, 1.552 ] ],
    	[ [ -0.36, 0.168 ], [ 0.776, -0.444 ], [ -0.476, -0.352 ] ]
    ]
    Value Statistics: {meanExponent=-0.17082115256629501, negative=10, min=-0.352, max=-0.352, mean=-0.25422222222222224, count=18.0, positive=8, stdDev=1.0868991344163734, zeros=0}
    Implemented Feedback: [ [ 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, ... ], ... ]
    Implemented Statistics: {meanExponent=0.0, negative=0, min=1.0, max=1.0, mean=0.05555555555555555, count=324.0, positive=18, stdDev=0.2290614236454256, zeros=306}
    Measured Feedback: [ [ 1.000165939331
```
...[skipping 846 bytes](etc/1.txt)...
```
    0.0, 0.0, ... ], [ 0.0, 0.0, 1.659393310546875E-4, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 1.659393310546875E-4, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 1.659393310546875E-4, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, -4.3010711669921875E-4, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.659393310546875E-4, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.659393310546875E-4, ... ], ... ]
    Error Statistics: {meanExponent=-3.884238688333719, negative=4, min=-1.3208389282226562E-4, max=-1.3208389282226562E-4, mean=1.8027645570260507E-6, count=324.0, positive=14, stdDev=7.054969510679156E-5, zeros=306}
    Learning Gradient for weight set 0
    Weights: [ 0.0, 0.0 ]
    Implemented Gradient: [ [ 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ] ]
    Implemented Statistics: {meanExponent=0.0, negative=0, min=1.0, max=1.0, mean=0.5, count=36.0, positive=18, stdDev=0.5, zeros=18}
    Measured Gradient: [ [ 1.0001659393310547, 1.0001659393310547, 0.9998679161071777, 1.0001659393310547, 1.0001659393310547, 1.0001659393310547, 1.0001659393310547, 1.0001659393310547, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ] ]
    Measured Statistics: {meanExponent=3.5659561643037246E-5, negative=0, min=0.9998679161071777, max=0.9998679161071777, mean=0.5000410601496696, count=36.0, positive=18, stdDev=0.5000410679022831, zeros=18}
    Gradient Error: [ [ 1.659393310546875E-4, 1.659393310546875E-4, -1.3208389282226562E-4, 1.659393310546875E-4, 1.659393310546875E-4, 1.659393310546875E-4, 1.659393310546875E-4, 1.659393310546875E-4, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ] ]
    Error Statistics: {meanExponent=-3.96769361840536, negative=5, min=-1.3208389282226562E-4, max=-1.3208389282226562E-4, mean=4.106014966964722E-5, count=36.0, positive=13, stdDev=9.715547382029345E-5, zeros=18}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 1.7507e-05 +- 7.2722e-05 [0.0000e+00 - 1.0262e-03] (360#)
    relativeTol: 8.7538e-05 +- 7.9573e-05 [8.4937e-07 - 5.1334e-04] (36#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=1.7507e-05 +- 7.2722e-05 [0.0000e+00 - 1.0262e-03] (360#), relativeTol=8.7538e-05 +- 7.9573e-05 [8.4937e-07 - 5.1334e-04] (36#)}
```



### Performance
Code from [LayerTestBase.java:149](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L149) executed in 0.11 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 3.5952 +- 0.6938 [3.1291 - 8.8657]
    Learning performance: 3.5957 +- 0.3979 [3.0122 - 5.7167]
    
```

