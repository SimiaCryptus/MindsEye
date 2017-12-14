# FullyConnectedLayer
## FullyConnectedLayerTest
### Json Serialization
Code from [StandardLayerTests.java:68](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L68) executed in 0.00 seconds: 
```java
    JsonObject json = layer.getJson();
    NNLayer echo = NNLayer.fromJson(json);
    if ((echo == null)) throw new AssertionError("Failed to deserialize");
    if ((layer == echo)) throw new AssertionError("Serialization did not copy");
    if ((!layer.equals(echo))) throw new AssertionError("Serialization not equal");
    return new GsonBuilder().setPrettyPrinting().create().toJson(json);
```

Returns: 

```
    {
      "class": "com.simiacryptus.mindseye.layers.java.FullyConnectedLayer",
      "id": "fe74c826-9c7e-4791-bfb8-f06728effb36",
      "isFrozen": false,
      "name": "FullyConnectedLayer/fe74c826-9c7e-4791-bfb8-f06728effb36",
      "outputDims": [
        3
      ],
      "inputDims": [
        3
      ],
      "weights": [
        [
          -0.8219380261039534,
          -0.2499612562705907,
          0.46554170031147474
        ],
        [
          0.8233131335060352,
          -0.10221874927732555,
          -0.8475029735888698
        ],
        [
          -0.6136735380538921,
          -0.5514241788157865,
          -0.33301547717644736
        ]
      ]
    }
```



### Example Input/Output Pair
Code from [StandardLayerTests.java:152](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L152) executed in 0.00 seconds: 
```java
    SimpleEval eval = SimpleEval.run(layer, inputPrototype);
    return String.format("--------------------\nInput: \n[%s]\n--------------------\nOutput: \n%s\n--------------------\nDerivative: \n%s",
      Arrays.stream(inputPrototype).map(t -> t.prettyPrint()).reduce((a, b) -> a + ",\n" + b).get(),
      eval.getOutput().prettyPrint(),
      Arrays.stream(eval.getDerivative()).map(t -> t.prettyPrint()).reduce((a, b) -> a + ",\n" + b).get());
```

Returns: 

```
    --------------------
    Input: 
    [[ -0.416, -1.276, 0.472 ]]
    --------------------
    Output: 
    [ -0.9982752494558935, -0.02585720571461811, 0.7305651417425413 ]
    --------------------
    Derivative: 
    [ -0.6063575820630693, -0.1264085893601602, -1.4981131940461259 ]
```



### Batch Execution
Code from [StandardLayerTests.java:101](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L101) executed in 0.00 seconds: 
```java
    return getBatchingTester().test(layer, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (60#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (60#)}
```



### Differential Validation
Code from [StandardLayerTests.java:109](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L109) executed in 0.00 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [ -1.944, -1.332, -1.712 ]
    Inputs Statistics: {meanExponent=0.2155680819218908, negative=3, min=-1.712, max=-1.712, mean=-1.6626666666666665, count=3.0, positive=0, stdDev=0.25227145872826967, zeros=0}
    Output: [ 1.5518035260643097, 1.5661182503600526, 0.7939833923409456 ]
    Outputs Statistics: {meanExponent=0.09515756777300387, negative=0, min=0.7939833923409456, max=0.7939833923409456, mean=1.3039683895884358, count=3.0, positive=3, stdDev=0.36066119918473666, zeros=0}
    Feedback for input 0
    Inputs Values: [ -1.944, -1.332, -1.712 ]
    Value Statistics: {meanExponent=0.2155680819218908, negative=3, min=-1.712, max=-1.712, mean=-1.6626666666666665, count=3.0, positive=0, stdDev=0.25227145872826967, zeros=0}
    Implemented Feedback: [ [ -0.8219380261039534, -0.2499612562705907, 0.46554170031147474 ], [ 0.8233131335060352, -0.10221874927732555, -0.8475029735888698 ], [ -0.6136735380538921, -0.5514241788157865, -0.33301547717644736 ] ]
    Implemented Statistics: {meanExponent=-0.346022798118883, negative=7, m
```
...[skipping 1942 bytes](etc/109.txt)...
```
     -1.7120000000003799, 0.0 ], [ 0.0, 0.0, -1.7119999999992697 ] ]
    Measured Statistics: {meanExponent=0.21556808192190058, negative=9, min=-1.7119999999992697, max=-1.7119999999992697, mean=-0.554222222222246, count=27.0, positive=0, stdDev=0.7972064806808817, zeros=18}
    Gradient Error: [ [ -3.89910326248355E-13, 0.0, 0.0 ], [ 0.0, -3.89910326248355E-13, 0.0 ], [ 0.0, 0.0, -3.89910326248355E-13 ], [ -5.548894677076532E-13, 0.0, 0.0 ], [ 0.0, -5.548894677076532E-13, 0.0 ], [ 0.0, 0.0, 1.6655565815426598E-12 ], [ -3.7991831902672857E-13, 0.0, 0.0 ], [ 0.0, -3.7991831902672857E-13, 0.0 ], [ 0.0, 0.0, 7.30304705598428E-13 ] ]
    Error Statistics: {meanExponent=-12.277138762861403, negative=7, min=7.30304705598428E-13, max=7.30304705598428E-13, mean=-2.3832787595286693E-14, count=27.0, positive=2, stdDev=4.151152597751596E-13, zeros=18}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 2.9171e-13 +- 3.8035e-13 [0.0000e+00 - 1.6656e-12] (36#)
    relativeTol: 5.8643e-13 +- 1.0481e-12 [2.9016e-14 - 4.7339e-12] (18#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=2.9171e-13 +- 3.8035e-13 [0.0000e+00 - 1.6656e-12] (36#), relativeTol=5.8643e-13 +- 1.0481e-12 [2.9016e-14 - 4.7339e-12] (18#)}
```



### Performance
Code from [StandardLayerTests.java:119](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L119) executed in 0.00 seconds: 
```java
    getPerformanceTester().test(layer, permPrototype);
```
Logging: 
```
    100 batches
    Input Dimensions:
    	[3]
    Performance:
    	Evaluation performance: 0.000136s +- 0.000022s [0.000112s - 0.000176s]
    	Learning performance: 0.000073s +- 0.000008s [0.000065s - 0.000086s]
    
```

