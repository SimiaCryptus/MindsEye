# MeanSqLossLayer
## MeanSqLossLayerTest
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
      "class": "com.simiacryptus.mindseye.layers.java.MeanSqLossLayer",
      "id": "e051c764-9664-49b8-adad-7bb56e0932db",
      "isFrozen": false,
      "name": "MeanSqLossLayer/e051c764-9664-49b8-adad-7bb56e0932db"
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
    [[
    	[ [ -1.736 ], [ -0.552 ], [ 1.872 ] ],
    	[ [ 0.432 ], [ -1.744 ], [ 1.776 ] ]
    ],
    [
    	[ [ -0.444 ], [ -1.82 ], [ 1.892 ] ],
    	[ [ -1.12 ], [ -0.392 ], [ 0.664 ] ]
    ]]
    --------------------
    Output: 
    [ 1.4584400000000002 ]
    --------------------
    Derivative: 
    [
    	[ [ -0.43066666666666664 ], [ 0.42266666666666663 ], [ -0.006666666666666599 ] ],
    	[ [ 0.5173333333333333 ], [ -0.4506666666666666 ], [ 0.3706666666666667 ] ]
    ],
    [
    	[ [ 0.43066666666666664 ], [ -0.42266666666666663 ], [ 0.006666666666666599 ] ],
    	[ [ -0.5173333333333333 ], [ 0.4506666666666666 ], [ -0.3706666666666667 ] ]
    ]
```



### Batch Execution
Code from [StandardLayerTests.java:101](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L101) executed in 0.00 seconds: 
```java
    return getBatchingTester().test(layer, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (130#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (130#)}
```



### Differential Validation
Code from [StandardLayerTests.java:109](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L109) executed in 0.00 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ 1.724 ], [ 0.436 ], [ 1.844 ] ],
    	[ [ 1.992 ], [ 0.948 ], [ -0.548 ] ]
    ],
    [
    	[ [ -0.06 ], [ 0.84 ], [ -1.196 ] ],
    	[ [ -1.208 ], [ -1.14 ], [ -1.44 ] ]
    ]
    Inputs Statistics: {meanExponent=0.02611048289750099, negative=1, min=-0.548, max=-0.548, mean=1.066, count=6.0, positive=5, stdDev=0.9047600050105367, zeros=0},
    {meanExponent=-0.15375066769754125, negative=5, min=-1.44, max=-1.44, mean=-0.7006666666666668, count=6.0, positive=1, stdDev=0.819194455276374, zeros=0}
    Output: [ 4.663813333333333 ]
    Outputs Statistics: {meanExponent=0.6687411596914326, negative=0, min=4.663813333333333, max=4.663813333333333, mean=4.663813333333333, count=1.0, positive=1, stdDev=0.0, zeros=0}
    Feedback for input 0
    Inputs Values: [
    	[ [ 1.724 ], [ 0.436 ], [ 1.844 ] ],
    	[ [ 1.992 ], [ 0.948 ], [ -0.548 ] ]
    ]
    Value Statistics: {meanExponent=0.02611048289750099, negative=1, min=-0.548, max=-0.548, mean=1.066, count=6.0, positive=5, stdDev=0.9047600050105367, zeros=0}
    Implemented Feedback: [ [ 0.5946666666666667 ]
```
...[skipping 1659 bytes](etc/127.txt)...
```
    8888888888889, count=6.0, positive=1, stdDev=0.4140185479235095, zeros=0}
    Measured Feedback: [ [ -0.5946499999964772 ], [ -1.0666499999878454 ], [ 0.13468333333932492 ], [ -0.6959833333297638 ], [ -1.0133166666648208 ], [ -0.29731666666421575 ] ]
    Measured Statistics: {meanExponent=-0.29113991700688185, negative=5, min=-0.29731666666421575, max=-0.29731666666421575, mean=-0.5888722222172996, count=6.0, positive=1, stdDev=0.41401854792272713, zeros=0}
    Feedback Error: [ [ 1.666667018951351E-5 ], [ 1.6666678821275482E-5 ], [ 1.666667265826094E-5 ], [ 1.6666670236142878E-5 ], [ 1.666666851241061E-5 ], [ 1.666666911753767E-5 ] ]
    Error Statistics: {meanExponent=-4.778151122114185, negative=0, min=1.666666911753767E-5, max=1.666666911753767E-5, mean=1.666667158919018E-5, count=6.0, positive=6, stdDev=3.4929806806160777E-12, zeros=0}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 1.6667e-05 +- 5.1298e-12 [1.6667e-05 - 1.6667e-05] (12#)
    relativeTol: 2.1988e-05 +- 1.9071e-05 [7.8124e-06 - 6.1885e-05] (12#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=1.6667e-05 +- 5.1298e-12 [1.6667e-05 - 1.6667e-05] (12#), relativeTol=2.1988e-05 +- 1.9071e-05 [7.8124e-06 - 6.1885e-05] (12#)}
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
    	[2, 3, 1]
    	[2, 3, 1]
    Performance:
    	Evaluation performance: 0.000220s +- 0.000019s [0.000197s - 0.000248s]
    	Learning performance: 0.000051s +- 0.000012s [0.000035s - 0.000069s]
    
```

