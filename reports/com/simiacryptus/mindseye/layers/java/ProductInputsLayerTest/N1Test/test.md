# ProductInputsLayer
## N1Test
### Json Serialization
Code from [StandardLayerTests.java:68](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L68) executed in 0.00 seconds: 
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
      "class": "com.simiacryptus.mindseye.layers.java.ProductInputsLayer",
      "id": "447f6510-d3ba-4eac-9122-05f1deb44fd2",
      "isFrozen": false,
      "name": "ProductInputsLayer/447f6510-d3ba-4eac-9122-05f1deb44fd2"
    }
```



### Example Input/Output Pair
Code from [StandardLayerTests.java:152](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L152) executed in 0.00 seconds: 
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
    [[ 1.012, 1.976, -1.452 ],
    [ 0.88 ]]
    --------------------
    Output: 
    [ 0.89056, 1.73888, -1.27776 ]
    --------------------
    Derivative: 
    [ 0.88, 0.88, 0.88 ],
    [ 1.536 ]
```



### Batch Execution
Code from [StandardLayerTests.java:101](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L101) executed in 0.00 seconds: 
```java
    return getBatchingTester().test(layer, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (70#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (70#)}
```



### Differential Validation
Code from [StandardLayerTests.java:109](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L109) executed in 0.00 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [ -1.388, -1.4, 1.016 ],
    [ -0.432 ]
    Inputs Statistics: {meanExponent=0.09847040324832484, negative=2, min=1.016, max=1.016, mean=-0.5906666666666666, count=3.0, positive=1, stdDev=1.136095457648207, zeros=0},
    {meanExponent=-0.3645162531850879, negative=1, min=-0.432, max=-0.432, mean=-0.432, count=1.0, positive=0, stdDev=0.0, zeros=0}
    Output: [ 0.5996159999999999, 0.6048, -0.438912 ]
    Outputs Statistics: {meanExponent=-0.26604584993676306, negative=1, min=-0.438912, max=-0.438912, mean=0.255168, count=3.0, positive=2, stdDev=0.49079323770402544, zeros=0}
    Feedback for input 0
    Inputs Values: [ -1.388, -1.4, 1.016 ]
    Value Statistics: {meanExponent=0.09847040324832484, negative=2, min=1.016, max=1.016, mean=-0.5906666666666666, count=3.0, positive=1, stdDev=1.136095457648207, zeros=0}
    Implemented Feedback: [ [ -0.432, 0.0, 0.0 ], [ 0.0, -0.432, 0.0 ], [ 0.0, 0.0, -0.432 ] ]
    Implemented Statistics: {meanExponent=-0.3645162531850879, negative=3, min=-0.432, max=-0.432, mean=-0.14400000000000002, coun
```
...[skipping 906 bytes](etc/136.txt)...
```
     stdDev=0.0, zeros=0}
    Implemented Feedback: [ [ -1.388, -1.4, 1.016 ] ]
    Implemented Statistics: {meanExponent=0.09847040324832484, negative=2, min=1.016, max=1.016, mean=-0.5906666666666666, count=3.0, positive=1, stdDev=1.136095457648207, zeros=0}
    Measured Feedback: [ [ -1.3879999999999448, -1.40000000000029, 1.01600000000035 ] ]
    Measured Statistics: {meanExponent=0.09847040324839895, negative=2, min=1.01600000000035, max=1.01600000000035, mean=-0.5906666666666283, count=3.0, positive=1, stdDev=1.136095457648428, zeros=0}
    Feedback Error: [ [ 5.5067062021407764E-14, -2.899902540320909E-13, 3.4994229736184934E-13 ] ]
    Error Statistics: {meanExponent=-12.750909417554086, negative=1, min=3.4994229736184934E-13, max=3.4994229736184934E-13, mean=3.833970178372207E-14, count=3.0, positive=2, stdDev=2.6151898722499476E-13, zeros=0}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 1.2168e-13 +- 1.4119e-13 [0.0000e+00 - 3.4994e-13] (12#)
    relativeTol: 1.9686e-13 +- 1.1969e-13 [1.9837e-14 - 3.9944e-13] (6#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=1.2168e-13 +- 1.4119e-13 [0.0000e+00 - 3.4994e-13] (12#), relativeTol=1.9686e-13 +- 1.1969e-13 [1.9837e-14 - 3.9944e-13] (6#)}
```



### Performance
Code from [StandardLayerTests.java:119](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L119) executed in 0.00 seconds: 
```java
    getPerformanceTester().test(layer, permPrototype);
```
Logging: 
```
    100 batches
    Input Dimensions:
    	[3]
    	[1]
    Performance:
    	Evaluation performance: 0.000138s +- 0.000022s [0.000118s - 0.000169s]
    	Learning performance: 0.000102s +- 0.000053s [0.000068s - 0.000208s]
    
```

