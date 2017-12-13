# EntropyLossLayer
## EntropyLossLayerTest
### Json Serialization
Code from [StandardLayerTests.java:69](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L69) executed in 0.00 seconds: 
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
      "class": "com.simiacryptus.mindseye.layers.java.EntropyLossLayer",
      "id": "70377540-eda5-4dba-8007-8d11b47c2904",
      "isFrozen": false,
      "name": "EntropyLossLayer/70377540-eda5-4dba-8007-8d11b47c2904"
    }
```



### Example Input/Output Pair
Code from [StandardLayerTests.java:153](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L153) executed in 0.00 seconds: 
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
    [[ 0.6978861637238377, 0.7288261569959478, 0.9258647652130878, 0.01486368218870715 ],
    [ 0.01449954948550114, 0.45929129582377826, 0.6391516210614389, 0.9611866633633933 ]]
    --------------------
    Output: 
    [ 4.245206082818212 ]
    --------------------
    Derivative: 
    [ -0.020776381936178912, -0.6301794898757069, -0.6903293494642688, -64.66679327237404 ],
    [ 0.35969927874112056, 0.31632004318105755, 0.07702709688762661, 4.208834478415667 ]
```



### Batch Execution
Code from [StandardLayerTests.java:102](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L102) executed in 0.00 seconds: 
```java
    return getBatchingTester().test(layer, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (90#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (90#)}
```



### Differential Validation
Code from [StandardLayerTests.java:110](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L110) executed in 0.00 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [ 0.5215741599020668, 0.4338711883688149, 0.3022570786042357, 0.23573890396573138 ],
    [ 0.6762617113028081, 0.6355822719860797, 0.551905665188599, 0.9013643976879967 ]
    Inputs Statistics: {meanExponent=-0.44812884524235386, negative=0, min=0.23573890396573138, max=0.23573890396573138, mean=0.3733603327102122, count=4.0, positive=4, stdDev=0.11138248585114782, zeros=0},
    {meanExponent=-0.16748704337151138, negative=0, min=0.9013643976879967, max=0.9013643976879967, mean=0.6912785115413709, count=4.0, positive=4, stdDev=0.12931400276893035, zeros=0}
    Output: [ 2.9337389633038375 ]
    Outputs Statistics: {meanExponent=0.46742146879829527, negative=0, min=2.9337389633038375, max=2.9337389633038375, mean=2.9337389633038375, count=1.0, positive=1, stdDev=0.0, zeros=0}
    Feedback for input 0
    Inputs Values: [ 0.5215741599020668, 0.4338711883688149, 0.3022570786042357, 0.23573890396573138 ]
    Value Statistics: {meanExponent=-0.44812884524235386, negative=0, min=0.23573890396573138, max=0.23573890396573138, mean=0.3
```
...[skipping 1522 bytes](etc/66.txt)...
```
    cs: {meanExponent=-0.006752820809830938, negative=0, min=1.4450304255296733, max=1.4450304255296733, mean=1.0318547987956794, count=4.0, positive=4, stdDev=0.308904179826749, zeros=0}
    Measured Feedback: [ [ 0.650903775323286 ], [ 0.8350075741958563 ], [ 1.1964773616313096 ], [ 1.4450304064439479 ] ]
    Measured Statistics: {meanExponent=-0.006752830782486785, negative=0, min=1.4450304064439479, max=1.4450304064439479, mean=1.0318547793986, count=4.0, positive=4, stdDev=0.3089041854149128, zeros=0}
    Feedback Error: [ [ -3.433861339718902E-8 ], [ -1.5733522040761727E-8 ], [ -8.430457887342868E-9 ], [ -1.9085725444512036E-8 ] ]
    Error Statistics: {meanExponent=-7.765207864386825, negative=4, min=-1.9085725444512036E-8, max=-1.9085725444512036E-8, mean=-1.9397079692451413E-8, count=4.0, positive=0, stdDev=9.447682512143601E-9, zeros=0}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 2.0520e-08 +- 1.0872e-08 [7.9146e-09 - 4.0878e-08] (8#)
    relativeTol: 8.2820e-09 +- 7.1040e-09 [2.7014e-09 - 2.6378e-08] (8#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=2.0520e-08 +- 1.0872e-08 [7.9146e-09 - 4.0878e-08] (8#), relativeTol=8.2820e-09 +- 7.1040e-09 [2.7014e-09 - 2.6378e-08] (8#)}
```



### Performance
Code from [StandardLayerTests.java:120](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L120) executed in 0.00 seconds: 
```java
    getPerformanceTester().test(layer, permPrototype);
```
Logging: 
```
    Evaluation performance: 0.000221s +- 0.000009s [0.000211s - 0.000235s]
    Learning performance: 0.000025s +- 0.000005s [0.000022s - 0.000034s]
    
```

