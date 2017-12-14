# ProductInputsLayer
## NNTest
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
      "id": "59810bfa-961f-44cc-bf57-5bb64786575a",
      "isFrozen": false,
      "name": "ProductInputsLayer/59810bfa-961f-44cc-bf57-5bb64786575a"
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
    [[ 0.108, -0.808, -1.22 ],
    [ -0.072, -1.2, 0.096 ]]
    --------------------
    Output: 
    [ -0.007775999999999999, 0.9696, -0.11712 ]
    --------------------
    Derivative: 
    [ -0.072, -1.2, 0.096 ],
    [ 0.108, -0.808, -1.22 ]
```



### Batch Execution
Code from [StandardLayerTests.java:101](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L101) executed in 0.00 seconds: 
```java
    return getBatchingTester().test(layer, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (90#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (90#)}
```



### Differential Validation
Code from [StandardLayerTests.java:109](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L109) executed in 0.00 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [ -1.368, 0.96, 0.828 ],
    [ 0.476, 1.836, -0.704 ]
    Inputs Statistics: {meanExponent=0.012129222402848661, negative=1, min=0.828, max=0.828, mean=0.13999999999999993, count=3.0, positive=2, stdDev=1.0676778540365066, zeros=0},
    {meanExponent=-0.07031590375739034, negative=1, min=-0.704, max=-0.704, mean=0.5360000000000001, count=3.0, positive=2, stdDev=1.0378182242891414, zeros=0}
    Output: [ -0.651168, 1.76256, -0.582912 ]
    Outputs Statistics: {meanExponent=-0.05818668135454169, negative=2, min=-0.582912, max=-0.582912, mean=0.17615999999999998, count=3.0, positive=1, stdDev=1.1221002451902415, zeros=0}
    Feedback for input 0
    Inputs Values: [ -1.368, 0.96, 0.828 ]
    Value Statistics: {meanExponent=0.012129222402848661, negative=1, min=0.828, max=0.828, mean=0.13999999999999993, count=3.0, positive=2, stdDev=1.0676778540365066, zeros=0}
    Implemented Feedback: [ [ 0.476, 0.0, 0.0 ], [ 0.0, 1.836, 0.0 ], [ 0.0, 0.0, -0.704 ] ]
    Implemented Statistics: {meanExponent=-0.07031590375739034, negative=1, min=-0.7
```
...[skipping 1101 bytes](etc/138.txt)...
```
    mplemented Statistics: {meanExponent=0.012129222402848661, negative=1, min=0.828, max=0.828, mean=0.04666666666666665, count=9.0, positive=2, stdDev=0.6199469511355162, zeros=6}
    Measured Feedback: [ [ -1.36800000000048, 0.0, 0.0 ], [ 0.0, 0.9600000000009601, 0.0 ], [ 0.0, 0.0, 0.828000000000495 ] ]
    Measured Statistics: {meanExponent=0.012129222403130769, negative=1, min=0.828000000000495, max=0.828000000000495, mean=0.04666666666677502, count=9.0, positive=2, stdDev=0.6199469511358645, zeros=6}
    Feedback Error: [ [ -4.798383912429927E-13, 0.0, 0.0 ], [ 0.0, 9.601208716958354E-13, 0.0 ], [ 0.0, 0.0, 4.950484466803573E-13 ] ]
    Error Statistics: {meanExponent=-12.213977131576975, negative=1, min=4.950484466803573E-13, max=4.950484466803573E-13, mean=1.083701030148E-13, count=9.0, positive=2, stdDev=3.7880707805928565E-13, zeros=6}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 2.9197e-13 +- 4.7823e-13 [0.0000e+00 - 1.7251e-12] (18#)
    relativeTol: 4.7179e-13 +- 1.9692e-13 [1.7538e-13 - 7.8275e-13] (6#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=2.9197e-13 +- 4.7823e-13 [0.0000e+00 - 1.7251e-12] (18#), relativeTol=4.7179e-13 +- 1.9692e-13 [1.7538e-13 - 7.8275e-13] (6#)}
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
    	[3]
    Performance:
    	Evaluation performance: 0.000150s +- 0.000008s [0.000138s - 0.000164s]
    	Learning performance: 0.000050s +- 0.000005s [0.000044s - 0.000058s]
    
```

