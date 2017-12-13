# FullyConnectedLayer
## FullyConnectedLayerTest
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
      "class": "com.simiacryptus.mindseye.layers.java.FullyConnectedLayer",
      "id": "854ce7d6-8555-48e5-ba00-ff3977cbb37a",
      "isFrozen": false,
      "name": "FullyConnectedLayer/854ce7d6-8555-48e5-ba00-ff3977cbb37a",
      "outputDims": [
        3
      ],
      "inputDims": [
        3
      ],
      "weights": [
        [
          0.4721442089926464,
          -0.825414906357326,
          0.18770801439145707
        ],
        [
          0.40039253396233426,
          0.2176920027655431,
          0.5249348541763259
        ],
        [
          -0.06212213056121643,
          0.7880841048517315,
          0.7234661766549604
        ]
      ]
    }
```



### Example Input/Output Pair
Code from [StandardLayerTests.java:153](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L153) executed in 0.01 seconds: 
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
    [[ -1.668, 0.988, -0.604 ]]
    --------------------
    Output: 
    [ -0.35442695018597326, 1.1158689632059304, -0.23143490277833645 ]
    --------------------
    Derivative: 
    [ -0.16556268297322246, 1.1430193909042032, 1.4494281509454754 ]
```



### Batch Execution
Code from [StandardLayerTests.java:102](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L102) executed in 0.00 seconds: 
```java
    return getBatchingTester().test(layer, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (60#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (60#)}
```



### Differential Validation
Code from [StandardLayerTests.java:110](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L110) executed in 0.00 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [ 0.732, 0.492, -0.1 ]
    Inputs Statistics: {meanExponent=-0.48117460539141593, negative=1, min=-0.1, max=-0.1, mean=0.37466666666666665, count=3.0, positive=2, stdDev=0.34964871259912034, zeros=0}
    Output: [ 0.5488149007482073, -0.5759076565780885, 0.3233235971238029 ]
    Outputs Statistics: {meanExponent=-0.3301946171347137, negative=1, min=0.3233235971238029, max=0.3233235971238029, mean=0.09874361376464054, count=3.0, positive=2, stdDev=0.48585137263286043, zeros=0}
    Feedback for input 0
    Inputs Values: [ 0.732, 0.492, -0.1 ]
    Value Statistics: {meanExponent=-0.48117460539141593, negative=1, min=-0.1, max=-0.1, mean=0.37466666666666665, count=3.0, positive=2, stdDev=0.34964871259912034, zeros=0}
    Implemented Feedback: [ [ 0.4721442089926464, -0.825414906357326, 0.18770801439145707 ], [ 0.40039253396233426, 0.2176920027655431, 0.5249348541763259 ], [ -0.06212213056121643, 0.7880841048517315, 0.7234661766549604 ] ]
    Implemented Statistics: {meanExponent=-0.4362332457923286, negative=2, min=0.7234661766
```
...[skipping 1903 bytes](etc/67.txt)...
```
    995449, 0.0 ], [ 0.0, 0.0, -0.10000000000010001 ] ]
    Measured Statistics: {meanExponent=-0.4811746053918997, negative=3, min=-0.10000000000010001, max=-0.10000000000010001, mean=0.12488888888888221, count=27.0, positive=6, stdDev=0.2682272852324443, zeros=18}
    Gradient Error: [ [ -4.5075054799781356E-14, 0.0, 0.0 ], [ 0.0, -4.5075054799781356E-14, 0.0 ], [ 0.0, 0.0, -4.5075054799781356E-14 ], [ -2.851052727237402E-13, 0.0, 0.0 ], [ 0.0, -2.851052727237402E-13, 0.0 ], [ 0.0, 0.0, -2.851052727237402E-13 ], [ 4.551081733694673E-13, 0.0, 0.0 ], [ 0.0, 4.551081733694673E-13, 0.0 ], [ 0.0, 0.0, -1.0000333894311098E-13 ] ]
    Error Statistics: {meanExponent=-12.81743685459136, negative=7, min=-1.0000333894311098E-13, max=-1.0000333894311098E-13, mean=-6.6788138805459646E-15, count=27.0, positive=2, stdDev=1.578787064964073E-13, zeros=18}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 1.3328e-13 +- 1.9925e-13 [0.0000e+00 - 6.9522e-13] (36#)
    relativeTol: 6.5518e-13 +- 9.5381e-13 [3.0789e-14 - 3.4950e-12] (18#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=1.3328e-13 +- 1.9925e-13 [0.0000e+00 - 6.9522e-13] (36#), relativeTol=6.5518e-13 +- 9.5381e-13 [3.0789e-14 - 3.4950e-12] (18#)}
```



### Performance
Code from [StandardLayerTests.java:120](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L120) executed in 0.00 seconds: 
```java
    getPerformanceTester().test(layer, permPrototype);
```
Logging: 
```
    Evaluation performance: 0.000279s +- 0.000054s [0.000211s - 0.000365s]
    Learning performance: 0.000296s +- 0.000034s [0.000245s - 0.000336s]
    
```

