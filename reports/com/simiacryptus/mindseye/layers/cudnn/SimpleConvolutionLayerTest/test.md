# SimpleConvolutionLayer
## SimpleConvolutionLayerTest
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
      "class": "com.simiacryptus.mindseye.layers.cudnn.SimpleConvolutionLayer",
      "id": "108ab0c0-fbe1-427c-bfa2-ed0667e50ca0",
      "isFrozen": false,
      "name": "SimpleConvolutionLayer/108ab0c0-fbe1-427c-bfa2-ed0667e50ca0",
      "filter": [
        [
          [
            -0.464
          ]
        ]
      ],
      "strideX": 1,
      "strideY": 1,
      "simple": false
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
    [[
    	[ [ -0.804 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 0.37305600000000005 ] ]
    ]
    --------------------
    Derivative: 
    [
    	[ [ -0.464 ] ]
    ]
```



[GPU Log](etc/cuda.log)

### Reference Implementation
Code from [StandardLayerTests.java:93](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L93) executed in 0.01 seconds: 
```java
    System.out.println(new GsonBuilder().setPrettyPrinting().create().toJson(referenceLayer.getJson()));
    getEquivalencyTester().test(referenceLayer, layer, inputPrototype);
```
Logging: 
```
    {
      "class": "com.simiacryptus.mindseye.layers.aparapi.ConvolutionLayer",
      "id": "5d4d9fa5-5380-44d9-8cf7-9a10fac695b4",
      "isFrozen": false,
      "name": "ConvolutionLayer/5d4d9fa5-5380-44d9-8cf7-9a10fac695b4",
      "filter": [
        [
          [
            -0.464
          ]
        ]
      ],
      "skip": [
        [
          0.0
        ]
      ],
      "simple": true
    }
    Inputs: [
    	[ [ -0.052 ] ]
    ]
    Error: [
    	[ [ 0.0 ] ]
    ]
    Accuracy:
    absoluteTol: 0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (1#)
    relativeTol: 0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (1#)
    
```

### Batch Execution
Code from [StandardLayerTests.java:102](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L102) executed in 0.01 seconds: 
```java
    return getBatchingTester().test(layer, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (20#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (20#)}
```



### Differential Validation
Code from [StandardLayerTests.java:110](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L110) executed in 0.01 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ 0.168 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.7746907182741372, negative=0, min=0.168, max=0.168, mean=0.168, count=1.0, positive=1, stdDev=0.0, zeros=0}
    Output: [
    	[ [ -0.07795200000000001 ] ]
    ]
    Outputs Statistics: {meanExponent=-1.1081727377192563, negative=1, min=-0.07795200000000001, max=-0.07795200000000001, mean=-0.07795200000000001, count=1.0, positive=0, stdDev=0.0, zeros=0}
    Feedback for input 0
    Inputs Values: [
    	[ [ 0.168 ] ]
    ]
    Value Statistics: {meanExponent=-0.7746907182741372, negative=0, min=0.168, max=0.168, mean=0.168, count=1.0, positive=1, stdDev=0.0, zeros=0}
    Implemented Feedback: [ [ -0.464 ] ]
    Implemented Statistics: {meanExponent=-0.3334820194451191, negative=1, min=-0.464, max=-0.464, mean=-0.464, count=1.0, positive=0, stdDev=0.0, zeros=0}
    Measured Feedback: [ [ -0.46400000000001995 ] ]
    Measured Statistics: {meanExponent=-0.3334820194451005, negative=1, min=-0.46400000000001995, max=-0.46400000000001995, mean=-0.46400000000001995, count=1.0, positive=0, stdDev
```
...[skipping 139 bytes](etc/44.txt)...
```
    850329202156E-14, max=-1.992850329202156E-14, mean=-1.992850329202156E-14, count=1.0, positive=0, stdDev=0.0, zeros=0}
    Learning Gradient for weight set 0
    Weights: [ -0.464 ]
    Implemented Gradient: [ [ 0.168 ] ]
    Implemented Statistics: {meanExponent=-0.7746907182741372, negative=0, min=0.168, max=0.168, mean=0.168, count=1.0, positive=1, stdDev=0.0, zeros=0}
    Measured Gradient: [ [ 0.16799999999997373 ] ]
    Measured Statistics: {meanExponent=-0.7746907182742051, negative=0, min=0.16799999999997373, max=0.16799999999997373, mean=0.16799999999997373, count=1.0, positive=1, stdDev=0.0, zeros=0}
    Gradient Error: [ [ -2.628453010800058E-14 ] ]
    Error Statistics: {meanExponent=-13.580299782515691, negative=1, min=-2.628453010800058E-14, max=-2.628453010800058E-14, mean=-2.628453010800058E-14, count=1.0, positive=0, stdDev=0.0, zeros=0}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 2.3107e-14 +- 3.1780e-15 [1.9929e-14 - 2.6285e-14] (2#)
    relativeTol: 4.9851e-14 +- 2.8377e-14 [2.1475e-14 - 7.8228e-14] (2#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=2.3107e-14 +- 3.1780e-15 [1.9929e-14 - 2.6285e-14] (2#), relativeTol=4.9851e-14 +- 2.8377e-14 [2.1475e-14 - 7.8228e-14] (2#)}
```



### Performance
Code from [StandardLayerTests.java:120](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L120) executed in 0.01 seconds: 
```java
    getPerformanceTester().test(layer, permPrototype);
```
Logging: 
```
    Evaluation performance: 0.000458s +- 0.000103s [0.000394s - 0.000664s]
    Learning performance: 0.000408s +- 0.000016s [0.000386s - 0.000432s]
    
```

