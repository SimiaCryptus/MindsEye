# SimpleConvolutionLayer
## SimpleConvolutionLayerTest
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
      "class": "com.simiacryptus.mindseye.layers.cudnn.SimpleConvolutionLayer",
      "id": "d4b0477b-6f39-4c78-b10b-67b86353dec1",
      "isFrozen": false,
      "name": "SimpleConvolutionLayer/d4b0477b-6f39-4c78-b10b-67b86353dec1",
      "filter": [
        [
          [
            -1.86
          ]
        ]
      ],
      "strideX": 1,
      "strideY": 1,
      "simple": false
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
    	[ [ 0.516 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ -0.9597600000000001 ] ]
    ]
    --------------------
    Derivative: 
    [
    	[ [ -1.86 ] ]
    ]
```



[GPU Log](etc/cuda.log)

### Reference Implementation
Code from [StandardLayerTests.java:92](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L92) executed in 0.01 seconds: 
```java
    System.out.println(new GsonBuilder().setPrettyPrinting().create().toJson(referenceLayer.getJson()));
    getEquivalencyTester().test(referenceLayer, layer, inputPrototype);
```
Logging: 
```
    {
      "class": "com.simiacryptus.mindseye.layers.aparapi.ConvolutionLayer",
      "id": "37512993-8c9b-47f0-be77-3b8ad898ab9f",
      "isFrozen": false,
      "name": "ConvolutionLayer/37512993-8c9b-47f0-be77-3b8ad898ab9f",
      "filter": [
        [
          [
            -1.86
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
    	[ [ -1.112 ] ]
    ]
    Error: [
    	[ [ 0.0 ] ]
    ]
    Accuracy:
    absoluteTol: 0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (1#)
    relativeTol: 0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (1#)
    
```

### Batch Execution
Code from [StandardLayerTests.java:101](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L101) executed in 0.01 seconds: 
```java
    return getBatchingTester().test(layer, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (20#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (20#)}
```



### Differential Validation
Code from [StandardLayerTests.java:109](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L109) executed in 0.01 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ -0.16 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.7958800173440752, negative=1, min=-0.16, max=-0.16, mean=-0.16, count=1.0, positive=0, stdDev=0.0, zeros=0}
    Output: [
    	[ [ 0.29760000000000003 ] ]
    ]
    Outputs Statistics: {meanExponent=-0.5263670731261588, negative=0, min=0.29760000000000003, max=0.29760000000000003, mean=0.29760000000000003, count=1.0, positive=1, stdDev=0.0, zeros=0}
    Feedback for input 0
    Inputs Values: [
    	[ [ -0.16 ] ]
    ]
    Value Statistics: {meanExponent=-0.7958800173440752, negative=1, min=-0.16, max=-0.16, mean=-0.16, count=1.0, positive=0, stdDev=0.0, zeros=0}
    Implemented Feedback: [ [ -1.86 ] ]
    Implemented Statistics: {meanExponent=0.26951294421791633, negative=1, min=-1.86, max=-1.86, mean=-1.86, count=1.0, positive=0, stdDev=0.0, zeros=0}
    Measured Feedback: [ [ -1.8599999999996397 ] ]
    Measured Statistics: {meanExponent=0.2695129442178322, negative=1, min=-1.8599999999996397, max=-1.8599999999996397, mean=-1.8599999999996397, count=1.0, positive=0, stdDev=0.0, zeros=0
```
...[skipping 128 bytes](etc/82.txt)...
```
    7933258E-13, max=3.603783937933258E-13, mean=3.603783937933258E-13, count=1.0, positive=1, stdDev=0.0, zeros=0}
    Learning Gradient for weight set 0
    Weights: [ -1.86 ]
    Implemented Gradient: [ [ -0.16 ] ]
    Implemented Statistics: {meanExponent=-0.7958800173440752, negative=1, min=-0.16, max=-0.16, mean=-0.16, count=1.0, positive=0, stdDev=0.0, zeros=0}
    Measured Gradient: [ [ -0.16000000000016001 ] ]
    Measured Statistics: {meanExponent=-0.7958800173436409, negative=1, min=-0.16000000000016001, max=-0.16000000000016001, mean=-0.16000000000016001, count=1.0, positive=0, stdDev=0.0, zeros=0}
    Gradient Error: [ [ -1.6001089342410069E-13 ] ]
    Error Statistics: {meanExponent=-12.795850449888247, negative=1, min=-1.6001089342410069E-13, max=-1.6001089342410069E-13, mean=-1.6001089342410069E-13, count=1.0, positive=0, stdDev=0.0, zeros=0}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 2.6019e-13 +- 1.0018e-13 [1.6001e-13 - 3.6038e-13] (2#)
    relativeTol: 2.9845e-13 +- 2.0158e-13 [9.6876e-14 - 5.0003e-13] (2#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=2.6019e-13 +- 1.0018e-13 [1.6001e-13 - 3.6038e-13] (2#), relativeTol=2.9845e-13 +- 2.0158e-13 [9.6876e-14 - 5.0003e-13] (2#)}
```



### Performance
Code from [StandardLayerTests.java:119](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L119) executed in 0.37 seconds: 
```java
    getPerformanceTester().test(layer, permPrototype);
```
Logging: 
```
    100 batches
    Input Dimensions:
    	[100, 100, 1]
    Performance:
    	Evaluation performance: 0.008325s +- 0.001067s [0.006658s - 0.009966s]
    	Learning performance: 0.055792s +- 0.014964s [0.036150s - 0.069676s]
    
```

