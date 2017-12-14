# LoggingWrapperLayer
## LoggingWrapperLayerTest
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
      "class": "com.simiacryptus.mindseye.layers.java.LoggingWrapperLayer",
      "id": "a86c4d1e-c95b-4ef2-923f-df543f434979",
      "isFrozen": false,
      "name": "LoggingWrapperLayer/a86c4d1e-c95b-4ef2-923f-df543f434979",
      "inner": {
        "class": "com.simiacryptus.mindseye.layers.java.LinearActivationLayer",
        "id": "7d86518b-ba3a-406a-9b42-d412b55a6c45",
        "isFrozen": false,
        "name": "LinearActivationLayer/7d86518b-ba3a-406a-9b42-d412b55a6c45",
        "weights": [
          1.0,
          0.0
        ]
      }
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
    [[ 1.724, 0.116, -0.404 ]]
    --------------------
    Output: 
    [ 1.724, 0.116, -0.404 ]
    --------------------
    Derivative: 
    [ 1.0, 1.0, 1.0 ]
```



### Differential Validation
Code from [StandardLayerTests.java:109](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L109) executed in 0.01 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [ -1.848, -1.18, 0.132 ]
    Inputs Statistics: {meanExponent=-0.18028069820131223, negative=2, min=0.132, max=0.132, mean=-0.9653333333333333, count=3.0, positive=1, stdDev=0.8224602658419974, zeros=0}
    Output: [ -1.848, -1.18, 0.132 ]
    Outputs Statistics: {meanExponent=-0.18028069820131223, negative=2, min=0.132, max=0.132, mean=-0.9653333333333333, count=3.0, positive=1, stdDev=0.8224602658419974, zeros=0}
    Input 0 for layer LinearActivationLayer/7d86518b-ba3a-406a-9b42-d412b55a6c45: 
    	[ -1.848, -1.18, 0.132 ]
    Output for layer LinearActivationLayer/7d86518b-ba3a-406a-9b42-d412b55a6c45: 
    	[ -1.848, -1.18, 0.132 ]
    Feedback Input for layer LinearActivationLayer/7d86518b-ba3a-406a-9b42-d412b55a6c45: 
    	[ 1.0, 0.0, 0.0 ]
    Feedback Output 0 for layer LinearActivationLayer/7d86518b-ba3a-406a-9b42-d412b55a6c45: 
    	[ 1.0, 0.0, 0.0 ]
    Input 0 for layer LinearActivationLayer/7d86518b-ba3a-406a-9b42-d412b55a6c45: 
    	[ -1.848, -1.18, 0.132 ]
    Output for layer LinearActivationLayer/7d86518b-ba3a-406a-9b42-d412b55a6c
```
...[skipping 2040 bytes](etc/121.txt)...
```
    1.0, max=1.0, mean=0.01733333333333333, count=6.0, positive=4, stdDev=1.1418643624451295, zeros=0}
    Measured Gradient: [ [ -1.8479999999998498, -1.180000000000625, 0.13199999999990997 ], [ 0.9999999999998899, 0.9999999999998899, 0.9999999999998899 ] ]
    Measured Statistics: {meanExponent=-0.090140349100697, negative=2, min=0.9999999999998899, max=0.9999999999998899, mean=0.017333333333184136, count=6.0, positive=4, stdDev=1.141864362445149, zeros=0}
    Gradient Error: [ [ 1.503241975342462E-13, -6.250555628639631E-13, -9.00390872971002E-14 ], [ -1.1013412404281553E-13, -1.1013412404281553E-13, -1.1013412404281553E-13 ] ]
    Error Statistics: {meanExponent=-12.824475948545311, negative=5, min=-1.1013412404281553E-13, max=-1.1013412404281553E-13, mean=-1.4919547079254394E-13, count=6.0, positive=1, stdDev=2.324620768637247E-13, zeros=0}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 1.0175e-13 +- 1.5031e-13 [0.0000e+00 - 6.2506e-13] (15#)
    relativeTol: 1.0855e-13 +- 1.0555e-13 [4.0672e-14 - 3.4106e-13] (9#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=1.0175e-13 +- 1.5031e-13 [0.0000e+00 - 6.2506e-13] (15#), relativeTol=1.0855e-13 +- 1.0555e-13 [4.0672e-14 - 3.4106e-13] (9#)}
```



### Performance
Code from [StandardLayerTests.java:119](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L119) executed in 0.04 seconds: 
```java
    getPerformanceTester().test(layer, permPrototype);
```
Logging: 
```
    100 batches
    Input Dimensions:
    	[3]
    Performance:
    	Evaluation performance: 0.004704s +- 0.001991s [0.001384s - 0.007029s]
    	Learning performance: 0.000607s +- 0.000166s [0.000422s - 0.000913s]
    
```

