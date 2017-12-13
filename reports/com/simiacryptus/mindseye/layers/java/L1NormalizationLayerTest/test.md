# L1NormalizationLayer
## L1NormalizationLayerTest
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
      "class": "com.simiacryptus.mindseye.layers.java.L1NormalizationLayer",
      "id": "fd70365c-214b-41d3-a9dc-ad988933a469",
      "isFrozen": false,
      "name": "L1NormalizationLayer/fd70365c-214b-41d3-a9dc-ad988933a469"
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
    [[ -172.4, 167.6, -71.2, 73.6 ]]
    --------------------
    Output: 
    [ 71.83333333333273, -69.83333333333275, 29.66666666666642, -30.66666666666641 ]
    --------------------
    Derivative: 
    [ 0.0, 0.0, 0.0, 0.0 ]
```



### Batch Execution
Code from [StandardLayerTests.java:102](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L102) executed in 0.00 seconds: 
```java
    return getBatchingTester().test(layer, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (80#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (40#)}
```



### Differential Validation
Code from [StandardLayerTests.java:110](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L110) executed in 0.00 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [ -122.8, -183.6, -28.4, 56.39999999999999 ]
    Inputs Statistics: {meanExponent=1.889417121925188, negative=3, min=56.39999999999999, max=56.39999999999999, mean=-69.6, count=4.0, positive=1, stdDev=91.37789667091272, zeros=0}
    Output: [ 0.44109195402298856, 0.6594827586206897, 0.10201149425287356, -0.2025862068965517 ]
    Outputs Statistics: {meanExponent=-0.5552521090133364, negative=1, min=-0.2025862068965517, max=-0.2025862068965517, mean=0.25, count=4.0, positive=3, stdDev=0.32822520355931295, zeros=0}
    Feedback for input 0
    Inputs Values: [ -122.8, -183.6, -28.4, 56.39999999999999 ]
    Value Statistics: {meanExponent=1.889417121925188, negative=3, min=56.39999999999999, max=56.39999999999999, mean=-69.6, count=4.0, positive=1, stdDev=91.37789667091272, zeros=0}
    Implemented Feedback: [ [ -0.002007572004227771, 0.002368831747919143, 3.6642059717267796E-4, -7.276803408640504E-4 ], [ 0.0015843820187607341, -0.0012231222750693624, 3.6642059717267796E-4, -7.276803408640504E-4 ], [ 0.0015843820187607341, 0
```
...[skipping 787 bytes](etc/76.txt)...
```
    ]
    Measured Statistics: {meanExponent=-2.90405400267177, negative=7, min=-0.004319635915905895, max=-0.004319635915905895, mean=-2.6020852139652106E-13, count=16.0, positive=9, stdDev=0.001951697386984192, zeros=0}
    Feedback Error: [ [ -7.221646327812414E-10, 8.499352215952938E-10, 1.315863535755052E-10, -2.612998322489707E-10 ], [ 5.685888745509543E-10, -4.4026317400774895E-10, 1.315863535755052E-10, -2.612998322489707E-10 ], [ 5.685888745509543E-10, 8.499352215952938E-10, -1.158612042244378E-9, -2.612998322489707E-10 ], [ 5.696990975755795E-10, 8.510454446199189E-10, 1.3186390933166148E-10, -1.5520533391474856E-9 ] ]
    Error Statistics: {meanExponent=-9.348679690452578, negative=7, min=-1.5520533391474856E-9, max=-1.5520533391474856E-9, mean=-2.6020837231872235E-13, count=16.0, positive=9, stdDev=7.010552934368815E-10, zeros=0}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 5.8186e-10 +- 3.9104e-10 [1.3159e-10 - 1.5521e-09] (16#)
    relativeTol: 1.7962e-07 +- 1.7730e-10 [1.7940e-07 - 1.7998e-07] (16#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=5.8186e-10 +- 3.9104e-10 [1.3159e-10 - 1.5521e-09] (16#), relativeTol=1.7962e-07 +- 1.7730e-10 [1.7940e-07 - 1.7998e-07] (16#)}
```



### Performance
Code from [StandardLayerTests.java:120](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L120) executed in 0.00 seconds: 
```java
    getPerformanceTester().test(layer, permPrototype);
```
Logging: 
```
    Evaluation performance: 0.000141s +- 0.000028s [0.000114s - 0.000185s]
    Learning performance: 0.000025s +- 0.000001s [0.000024s - 0.000027s]
    
```

