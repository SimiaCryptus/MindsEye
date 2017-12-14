# SoftmaxActivationLayer
## SoftmaxActivationLayerTest
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
      "class": "com.simiacryptus.mindseye.layers.java.SoftmaxActivationLayer",
      "id": "3064ffd5-c687-4810-8f02-3c3eab1359d3",
      "isFrozen": false,
      "name": "SoftmaxActivationLayer/3064ffd5-c687-4810-8f02-3c3eab1359d3"
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
    [[ 1.168, -1.06, -1.488, -0.296 ]]
    --------------------
    Output: 
    [ 0.7095814617712907, 0.07645293370115071, 0.0498328948812607, 0.16413270964629792 ]
    --------------------
    Derivative: 
    [ 0.0, 0.0, 0.0, 0.0 ]
```



### Batch Execution
Code from [StandardLayerTests.java:101](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L101) executed in 0.00 seconds: 
```java
    return getBatchingTester().test(layer, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (80#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (40#)}
```



### Differential Validation
Code from [StandardLayerTests.java:109](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L109) executed in 0.00 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [ 0.704, 0.368, 0.844, -0.096 ]
    Inputs Statistics: {meanExponent=-0.4194914606297867, negative=1, min=-0.096, max=-0.096, mean=0.45499999999999996, count=4.0, positive=3, stdDev=0.36210909958188015, zeros=0}
    Output: [ 0.30172958621101464, 0.2156229340147091, 0.34707163735856306, 0.1355758424157131 ]
    Outputs Statistics: {meanExponent=-0.6285214288179257, negative=0, min=0.1355758424157131, max=0.1355758424157131, mean=0.24999999999999997, count=4.0, positive=4, stdDev=0.08119963573450613, zeros=0}
    Feedback for input 0
    Inputs Values: [ 0.704, 0.368, 0.844, -0.096 ]
    Value Statistics: {meanExponent=-0.4194914606297867, negative=1, min=-0.096, max=-0.096, mean=0.45499999999999996, count=4.0, positive=3, stdDev=0.36210909958188015, zeros=0}
    Implemented Feedback: [ [ 0.21068884301594454, -0.0650598186578631, -0.10472178152577856, -0.04090724283230285 ], [ -0.0650598186578631, 0.1691296843415975, -0.07483660476054249, -0.029233260923191912 ], [ -0.10472178152577856, -0.07483660476054249, 0.226612915899
```
...[skipping 712 bytes](etc/145.txt)...
```
    8877 ] ]
    Measured Statistics: {meanExponent=-1.1317678334145058, negative=12, min=0.11719930429698877, max=0.11719930429698877, mean=1.214306433183765E-13, count=16.0, positive=4, stdDev=0.1087356345649832, zeros=0}
    Feedback Error: [ [ 4.177243500269601E-6, -1.2899149871986593E-6, -2.0762770234633576E-6, -8.110512120795832E-7 ], [ -1.8501501787698826E-6, 4.8096561493238266E-6, -2.1281791324212485E-6, -8.313260054619576E-7 ], [ -1.6014301949390797E-6, -1.144419114529338E-6, 3.4654187914540113E-6, -7.195686493599585E-7 ], [ -1.4907788751239437E-6, -1.0653449208190835E-6, -1.7148042099901506E-6, 4.2709280059227694E-6 ] ]
    Error Statistics: {meanExponent=-5.7574721030723754, negative=12, min=4.2709280059227694E-6, max=4.2709280059227694E-6, mean=1.2142587282881756E-13, count=16.0, positive=4, stdDev=2.4596495275338047E-6, zeros=0}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 2.0904e-06 +- 1.2962e-06 [7.1957e-07 - 4.8097e-06] (16#)
    relativeTol: 1.2500e-05 +- 4.0601e-06 [7.6461e-06 - 1.8221e-05] (16#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=2.0904e-06 +- 1.2962e-06 [7.1957e-07 - 4.8097e-06] (16#), relativeTol=1.2500e-05 +- 4.0601e-06 [7.6461e-06 - 1.8221e-05] (16#)}
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
    	[4]
    Performance:
    	Evaluation performance: 0.000179s +- 0.000011s [0.000166s - 0.000196s]
    	Learning performance: 0.000027s +- 0.000002s [0.000026s - 0.000030s]
    
```

