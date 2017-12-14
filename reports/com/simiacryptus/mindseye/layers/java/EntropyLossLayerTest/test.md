# EntropyLossLayer
## EntropyLossLayerTest
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
      "class": "com.simiacryptus.mindseye.layers.java.EntropyLossLayer",
      "id": "2bebe267-c785-47b3-a9aa-59d465a57663",
      "isFrozen": false,
      "name": "EntropyLossLayer/2bebe267-c785-47b3-a9aa-59d465a57663"
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
    [[ 0.39547697675157234, 0.09922563135589801, 0.996267979549125, 0.5407752639948944 ],
    [ 0.8826675542131444, 0.2963204338173412, 0.9694916393140175, 0.5335641152084147 ]]
    --------------------
    Output: 
    [ 1.8350585974416944 ]
    --------------------
    Derivative: 
    [ -2.2319062956921805, -2.9863295377230954, -0.9731233555782596, -0.9866651652421962 ],
    [ 0.9276627064863882, 2.3103589174620187, 0.0037390018143380983, 0.6147514949538838 ]
```



### Batch Execution
Code from [StandardLayerTests.java:101](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L101) executed in 0.00 seconds: 
```java
    return getBatchingTester().test(layer, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (90#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (90#)}
```



### Differential Validation
Code from [StandardLayerTests.java:109](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L109) executed in 0.00 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [ 0.026384855339804947, 0.9394665716317756, 0.9699964696544269, 0.481187460508152 ],
    [ 0.7407152010982546, 0.3841934604231181, 0.3249129612841519, 0.14333592595604172 ]
    Inputs Statistics: {meanExponent=-0.4841698740250414, negative=0, min=0.481187460508152, max=0.481187460508152, mean=0.6042588392835399, count=4.0, positive=4, stdDev=0.38575028959420005, zeros=0},
    {meanExponent=-0.4694191704149083, negative=0, min=0.14333592595604172, max=0.14333592595604172, mean=0.39828938719039164, count=4.0, positive=4, stdDev=0.21670227297841133, zeros=0}
    Output: [ 2.8312118772303765 ]
    Outputs Statistics: {meanExponent=0.45197237154412123, negative=0, min=2.8312118772303765, max=2.8312118772303765, mean=2.8312118772303765, count=1.0, positive=1, stdDev=0.0, zeros=0}
    Feedback for input 0
    Inputs Values: [ 0.026384855339804947, 0.9394665716317756, 0.9699964696544269, 0.481187460508152 ]
    Value Statistics: {meanExponent=-0.4841698740250414, negative=0, min=0.481187460508152, max=0.481187460508152, mean=0.6042588
```
...[skipping 1527 bytes](etc/108.txt)...
```
    istics: {meanExponent=-0.5740079654265866, negative=0, min=0.7314983540155137, max=0.7314983540155137, mean=1.1148423344068654, count=4.0, positive=4, stdDev=1.4816708877622926, zeros=0}
    Measured Feedback: [ [ 3.6349650844869075 ], [ 0.062443028525649424 ], [ 0.03046287666563785 ], [ 0.731498328576663 ] ]
    Measured Statistics: {meanExponent=-0.5740078869087774, negative=0, min=0.731498328576663, max=0.731498328576663, mean=1.1148423295637144, count=4.0, positive=4, stdDev=1.4816708819494433, zeros=0}
    Feedback Error: [ [ -1.0320623466952838E-8 ], [ -1.3255917301335796E-8 ], [ 2.9642787505862467E-8 ], [ -2.54388506926162E-8 ] ]
    Error Statistics: {meanExponent=-7.746616938281019, negative=3, min=-2.54388506926162E-8, max=-2.54388506926162E-8, mean=-4.843150988760592E-9, count=4.0, positive=1, stdDev=2.0701691609175785E-8, zeros=0}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 6.7504e-07 +- 1.7521e-06 [2.2146e-10 - 5.3105e-06] (8#)
    relativeTol: 9.0398e-08 +- 1.5494e-07 [3.3058e-10 - 4.8654e-07] (8#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=6.7504e-07 +- 1.7521e-06 [2.2146e-10 - 5.3105e-06] (8#), relativeTol=9.0398e-08 +- 1.5494e-07 [3.3058e-10 - 4.8654e-07] (8#)}
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
    	[4]
    Performance:
    	Evaluation performance: 0.000084s +- 0.000015s [0.000074s - 0.000114s]
    	Learning performance: 0.000026s +- 0.000003s [0.000024s - 0.000030s]
    
```

