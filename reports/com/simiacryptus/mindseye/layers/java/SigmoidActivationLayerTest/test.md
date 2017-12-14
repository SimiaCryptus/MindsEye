# SigmoidActivationLayer
## SigmoidActivationLayerTest
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
      "class": "com.simiacryptus.mindseye.layers.java.SigmoidActivationLayer",
      "id": "5b6a1bcb-3ca6-4591-9560-d31adc4abd11",
      "isFrozen": true,
      "name": "SigmoidActivationLayer/5b6a1bcb-3ca6-4591-9560-d31adc4abd11",
      "balanced": true
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
    	[ [ -1.476 ], [ 1.784 ], [ 1.412 ] ],
    	[ [ 1.268 ], [ -1.088 ], [ 0.432 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ -0.627935287923356 ], [ 0.712380164935704 ], [ 0.6081624090021611 ] ],
    	[ [ 0.5608003773305534 ], [ -0.4960098430228167 ], [ 0.21270229740018176 ] ]
    ]
    --------------------
    Derivative: 
    [
    	[ [ 0.302848637090306 ], [ 0.24625725030308965 ], [ 0.31506924213834414 ] ],
    	[ [ 0.34275146839295445 ], [ 0.37698711781224037 ], [ 0.4773788663403423 ] ]
    ]
```



### Batch Execution
Code from [StandardLayerTests.java:101](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L101) executed in 0.00 seconds: 
```java
    return getBatchingTester().test(layer, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (120#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (120#)}
```



### Differential Validation
Code from [StandardLayerTests.java:109](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L109) executed in 0.00 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ 1.2 ], [ -1.008 ], [ 0.652 ] ],
    	[ [ -0.408 ], [ -1.064 ], [ -1.624 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.042487135026147115, negative=4, min=-1.624, max=-1.624, mean=-0.37533333333333335, count=6.0, positive=2, stdDev=0.9976516871578421, zeros=0}
    Output: [
    	[ [ 0.5370495669980351 ], [ -0.46525712730925217 ], [ 0.314922032938433 ] ],
    	[ [ -0.20121643910254905 ], [ -0.48690841629268766 ], [ -0.6706920803812234 ] ]
    ]
    Outputs Statistics: {meanExponent=-0.3810759382274103, negative=4, min=-0.6706920803812234, max=-0.6706920803812234, mean=-0.1620170771915407, count=6.0, positive=2, stdDev=0.44233487443576586, zeros=0}
    Feedback for input 0
    Inputs Values: [
    	[ [ 1.2 ], [ -1.008 ], [ 0.652 ] ],
    	[ [ -0.408 ], [ -1.064 ], [ -1.624 ] ]
    ]
    Value Statistics: {meanExponent=-0.042487135026147115, negative=4, min=-1.624, max=-1.624, mean=-0.37533333333333335, count=6.0, positive=2, stdDev=0.9976516871578421, zeros=0}
    Implemented Feedback: [ [ 0.3557888812936114, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.
```
...[skipping 671 bytes](etc/141.txt)...
```
     0.0, 0.0, 0.0, 0.0, 0.45040496408743635, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.27509529163971536 ] ]
    Measured Statistics: {meanExponent=-0.41670159390709266, negative=0, min=0.27509529163971536, max=0.27509529163971536, mean=0.06484129949805048, count=36.0, positive=6, stdDev=0.14747328739925303, zeros=30}
    Feedback Error: [ [ -9.553852481136715E-6, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 4.826389817691457E-6, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 9.113524719495114E-6, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 9.286713908762678E-6, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, -7.0924975260067136E-6, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 9.22498276206829E-6 ] ]
    Error Statistics: {meanExponent=-5.098814312323197, negative=2, min=9.22498276206829E-6, max=9.22498276206829E-6, mean=4.3903503335761416E-7, count=36.0, positive=4, stdDev=3.384336011842769E-6, zeros=30}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 1.3638e-06 +- 3.1283e-06 [0.0000e+00 - 9.5539e-06] (36#)
    relativeTol: 1.1150e-05 +- 3.7904e-06 [5.0300e-06 - 1.6767e-05] (6#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=1.3638e-06 +- 3.1283e-06 [0.0000e+00 - 9.5539e-06] (36#), relativeTol=1.1150e-05 +- 3.7904e-06 [5.0300e-06 - 1.6767e-05] (6#)}
```



### Performance
Code from [StandardLayerTests.java:119](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L119) executed in 0.13 seconds: 
```java
    getPerformanceTester().test(layer, permPrototype);
```
Logging: 
```
    100 batches
    Input Dimensions:
    	[100, 100, 1]
    Performance:
    	Evaluation performance: 0.007216s +- 0.000715s [0.006274s - 0.008338s]
    	Learning performance: 0.010537s +- 0.000288s [0.010082s - 0.010983s]
    
```

### Function Plots
Code from [ActivationLayerTestBase.java:110](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/java/ActivationLayerTestBase.java#L110) executed in 0.00 seconds: 
```java
    return plot("Value Plot", plotData, x -> new double[]{x[0], x[1]});
```

Returns: 

![Result](etc/test.677.png)



Code from [ActivationLayerTestBase.java:114](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/java/ActivationLayerTestBase.java#L114) executed in 0.00 seconds: 
```java
    return plot("Derivative Plot", plotData, x -> new double[]{x[0], x[2]});
```

Returns: 

![Result](etc/test.678.png)



