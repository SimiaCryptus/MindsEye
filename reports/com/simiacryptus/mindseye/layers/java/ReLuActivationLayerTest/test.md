# ReLuActivationLayer
## ReLuActivationLayerTest
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
      "class": "com.simiacryptus.mindseye.layers.java.ReLuActivationLayer",
      "id": "a0a331cd-8888-4213-88dd-5d6e90ff9c89",
      "isFrozen": true,
      "name": "ReLuActivationLayer/a0a331cd-8888-4213-88dd-5d6e90ff9c89",
      "weights": [
        1.0
      ]
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
    	[ [ -1.852 ], [ 1.272 ], [ -1.688 ] ],
    	[ [ -0.916 ], [ -1.996 ], [ 1.988 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 0.0 ], [ 1.272 ], [ 0.0 ] ],
    	[ [ 0.0 ], [ 0.0 ], [ 1.988 ] ]
    ]
    --------------------
    Derivative: 
    [
    	[ [ 0.0 ], [ 1.0 ], [ 0.0 ] ],
    	[ [ 0.0 ], [ 0.0 ], [ 1.0 ] ]
    ]
```



### Batch Execution
Code from [StandardLayerTests.java:101](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L101) executed in 0.00 seconds: 
```java
    return getBatchingTester().test(layer, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (120#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (64#)}
```



### Differential Validation
Code from [StandardLayerTests.java:109](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L109) executed in 0.00 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ -1.292 ], [ -0.672 ], [ 1.112 ] ],
    	[ [ -0.916 ], [ -1.404 ], [ -1.628 ] ]
    ]
    Inputs Statistics: {meanExponent=0.050942259328958094, negative=5, min=-1.628, max=-1.628, mean=-0.8000000000000002, count=6.0, positive=1, stdDev=0.910864790551631, zeros=0}
    Output: [
    	[ [ 0.0 ], [ 0.0 ], [ 1.112 ] ],
    	[ [ 0.0 ], [ 0.0 ], [ 0.0 ] ]
    ]
    Outputs Statistics: {meanExponent=0.046104787246038705, negative=0, min=0.0, max=0.0, mean=0.18533333333333335, count=6.0, positive=1, stdDev=0.41441793182996106, zeros=5}
    Feedback for input 0
    Inputs Values: [
    	[ [ -1.292 ], [ -0.672 ], [ 1.112 ] ],
    	[ [ -0.916 ], [ -1.404 ], [ -1.628 ] ]
    ]
    Value Statistics: {meanExponent=0.050942259328958094, negative=5, min=-1.628, max=-1.628, mean=-0.8000000000000002, count=6.0, positive=1, stdDev=0.910864790551631, zeros=0}
    Implemented Feedback: [ [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 1.0, 0.0 ], [ 0.0, 0.0, 0
```
...[skipping 968 bytes](etc/140.txt)...
```
    =36.0, positive=0, stdDev=1.8098951796641285E-14, zeros=35}
    Learning Gradient for weight set 0
    Weights: [ 1.0 ]
    Implemented Gradient: [ [ 0.0, 0.0, 0.0, 0.0, 1.112, 0.0 ] ]
    Implemented Statistics: {meanExponent=0.046104787246038705, negative=0, min=0.0, max=0.0, mean=0.18533333333333335, count=6.0, positive=1, stdDev=0.41441793182996106, zeros=5}
    Measured Gradient: [ [ 0.0, 0.0, 0.0, 0.0, 1.11200000000089, 0.0 ] ]
    Measured Statistics: {meanExponent=0.04610478724638628, negative=0, min=0.0, max=0.0, mean=0.18533333333348168, count=6.0, positive=1, stdDev=0.4144179318302927, zeros=5}
    Gradient Error: [ [ 0.0, 0.0, 0.0, 0.0, 8.899547765395255E-13, 0.0 ] ]
    Error Statistics: {meanExponent=-12.050632061667832, negative=0, min=0.0, max=0.0, mean=1.4832579608992091E-13, count=6.0, positive=1, stdDev=3.316665628738357E-13, zeros=5}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 2.3812e-14 +- 1.3631e-13 [0.0000e+00 - 8.8995e-13] (42#)
    relativeTol: 2.2761e-13 +- 1.7255e-13 [5.5067e-14 - 4.0016e-13] (2#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=2.3812e-14 +- 1.3631e-13 [0.0000e+00 - 8.8995e-13] (42#), relativeTol=2.2761e-13 +- 1.7255e-13 [5.5067e-14 - 4.0016e-13] (2#)}
```



### Performance
Code from [StandardLayerTests.java:119](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L119) executed in 0.33 seconds: 
```java
    getPerformanceTester().test(layer, permPrototype);
```
Logging: 
```
    100 batches
    Input Dimensions:
    	[100, 100, 1]
    Performance:
    	Evaluation performance: 0.020045s +- 0.000262s [0.019721s - 0.020500s]
    	Learning performance: 0.025050s +- 0.014180s [0.017397s - 0.053395s]
    
```

### Function Plots
Code from [ActivationLayerTestBase.java:110](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/java/ActivationLayerTestBase.java#L110) executed in 0.00 seconds: 
```java
    return plot("Value Plot", plotData, x -> new double[]{x[0], x[1]});
```

Returns: 

![Result](etc/test.675.png)



Code from [ActivationLayerTestBase.java:114](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/java/ActivationLayerTestBase.java#L114) executed in 0.00 seconds: 
```java
    return plot("Derivative Plot", plotData, x -> new double[]{x[0], x[2]});
```

Returns: 

![Result](etc/test.676.png)



