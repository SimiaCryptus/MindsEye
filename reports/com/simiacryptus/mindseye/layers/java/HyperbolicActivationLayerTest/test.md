# HyperbolicActivationLayer
## HyperbolicActivationLayerTest
### Json Serialization
Code from [LayerTestBase.java:121](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L121) executed in 0.00 seconds: 
```java
    JsonObject json = layer.getJson();
    NNLayer echo = NNLayer.fromJson(json);
    assert (echo != null) : "Failed to deserialize";
    assert (layer != echo) : "Serialization did not copy";
    Assert.assertEquals("Serialization not equal", layer, echo);
    return new GsonBuilder().setPrettyPrinting().create().toJson(json);
```

Returns: 

```
    {
      "class": "com.simiacryptus.mindseye.layers.java.HyperbolicActivationLayer",
      "id": "e2a3bda5-e7e7-4c05-aeb3-4ede00001e63",
      "isFrozen": false,
      "name": "HyperbolicActivationLayer/e2a3bda5-e7e7-4c05-aeb3-4ede00001e63",
      "weights": [
        1.0,
        1.0
      ],
      "negativeMode": 1
    }
```



### Reference Input/Output Pairs
Code from [LayerTestBase.java:148](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L148) executed in 0.00 seconds: 
```java
    SimpleEval eval = SimpleEval.run(layer, input);
    DoubleStatistics error = new DoubleStatistics().accept(eval.getOutput().add(output.scale(-1)).getData());
    return String.format("--------------------\nInput: \n[%s]\n--------------------\nOutput: \n%s\nError: %s",
      Arrays.stream(input).map(t -> t.prettyPrint()).reduce((a, b) -> a + ",\n" + b).get(),
      eval.getOutput().prettyPrint(), error);
```

Returns: 

```
    --------------------
    Input: 
    [[ 0.0 ]]
    --------------------
    Output: 
    [ 0.0 ]
    Error: 0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (1#)
```



### Batch Execution
Code from [LayerTestBase.java:178](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L178) executed in 0.00 seconds: 
```java
    BatchingTester batchingTester = getBatchingTester();
    return batchingTester == null ? null : batchingTester.test(layer, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (120#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (120#)}
```



### Differential Validation
Code from [LayerTestBase.java:186](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L186) executed in 0.00 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ -0.68 ], [ 1.904 ], [ -0.452 ] ],
    	[ [ 1.604 ], [ -0.892 ], [ 1.34 ] ]
    ]
    Inputs Statistics: {meanExponent=0.008331384709191572, negative=3, min=1.34, max=1.34, mean=0.4706666666666666, count=6.0, positive=3, stdDev=1.1638182370494496, zeros=0}
    Output: [
    	[ [ 0.209297316626478 ], [ 1.1506315351542673 ], [ 0.09740785490172255 ] ],
    	[ [ 0.890189408498524 ], [ 0.3400238803842266 ], [ 0.6720047846821493 ] ]
    ]
    Outputs Statistics: {meanExponent=-0.38689031340681135, negative=0, min=0.6720047846821493, max=0.6720047846821493, mean=0.5599257967078947, count=6.0, positive=6, stdDev=0.3776658692088025, zeros=0}
    Feedback for input 0
    Inputs Values: [
    	[ [ -0.68 ], [ 1.904 ], [ -0.452 ] ],
    	[ [ 1.604 ], [ -0.892 ], [ 1.34 ] ]
    ]
    Value Statistics: {meanExponent=0.008331384709191572, negative=3, min=1.34, max=1.34, mean=0.4706666666666666, count=6.0, positive=3, stdDev=1.1638182370494496, zeros=0}
    Implemented Feedback: [ [ -0.5623100214072791, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.8485922060446529, 0.0, 0.
```
...[skipping 1794 bytes](etc/58.txt)...
```
    egative=6, min=0.0, max=0.0, mean=-0.33971097647203014, count=12.0, positive=0, stdDev=0.35823867811397686, zeros=6}
    Measured Gradient: [ [ 0.0, -0.5289755652293504, -0.4649149847191225, 0.0, 0.0, -0.5980054062548756 ], [ -0.8268307460090885, 0.0, 0.0, -0.7461642084782838, -0.9111393947845714, 0.0 ] ]
    Measured Statistics: {meanExponent=-0.1804416464673789, negative=6, min=0.0, max=0.0, mean=-0.33966919212294105, count=12.0, positive=0, stdDev=0.3581962616005247, zeros=6}
    Gradient Error: [ [ 0.0, 7.194477355032536E-5, 6.471247864447971E-5, 0.0, 0.0, 7.900663757998139E-5 ], [ 9.575606043965568E-5, 0.0, 0.0, 9.114908539542554E-5, 9.884315345953798E-5, 0.0 ] ]
    Error Statistics: {meanExponent=-4.083080659704886, negative=0, min=0.0, max=0.0, mean=4.1784349089117136E-5, count=12.0, positive=6, stdDev=4.272194014279163E-5, zeros=6}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 1.2738e-05 +- 2.8108e-05 [0.0000e+00 - 9.8843e-05] (48#)
    relativeTol: 3.9785e-05 +- 2.5687e-05 [2.8387e-06 - 6.9591e-05] (12#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=1.2738e-05 +- 2.8108e-05 [0.0000e+00 - 9.8843e-05] (48#), relativeTol=3.9785e-05 +- 2.5687e-05 [2.8387e-06 - 6.9591e-05] (12#)}
```



### Performance
Code from [LayerTestBase.java:192](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L192) executed in 0.00 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 0.1653 +- 0.0420 [0.1054 - 0.3762]
    Learning performance: 0.0553 +- 0.0246 [0.0399 - 0.2223]
    
```

### Function Plots
Code from [ActivationLayerTestBase.java:103](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/java/ActivationLayerTestBase.java#L103) executed in 0.00 seconds: 
```java
    return plot("Value Plot", plotData, x -> new double[]{x[0], x[1]});
```

Returns: 

![Result](etc/test.22.png)



Code from [ActivationLayerTestBase.java:107](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/java/ActivationLayerTestBase.java#L107) executed in 0.00 seconds: 
```java
    return plot("Derivative Plot", plotData, x -> new double[]{x[0], x[2]});
```

Returns: 

![Result](etc/test.23.png)



