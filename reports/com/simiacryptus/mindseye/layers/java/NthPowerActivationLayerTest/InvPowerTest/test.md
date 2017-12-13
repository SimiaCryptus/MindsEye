# NthPowerActivationLayer
## InvPowerTest
### Json Serialization
Code from [StandardLayerTests.java:69](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L69) executed in 0.00 seconds: 
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
      "class": "com.simiacryptus.mindseye.layers.java.NthPowerActivationLayer",
      "id": "a559e487-e29f-46ea-907d-dfff07481dde",
      "isFrozen": false,
      "name": "NthPowerActivationLayer/a559e487-e29f-46ea-907d-dfff07481dde",
      "power": -1.0
    }
```



### Example Input/Output Pair
Code from [StandardLayerTests.java:153](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L153) executed in 0.00 seconds: 
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
    	[ [ -0.54 ], [ 0.276 ], [ -1.912 ] ],
    	[ [ 0.224 ], [ 0.5 ], [ 1.456 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ -1.8518518518518516 ], [ 3.623188405797101 ], [ -0.5230125523012552 ] ],
    	[ [ 4.464285714285714 ], [ 2.0 ], [ 0.6868131868131868 ] ]
    ]
    --------------------
    Derivative: 
    [
    	[ [ -3.4293552812071324 ], [ -13.12749422390254 ], [ -0.27354212986467324 ] ],
    	[ [ -19.92984693877551 ], [ -4.0 ], [ -0.47171235358048547 ] ]
    ]
```



### Batch Execution
Code from [StandardLayerTests.java:102](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L102) executed in 0.00 seconds: 
```java
    return getBatchingTester().test(layer, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (120#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (120#)}
```



### Differential Validation
Code from [StandardLayerTests.java:110](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L110) executed in 0.00 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ 0.572 ], [ 0.56 ], [ 0.744 ] ],
    	[ [ -1.572 ], [ 0.464 ], [ -1.9 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.08018648090729964, negative=2, min=-1.9, max=-1.9, mean=-0.18866666666666665, count=6.0, positive=4, stdDev=1.1013044790409035, zeros=0}
    Output: [
    	[ [ 1.7482517482517483 ], [ 1.7857142857142856 ], [ 1.3440860215053763 ] ],
    	[ [ -0.6361323155216285 ], [ 2.155172413793103 ], [ -0.5263157894736842 ] ]
    ]
    Outputs Statistics: {meanExponent=0.08018648090729961, negative=2, min=-0.5263157894736842, max=-0.5263157894736842, mean=0.9784627273782002, count=6.0, positive=4, stdDev=1.1279651170537306, zeros=0}
    Feedback for input 0
    Inputs Values: [
    	[ [ 0.572 ], [ 0.56 ], [ 0.744 ] ],
    	[ [ -1.572 ], [ 0.464 ], [ -1.9 ] ]
    ]
    Value Statistics: {meanExponent=-0.08018648090729964, negative=2, min=-1.9, max=-1.9, mean=-0.18866666666666665, count=6.0, positive=4, stdDev=1.1013044790409035, zeros=0}
    Implemented Feedback: [ [ -3.0563841752652947, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, -0.4046643228509087, 0.0
```
...[skipping 654 bytes](etc/89.txt)...
```
    0.0, 0.0, 0.0, -1.8063244476618223, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, -0.2770228904014349 ] ]
    Measured Statistics: {meanExponent=0.1603304725592727, negative=6, min=-0.2770228904014349, max=-0.2770228904014349, mean=-0.3715516902567168, count=36.0, positive=0, stdDev=1.0488214615263298, zeros=30}
    Feedback Error: [ [ 5.342395006073808E-4, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, -2.5743642443010195E-5, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 5.693225342855435E-4, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.001000811924077638, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 2.4278554432877186E-4, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, -1.4580152127408041E-5 ] ]
    Error Statistics: {meanExponent=-3.7594830473644607, negative=2, min=-1.4580152127408041E-5, max=-1.4580152127408041E-5, mean=6.407876968691433E-5, count=36.0, positive=4, stdDev=2.056943148516643E-4, zeros=30}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 6.6319e-05 +- 2.0498e-04 [0.0000e+00 - 1.0008e-03] (36#)
    relativeTol: 6.8292e-05 +- 3.0157e-05 [2.6316e-05 - 1.0775e-04] (6#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=6.6319e-05 +- 2.0498e-04 [0.0000e+00 - 1.0008e-03] (36#), relativeTol=6.8292e-05 +- 3.0157e-05 [2.6316e-05 - 1.0775e-04] (6#)}
```



### Performance
Code from [StandardLayerTests.java:120](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L120) executed in 0.00 seconds: 
```java
    getPerformanceTester().test(layer, permPrototype);
```
Logging: 
```
    Evaluation performance: 0.000207s +- 0.000047s [0.000172s - 0.000298s]
    Learning performance: 0.000042s +- 0.000004s [0.000038s - 0.000050s]
    
```

### Function Plots
Code from [ActivationLayerTestBase.java:103](../../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/java/ActivationLayerTestBase.java#L103) executed in 0.00 seconds: 
```java
    return plot("Value Plot", plotData, x -> new double[]{x[0], x[1]});
```

Returns: 

![Result](etc/test.30.png)



Code from [ActivationLayerTestBase.java:107](../../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/java/ActivationLayerTestBase.java#L107) executed in 0.00 seconds: 
```java
    return plot("Derivative Plot", plotData, x -> new double[]{x[0], x[2]});
```

Returns: 

![Result](etc/test.31.png)



