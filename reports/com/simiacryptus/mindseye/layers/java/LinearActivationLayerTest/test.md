# LinearActivationLayer
## LinearActivationLayerTest
### Json Serialization
Code from [LayerTestBase.java:83](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L83) executed in 0.00 seconds: 
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
      "class": "com.simiacryptus.mindseye.layers.java.LinearActivationLayer",
      "id": "a864e734-2f23-44db-97c1-504000002c0b",
      "isFrozen": false,
      "name": "LinearActivationLayer/a864e734-2f23-44db-97c1-504000002c0b",
      "weights": {
        "dimensions": [
          2
        ],
        "data": [
          1.0,
          0.0
        ]
      }
    }
```



### Example Input/Output Pair
Code from [LayerTestBase.java:120](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L120) executed in 0.00 seconds: 
```java
    SimpleEval eval = SimpleEval.run(layer, inputPrototype);
    return String.format("--------------------\nInput: \n[%s]\n--------------------\nOutput: \n%s",
      Arrays.stream(inputPrototype).map(t->t.prettyPrint()).reduce((a,b)->a+",\n"+b).get(),
      eval.getOutput().prettyPrint());
```

Returns: 

```
    --------------------
    Input: 
    [[
    	[ [ -1.6 ], [ -0.824 ], [ 0.668 ] ],
    	[ [ -1.328 ], [ 0.888 ], [ -0.968 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ -1.6 ], [ -0.824 ], [ 0.668 ] ],
    	[ [ -1.328 ], [ 0.888 ], [ -0.968 ] ]
    ]
```



### Batch Execution
Code from [LayerTestBase.java:138](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L138) executed in 0.00 seconds: 
```java
    BatchingTester batchingTester = getBatchingTester();
    return batchingTester==null?null:batchingTester.test(layer, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (120#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (120#)}
```



### Differential Validation
Code from [LayerTestBase.java:144](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.00 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ -1.6 ], [ -0.824 ], [ 0.668 ] ],
    	[ [ -1.328 ], [ 0.888 ], [ -0.968 ] ]
    ]
    Inputs Statistics: {meanExponent=3.850091579299392E-4, negative=4, min=-0.968, max=-0.968, mean=-0.5273333333333333, count=6.0, positive=2, stdDev=0.9579378314321284, zeros=0}
    Output: [
    	[ [ -1.6 ], [ -0.824 ], [ 0.668 ] ],
    	[ [ -1.328 ], [ 0.888 ], [ -0.968 ] ]
    ]
    Outputs Statistics: {meanExponent=3.850091579299392E-4, negative=4, min=-0.968, max=-0.968, mean=-0.5273333333333333, count=6.0, positive=2, stdDev=0.9579378314321284, zeros=0}
    Feedback for input 0
    Inputs Values: [
    	[ [ -1.6 ], [ -0.824 ], [ 0.668 ] ],
    	[ [ -1.328 ], [ 0.888 ], [ -0.968 ] ]
    ]
    Value Statistics: {meanExponent=3.850091579299392E-4, negative=4, min=-0.968, max=-0.968, mean=-0.5273333333333333, count=6.0, positive=2, stdDev=0.9579378314321284, zeros=0}
    Implemented Feedback: [ [ 1.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 1.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 1.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 1.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 1.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 1.0 ] ]
    Implemented Statistics: {meanExponent=0.0, negative=0, min=1.0, max=1.0, mean=0.16666666666666666, count=36.0, positive=6, stdDev=0.37267799624996495, zeros=30}
    Measured Feedback: [ [ 0.9999999999998899, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.9999999999998899, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.9999999999998899, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.9999999999998899, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.9999999999998899, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.9999999999998899 ] ]
    Measured Statistics: {meanExponent=-4.7830642341045674E-14, negative=0, min=0.9999999999998899, max=0.9999999999998899, mean=0.1666666666666483, count=36.0, positive=6, stdDev=0.3726779962499239, zeros=30}
    Feedback Error: [ [ -1.1013412404281553E-13, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, -1.1013412404281553E-13, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, -1.1013412404281553E-13, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, -1.1013412404281553E-13, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, -1.1013412404281553E-13, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, -1.1013412404281553E-13 ] ]
    Error Statistics: {meanExponent=-12.958078098036824, negative=6, min=-1.1013412404281553E-13, max=-1.1013412404281553E-13, mean=-1.8355687340469256E-14, count=36.0, positive=0, stdDev=4.104456466702158E-14, zeros=30}
    Learning Gradient for weight set 0
    Weights: [ 1.0, 0.0 ]
    Implemented Gradient: [ [ -1.6, -1.328, -0.824, 0.888, 0.668, -0.968 ], [ 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 ] ]
    Implemented Statistics: {meanExponent=1.925045789649696E-4, negative=4, min=1.0, max=1.0, mean=0.2363333333333333, count=12.0, positive=8, stdDev=1.0207885296290422, zeros=0}
    Measured Gradient: [ [ -1.5999999999993797, -1.32800000000044, -0.82400000000038, 0.8879999999999999, 0.668000000000335, -0.9680000000000799 ], [ 0.9999999999998899, 0.9999999999998899, 0.9999999999998899, 0.9999999999998899, 0.9999999999998899, 0.9999999999998899 ] ]
    Measured Statistics: {meanExponent=1.9250457897683633E-4, negative=4, min=0.9999999999998899, max=0.9999999999998899, mean=0.23633333333328288, count=12.0, positive=8, stdDev=1.020788529629017, zeros=0}
    Gradient Error: [ [ 6.203926261605375E-13, -4.39870362356487E-13, -3.800293413291911E-13, -1.1102230246251565E-16, 3.3495428652940973E-13, -7.993605777301127E-14 ], [ -1.1013412404281553E-13, -1.1013412404281553E-13, -1.1013412404281553E-13, -1.1013412404281553E-13, -1.1013412404281553E-13, -1.1013412404281553E-13 ] ]
    Error Statistics: {meanExponent=-13.021626802914847, negative=10, min=-1.1013412404281553E-13, max=-1.1013412404281553E-13, mean=-5.0450384610674824E-14, count=12.0, positive=2, stdDev=2.713567243579706E-13, zeros=0}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 6.6185e-14 +- 1.2720e-13 [0.0000e+00 - 6.2039e-13] (48#)
    relativeTol: 8.5720e-14 +- 6.9474e-14 [6.2513e-17 - 2.5071e-13] (18#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=6.6185e-14 +- 1.2720e-13 [0.0000e+00 - 6.2039e-13] (48#), relativeTol=8.5720e-14 +- 6.9474e-14 [6.2513e-17 - 2.5071e-13] (18#)}
```



### Performance
Code from [LayerTestBase.java:149](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L149) executed in 0.00 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 0.1661 +- 0.0620 [0.1111 - 0.7011]
    Learning performance: 0.0365 +- 0.0157 [0.0257 - 0.1710]
    
```

### Function Plots
Code from [ActivationLayerTestBase.java:74](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/java/ActivationLayerTestBase.java#L74) executed in 0.01 seconds: 
```java
    return plot("Value Plot", plotData, x -> new double[]{x[0], x[1]});
```

Returns: 

![Result](etc/test.1.png)



Code from [ActivationLayerTestBase.java:78](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/java/ActivationLayerTestBase.java#L78) executed in 0.01 seconds: 
```java
    return plot("Derivative Plot", plotData, x -> new double[]{x[0], x[2]});
```

Returns: 

![Result](etc/test.2.png)



