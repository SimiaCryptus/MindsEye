# LogActivationLayer
## LogActivationLayerTest
### Json Serialization
Code from [LayerTestBase.java:83](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L83) executed in 0.00 seconds: 
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
      "class": "com.simiacryptus.mindseye.layers.java.LogActivationLayer",
      "id": "370a9587-74a1-4959-b406-fa4500002c14",
      "isFrozen": true,
      "name": "LogActivationLayer/370a9587-74a1-4959-b406-fa4500002c14"
    }
```



### Example Input/Output Pair
Code from [LayerTestBase.java:120](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L120) executed in 0.00 seconds: 
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
    	[ [ -0.836 ], [ -1.052 ], [ 0.852 ] ],
    	[ [ -0.876 ], [ -0.28 ], [ 1.18 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ -0.17912666589743548 ], [ 0.050693114315518165 ], [ -0.16016875215282134 ] ],
    	[ [ -0.13238918804574562 ], [ -1.2729656758128873 ], [ 0.16551443847757333 ] ]
    ]
```



### Batch Execution
Code from [LayerTestBase.java:138](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L138) executed in 0.00 seconds: 
```java
    BatchingTester batchingTester = getBatchingTester();
    return batchingTester==null?null:batchingTester.test(layer, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (120#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (120#)}
```



### Differential Validation
Code from [LayerTestBase.java:144](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.00 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ -0.836 ], [ -1.052 ], [ 0.852 ] ],
    	[ [ -0.876 ], [ -0.28 ], [ 1.18 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.11063237386002299, negative=4, min=1.18, max=1.18, mean=-0.16866666666666677, count=6.0, positive=2, stdDev=0.8754988419308286, zeros=0}
    Output: [
    	[ [ -0.17912666589743548 ], [ 0.050693114315518165 ], [ -0.16016875215282134 ] ],
    	[ [ -0.13238918804574562 ], [ -1.2729656758128873 ], [ 0.16551443847757333 ] ]
    ]
    Outputs Statistics: {meanExponent=-0.7319679826023934, negative=4, min=0.16551443847757333, max=0.16551443847757333, mean=-0.2547404548526331, count=6.0, positive=2, stdDev=0.47193176160388733, zeros=0}
    Feedback for input 0
    Inputs Values: [
    	[ [ -0.836 ], [ -1.052 ], [ 0.852 ] ],
    	[ [ -0.876 ], [ -0.28 ], [ 1.18 ] ]
    ]
    Value Statistics: {meanExponent=-0.11063237386002299, negative=4, min=1.18, max=1.18, mean=-0.16866666666666677, count=6.0, positive=2, stdDev=0.8754988419308286, zeros=0}
    Implemented Feedback: [ [ -1.1961722488038278, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, -1.1415525114155252, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, -0.9505703422053231, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, -3.571428571428571, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 1.1737089201877935, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.8474576271186441 ] ]
    Implemented Statistics: {meanExponent=0.11063237386002299, negative=4, min=0.8474576271186441, max=0.8474576271186441, mean=-0.13440436462630026, count=36.0, positive=2, stdDev=0.703919780471637, zeros=30}
    Measured Feedback: [ [ -1.1961722612419123, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, -1.1415525219549139, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, -0.9505703411905042, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, -3.5714286417132257, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 1.1737089189445271, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.8474576207451179 ] ]
    Measured Statistics: {meanExponent=0.11063237600709042, negative=4, min=0.8474576207451179, max=0.8474576207451179, mean=-0.1344043674003031, count=36.0, positive=2, stdDev=0.7039197906005777, zeros=30}
    Feedback Error: [ [ -1.2438084562305107E-8, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, -1.0539388695107732E-8, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 1.0148188955838577E-9, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, -7.028465454084198E-8, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, -1.2432663787365072E-9, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, -6.373526262315465E-9 ] ]
    Error Statistics: {meanExponent=-8.188373007436114, negative=5, min=-6.373526262315465E-9, max=-6.373526262315465E-9, mean=-2.7740028206589704E-9, count=36.0, positive=1, stdDev=1.175194006667033E-8, zeros=30}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 2.8304e-09 +- 1.1738e-08 [0.0000e+00 - 7.0285e-08] (36#)
    relativeTol: 4.0798e-09 +- 3.1622e-09 [5.2963e-10 - 9.8399e-09] (6#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=2.8304e-09 +- 1.1738e-08 [0.0000e+00 - 7.0285e-08] (36#), relativeTol=4.0798e-09 +- 3.1622e-09 [5.2963e-10 - 9.8399e-09] (6#)}
```



### Performance
Code from [LayerTestBase.java:149](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L149) executed in 0.00 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 0.1408 +- 0.0896 [0.0940 - 0.9518]
    Learning performance: 0.0039 +- 0.0203 [0.0000 - 0.2023]
    
```

### Function Plots
Code from [ActivationLayerTestBase.java:74](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/java/ActivationLayerTestBase.java#L74) executed in 0.01 seconds: 
```java
    return plot("Value Plot", plotData, x -> new double[]{x[0], x[1]});
```

Returns: 

![Result](etc/test.1.png)



Code from [ActivationLayerTestBase.java:78](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/java/ActivationLayerTestBase.java#L78) executed in 0.00 seconds: 
```java
    return plot("Derivative Plot", plotData, x -> new double[]{x[0], x[2]});
```

Returns: 

![Result](etc/test.2.png)



