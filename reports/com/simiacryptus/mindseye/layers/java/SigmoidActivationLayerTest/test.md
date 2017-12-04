# SigmoidActivationLayer
## SigmoidActivationLayerTest
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
      "class": "com.simiacryptus.mindseye.layers.java.SigmoidActivationLayer",
      "id": "370a9587-74a1-4959-b406-fa4500002c8d",
      "isFrozen": true,
      "name": "SigmoidActivationLayer/370a9587-74a1-4959-b406-fa4500002c8d",
      "balanced": true
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
    	[ [ -1.388 ], [ 0.8 ], [ -0.948 ] ],
    	[ [ -1.604 ], [ -0.412 ], [ -1.612 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ -0.6005455251687 ], [ 0.379948962255225 ], [ -0.44142556810013744 ] ],
    	[ [ -0.6651533961531864 ], [ -0.20313468846754057 ], [ -0.6673777499823074 ] ]
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
    	[ [ -1.388 ], [ 0.8 ], [ -0.948 ] ],
    	[ [ -1.604 ], [ -0.412 ], [ -1.612 ] ]
    ]
    Inputs Statistics: {meanExponent=0.008292401316532846, negative=5, min=-1.612, max=-1.612, mean=-0.8606666666666666, count=6.0, positive=1, stdDev=0.8526301790472951, zeros=0}
    Output: [
    	[ [ -0.6005455251687 ], [ 0.379948962255225 ], [ -0.44142556810013744 ] ],
    	[ [ -0.6651533961531864 ], [ -0.20313468846754057 ], [ -0.6673777499823074 ] ]
    ]
    Outputs Statistics: {meanExponent=-0.34029894812989814, negative=5, min=-0.6673777499823074, max=-0.6673777499823074, mean=-0.3662813276027745, count=6.0, positive=1, stdDev=0.3705820672566255, zeros=0}
    Feedback for input 0
    Inputs Values: [
    	[ [ -1.388 ], [ 0.8 ], [ -0.948 ] ],
    	[ [ -1.604 ], [ -0.412 ], [ -1.612 ] ]
    ]
    Value Statistics: {meanExponent=0.008292401316532846, negative=5, min=-1.612, max=-1.612, mean=-0.8606666666666666, count=6.0, positive=1, stdDev=0.8526301790472951, zeros=0}
    Implemented Feedback: [ [ 0.3196725360999252, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.27878547979294116, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.42781939304058886, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.47936814917059756, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.4025717339137355, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.2773034694142764 ] ]
    Implemented Statistics: {meanExponent=-0.4483827166556014, negative=0, min=0.2773034694142764, max=0.2773034694142764, mean=0.060708910039779566, count=36.0, positive=6, stdDev=0.13934527075861047, zeros=30}
    Measured Feedback: [ [ 0.3196821350170964, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.27879475162473355, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.4278112653599564, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.47937301763556484, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.402580619046633, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.27731272279951646 ] ]
    Measured Statistics: {meanExponent=-0.4483747631727127, negative=0, min=0.27731272279951646, max=0.27731272279951646, mean=0.06070984754120835, count=36.0, positive=6, stdDev=0.13934698594015316, zeros=30}
    Feedback Error: [ [ 9.598917171227406E-6, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 9.271831792390106E-6, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, -8.127680632463719E-6, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 4.868464967278108E-6, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 8.885132897484826E-6, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 9.253385240048928E-6 ] ]
    Error Statistics: {meanExponent=-5.089714826825322, negative=1, min=9.253385240048928E-6, max=9.253385240048928E-6, mean=9.375014287768238E-7, count=36.0, positive=5, stdDev=3.3366534415805748E-6, zeros=30}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 1.3890e-06 +- 3.1753e-06 [0.0000e+00 - 9.5989e-06] (36#)
    relativeTol: 1.2323e-05 +- 4.2214e-06 [5.0780e-06 - 1.6684e-05] (6#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=1.3890e-06 +- 3.1753e-06 [0.0000e+00 - 9.5989e-06] (36#), relativeTol=1.2323e-05 +- 4.2214e-06 [5.0780e-06 - 1.6684e-05] (6#)}
```



### Performance
Code from [LayerTestBase.java:149](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L149) executed in 0.00 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 0.1371 +- 0.0375 [0.0912 - 0.3277]
    Learning performance: 0.0013 +- 0.0015 [0.0000 - 0.0057]
    
```

### Function Plots
Code from [ActivationLayerTestBase.java:74](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/java/ActivationLayerTestBase.java#L74) executed in 0.00 seconds: 
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



