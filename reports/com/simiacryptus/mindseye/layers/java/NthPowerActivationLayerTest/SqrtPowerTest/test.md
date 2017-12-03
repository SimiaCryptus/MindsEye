# NthPowerActivationLayer
## SqrtPowerTest
### Json Serialization
Code from [LayerTestBase.java:84](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L84) executed in 0.11 seconds: 
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
      "class": "com.simiacryptus.mindseye.layers.java.NthPowerActivationLayer",
      "id": "2435ff03-3c85-4419-8a0e-aa4e00000001",
      "isFrozen": false,
      "name": "NthPowerActivationLayer/2435ff03-3c85-4419-8a0e-aa4e00000001",
      "power": 0.5
    }
```



### Example Input/Output Pair
Code from [LayerTestBase.java:121](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L121) executed in 0.03 seconds: 
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
    	[ [ -0.94 ], [ -0.892 ], [ 1.42 ] ],
    	[ [ 1.296 ], [ 1.18 ], [ 1.46 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 0.0 ], [ 0.0 ], [ 1.1916375287812984 ] ],
    	[ [ 1.1384199576606167 ], [ 1.0862780491200215 ], [ 1.2083045973594573 ] ]
    ]
```



### Differential Validation
Code from [LayerTestBase.java:139](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L139) executed in 0.06 seconds: 
```java
    getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ -0.94 ], [ -0.892 ], [ 1.42 ] ],
    	[ [ 1.296 ], [ 1.18 ], [ 1.46 ] ]
    ]
    Inputs Statistics: {meanExponent=0.07077015283066919, negative=2, min=1.46, max=1.46, mean=0.5873333333333334, count=6.0, positive=4, stdDev=1.0668893517553208, zeros=0}
    Output: [
    	[ [ 0.0 ], [ 0.0 ], [ 1.1916375287812984 ] ],
    	[ [ 1.1384199576606167 ], [ 1.0862780491200215 ], [ 1.2083045973594573 ] ]
    ]
    Outputs Statistics: {meanExponent=0.06264102612602418, negative=0, min=1.2083045973594573, max=1.2083045973594573, mean=0.7707733554868991, count=6.0, positive=4, stdDev=0.5464202605487217, zeros=2}
    Feedback for input 0
    Inputs Values: [
    	[ [ -0.94 ], [ -0.892 ], [ 1.42 ] ],
    	[ [ 1.296 ], [ 1.18 ], [ 1.46 ] ]
    ]
    Value Statistics: {meanExponent=0.07077015283066919, negative=2, min=1.46, max=1.46, mean=0.5873333333333334, count=6.0, positive=4, stdDev=1.0668893517553208, zeros=0}
    Implemented Feedback: [ [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.43920523057894156, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.46028730894916176, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.4195906791483446, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.413802944301184 ] ]
    Implemented Statistics: {meanExponent=-0.3636710217900054, negative=0, min=0.413802944301184, max=0.413802944301184, mean=0.04813572674937866, count=36.0, positive=4, stdDev=0.13628413142923843, zeros=32}
    Measured Feedback: [ [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.439196758581506, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.46027755751243404, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.4195832922504472, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.4137958588756874 ] ]
    Measured Statistics: {meanExponent=-0.3636791868901163, negative=0, min=0.4137958588756874, max=0.4137958588756874, mean=0.04813481853389096, count=36.0, positive=4, stdDev=0.13628154960575425, zeros=32}
    Feedback Error: [ [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, -8.471997435588463E-6, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, -9.751436727711837E-6, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, -7.386897897365685E-6, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, -7.085425496600184E-6 ] ]
    Error Statistics: {meanExponent=-5.091029386302124, negative=4, min=-7.085425496600184E-6, max=-7.085425496600184E-6, mean=-9.082154877018381E-7, count=36.0, positive=0, stdDev=2.5924033273602662E-6, zeros=32}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 9.0822e-07 +- 2.5924e-06 [0.0000e+00 - 9.7514e-06] (36#)
    relativeTol: 9.4004e-06 +- 7.9734e-07 [8.5614e-06 - 1.0593e-05] (4#)
    
```

### Performance
Code from [LayerTestBase.java:144](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.04 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 1.0911 +- 1.4537 [0.7238 - 15.4772]
    Learning performance: 0.0288 +- 0.0137 [0.0228 - 0.1482]
    
```

### Function Plots
Code from [ActivationLayerTestBase.java:73](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/java/ActivationLayerTestBase.java#L73) executed in 0.45 seconds: 
```java
    return plot("Value Plot", plotData, x -> new double[]{x[0], x[1]});
```

Returns: 

![Result](etc/test.1.png)



Code from [ActivationLayerTestBase.java:77](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/java/ActivationLayerTestBase.java#L77) executed in 0.02 seconds: 
```java
    return plot("Derivative Plot", plotData, x -> new double[]{x[0], x[2]});
```

Returns: 

![Result](etc/test.2.png)



