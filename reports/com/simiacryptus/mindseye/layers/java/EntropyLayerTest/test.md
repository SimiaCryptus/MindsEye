# EntropyLayer
## EntropyLayerTest
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
      "class": "com.simiacryptus.mindseye.layers.java.EntropyLayer",
      "id": "e2a3bda5-e7e7-4c05-aeb3-4ede00001e47",
      "isFrozen": true,
      "name": "EntropyLayer/e2a3bda5-e7e7-4c05-aeb3-4ede00001e47"
    }
```



### Example Input/Output Pair
Code from [LayerTestBase.java:159](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L159) executed in 0.00 seconds: 
```java
    SimpleEval eval = SimpleEval.run(layer, inputPrototype);
    return String.format("--------------------\nInput: \n[%s]\n--------------------\nOutput: \n%s",
      Arrays.stream(inputPrototype).map(t -> t.prettyPrint()).reduce((a, b) -> a + ",\n" + b).get(),
      eval.getOutput().prettyPrint());
```

Returns: 

```
    --------------------
    Input: 
    [[
    	[ [ 0.708 ], [ 0.444 ], [ -1.204 ] ],
    	[ [ 1.632 ], [ -0.56 ], [ 1.824 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 0.2444803191841995 ], [ 0.3604972381481611 ], [ 0.22352181365150162 ] ],
    	[ [ -0.7993638106764055 ], [ -0.3246983573416476 ], [ -1.0962821703735028 ] ]
    ]
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
    	[ [ 0.708 ], [ 0.444 ], [ -1.204 ] ],
    	[ [ 1.632 ], [ -0.56 ], [ 1.824 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.033337378309560885, negative=2, min=1.824, max=1.824, mean=0.474, count=6.0, positive=4, stdDev=1.0881145773002645, zeros=0}
    Output: [
    	[ [ 0.2444803191841995 ], [ 0.3604972381481611 ], [ 0.22352181365150162 ] ],
    	[ [ -0.7993638106764055 ], [ -0.3246983573416476 ], [ -1.0962821703735028 ] ]
    ]
    Outputs Statistics: {meanExponent=-0.37523121941274457, negative=3, min=-1.0962821703735028, max=-1.0962821703735028, mean=-0.23197416123461564, count=6.0, positive=3, stdDev=0.5572349803021766, zeros=0}
    Feedback for input 0
    Inputs Values: [
    	[ [ 0.708 ], [ 0.444 ], [ -1.204 ] ],
    	[ [ 1.632 ], [ -0.56 ], [ 1.824 ] ]
    ]
    Value Statistics: {meanExponent=-0.033337378309560885, negative=2, min=1.824, max=1.824, mean=0.474, count=6.0, positive=4, stdDev=1.0881145773002645, zeros=0}
    Implemented Feedback: [ [ -0.6546888147115826, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, -1.4898062565419152, 0.0, 0.0, 0.0, 0.
```
...[skipping 653 bytes](etc/53.txt)...
```
    0.0, 0.0, -1.1856078174973805, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, -1.6010593034310183 ] ]
    Measured Statistics: {meanExponent=-0.13908476435857245, negative=6, min=-1.6010593034310183, max=-1.6010593034310183, mean=-0.1538760430078553, count=36.0, positive=0, stdDev=0.4074605790751727, zeros=30}
    Feedback Error: [ [ -7.061814439546232E-5, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, -3.063662958813218E-5, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, -1.1260415918568079E-4, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 8.929102942023537E-5, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 4.1529389248839976E-5, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, -2.7411778878771997E-5 ] ]
    Error Statistics: {meanExponent=-4.267697947767291, negative=4, min=-2.7411778878771997E-5, max=-2.7411778878771997E-5, mean=-3.068063704971443E-6, count=36.0, positive=2, stdDev=2.8242667311661595E-5, zeros=30}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 1.0336e-05 +- 2.6462e-05 [0.0000e+00 - 1.1260e-04] (36#)
    relativeTol: 8.2638e-05 +- 1.0270e-04 [8.5606e-06 - 2.9928e-04] (6#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=1.0336e-05 +- 2.6462e-05 [0.0000e+00 - 1.1260e-04] (36#), relativeTol=8.2638e-05 +- 1.0270e-04 [8.5606e-06 - 2.9928e-04] (6#)}
```



### Performance
Code from [LayerTestBase.java:192](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L192) executed in 0.00 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 0.1463 +- 0.0999 [0.0912 - 0.8122]
    Learning performance: 0.0101 +- 0.0810 [0.0000 - 0.8151]
    
```

### Function Plots
Code from [ActivationLayerTestBase.java:103](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/java/ActivationLayerTestBase.java#L103) executed in 0.01 seconds: 
```java
    return plot("Value Plot", plotData, x -> new double[]{x[0], x[1]});
```

Returns: 

![Result](etc/test.16.png)



Code from [ActivationLayerTestBase.java:107](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/java/ActivationLayerTestBase.java#L107) executed in 0.01 seconds: 
```java
    return plot("Derivative Plot", plotData, x -> new double[]{x[0], x[2]});
```

Returns: 

![Result](etc/test.17.png)



