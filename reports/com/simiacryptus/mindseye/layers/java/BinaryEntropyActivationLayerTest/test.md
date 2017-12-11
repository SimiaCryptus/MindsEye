# BinaryEntropyActivationLayer
## BinaryEntropyActivationLayerTest
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
      "class": "com.simiacryptus.mindseye.layers.java.BinaryEntropyActivationLayer",
      "id": "e2a3bda5-e7e7-4c05-aeb3-4ede00001e20",
      "isFrozen": true,
      "name": "BinaryEntropyActivationLayer/e2a3bda5-e7e7-4c05-aeb3-4ede00001e20"
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
    	[ [ 0.2999681884640649 ], [ 0.1664723725916254 ], [ 0.18166615832861965 ] ],
    	[ [ 0.1665291042609699 ], [ 0.20268159066502567 ], [ 0.2438054241289117 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ -0.6108373457990499 ], [ -0.45024836867282325 ], [ -0.4739105802674475 ] ],
    	[ [ -0.4503397425759544 ], [ -0.5040975006863339 ], [ -0.5554268083509369 ] ]
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
    	[ [ 0.2999681884640649 ], [ 0.1664723725916254 ], [ 0.18166615832861965 ] ],
    	[ [ 0.1665291042609699 ], [ 0.20268159066502567 ], [ 0.2438054241289117 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.6878267974282036, negative=0, min=0.2438054241289117, max=0.2438054241289117, mean=0.2101871397398695, count=6.0, positive=6, stdDev=0.048092983430636525, zeros=0}
    Output: [
    	[ [ -0.6108373457990499 ], [ -0.45024836867282325 ], [ -0.4739105802674475 ] ],
    	[ [ -0.4503397425759544 ], [ -0.5040975006863339 ], [ -0.5554268083509369 ] ]
    ]
    Outputs Statistics: {meanExponent=-0.2973740348148793, negative=6, min=-0.5554268083509369, max=-0.5554268083509369, mean=-0.507476724392091, count=6.0, positive=0, stdDev=0.05868056632158456, zeros=0}
    Feedback for input 0
    Inputs Values: [
    	[ [ 0.2999681884640649 ], [ 0.1664723725916254 ], [ 0.18166615832861965 ] ],
    	[ [ 0.1665291042609699 ], [ 0.20268159066502567 ], [ 0.2438054241289117 ] ]
    ]
    Value Statistics: {meanExponent=-0.6878267974282036, negative=0, min=0.24380542412
```
...[skipping 873 bytes](etc/48.txt)...
```
    , [ 0.0, 0.0, 0.0, 0.0, -1.504763383058938, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, -1.131657076479664 ] ]
    Measured Statistics: {meanExponent=0.11824583678715823, negative=6, min=-1.131657076479664, max=-1.131657076479664, mean=-0.2242634983848974, count=36.0, positive=0, stdDev=0.5140552965173294, zeros=30}
    Feedback Error: [ [ 2.380945500343179E-4, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 3.601802329602499E-4, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 3.602784284870708E-4, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 3.0936462484154603E-4, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 3.3628188826817507E-4, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 2.711770028422933E-4 ] ]
    Error Statistics: {meanExponent=-3.5099442376184586, negative=0, min=2.711770028422933E-4, max=2.711770028422933E-4, mean=5.209379798426814E-5, count=36.0, positive=6, stdDev=1.1795159211541686E-4, zeros=30}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 5.2094e-05 +- 1.1795e-04 [0.0000e+00 - 3.6028e-04] (36#)
    relativeTol: 1.1811e-04 +- 1.0408e-05 [1.1173e-04 - 1.4050e-04] (6#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=5.2094e-05 +- 1.1795e-04 [0.0000e+00 - 3.6028e-04] (36#), relativeTol=1.1811e-04 +- 1.0408e-05 [1.1173e-04 - 1.4050e-04] (6#)}
```



### Performance
Code from [LayerTestBase.java:192](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L192) executed in 0.00 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 0.1658 +- 0.1087 [0.0941 - 0.8521]
    Learning performance: 0.0024 +- 0.0038 [0.0000 - 0.0285]
    
```

### Function Plots
Code from [ActivationLayerTestBase.java:103](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/java/ActivationLayerTestBase.java#L103) executed in 0.01 seconds: 
```java
    return plot("Value Plot", plotData, x -> new double[]{x[0], x[1]});
```

Returns: 

![Result](etc/test.14.png)



Code from [ActivationLayerTestBase.java:107](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/java/ActivationLayerTestBase.java#L107) executed in 0.01 seconds: 
```java
    return plot("Derivative Plot", plotData, x -> new double[]{x[0], x[2]});
```

Returns: 

![Result](etc/test.15.png)



