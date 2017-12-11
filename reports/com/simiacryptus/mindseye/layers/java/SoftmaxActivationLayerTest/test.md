# SoftmaxActivationLayer
## SoftmaxActivationLayerTest
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
      "class": "com.simiacryptus.mindseye.layers.java.SoftmaxActivationLayer",
      "id": "e2a3bda5-e7e7-4c05-aeb3-4ede00003660",
      "isFrozen": false,
      "name": "SoftmaxActivationLayer/e2a3bda5-e7e7-4c05-aeb3-4ede00003660"
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
    [[ 1.656, 1.164, 1.688, 1.424 ]]
    --------------------
    Output: 
    [ 0.2909627554919301, 0.17789531707688272, 0.3004241384372308, 0.23071778899395645 ]
```



### Batch Execution
Code from [LayerTestBase.java:178](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L178) executed in 0.00 seconds: 
```java
    BatchingTester batchingTester = getBatchingTester();
    return batchingTester == null ? null : batchingTester.test(layer, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (80#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (40#)}
```



### Differential Validation
Code from [LayerTestBase.java:186](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L186) executed in 0.00 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [ 1.656, 1.164, 1.688, 1.424 ]
    Inputs Statistics: {meanExponent=0.1664739360883012, negative=0, min=1.424, max=1.424, mean=1.483, count=4.0, positive=4, stdDev=0.210473276213394, zeros=0}
    Output: [ 0.2909627554919301, 0.17789531707688272, 0.3004241384372308, 0.23071778899395645 ]
    Outputs Statistics: {meanExponent=-0.6112955444229864, negative=0, min=0.23071778899395645, max=0.23071778899395645, mean=0.25, count=4.0, positive=4, stdDev=0.04947582752050136, zeros=0}
    Feedback for input 0
    Inputs Values: [ 1.656, 1.164, 1.688, 1.424 ]
    Value Statistics: {meanExponent=0.1664739360883012, negative=0, min=1.424, max=1.424, mean=1.483, count=4.0, positive=4, stdDev=0.210473276213394, zeros=0}
    Implemented Feedback: [ [ 0.2063034304084734, -0.05176091164580041, -0.08741223513598576, -0.06713028362668727 ], [ -0.05176091164580041, 0.1462485732389981, -0.053444047364840486, -0.04104361422835721 ], [ -0.08741223513598576, -0.053444047364840486, 0.2101694754814784, -0.06931319298065218 ], [ -0.0671302836266872
```
...[skipping 656 bytes](etc/88.txt)...
```
    22797486 ] ]
    Measured Statistics: {meanExponent=-1.1012240533201565, negative=12, min=0.17749187022797486, max=0.17749187022797486, mean=-1.214306433183765E-13, count=16.0, positive=4, stdDev=0.10839020632819855, zeros=0}
    Feedback Error: [ [ 4.3124281133011255E-6, -1.081975257409551E-6, -1.8272064866431403E-6, -1.4032463692137398E-6 ], [ -1.667254027303E-6, 4.710764807458467E-6, -1.7214685306687905E-6, -1.3220425270285552E-6 ], [ -1.744499867312177E-6, -1.0665909895846148E-6, 4.194383609601449E-6, -1.3832938629138036E-6 ], [ -1.8076920128562435E-6, -1.1052270725495084E-6, -1.8664737479051619E-6, 4.779392278220218E-6 ] ]
    Error Statistics: {meanExponent=-5.711604973104892, negative=12, min=4.779392278220218E-6, max=4.779392278220218E-6, mean=-1.2142543914794857E-13, count=16.0, positive=4, stdDev=2.6131951947452555E-6, zeros=0}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 2.2496e-06 +- 1.3297e-06 [1.0666e-06 - 4.7794e-06] (16#)
    relativeTol: 1.2500e-05 +- 2.4739e-06 [9.9785e-06 - 1.6105e-05] (16#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=2.2496e-06 +- 1.3297e-06 [1.0666e-06 - 4.7794e-06] (16#), relativeTol=1.2500e-05 +- 2.4739e-06 [9.9785e-06 - 1.6105e-05] (16#)}
```



### Performance
Code from [LayerTestBase.java:192](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L192) executed in 0.01 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 0.2781 +- 0.1010 [0.1767 - 0.8350]
    Learning performance: 0.0019 +- 0.0020 [0.0000 - 0.0142]
    
```

