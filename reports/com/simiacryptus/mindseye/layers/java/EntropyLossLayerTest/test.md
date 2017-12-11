# EntropyLossLayer
## EntropyLossLayerTest
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
      "class": "com.simiacryptus.mindseye.layers.java.EntropyLossLayer",
      "id": "e2a3bda5-e7e7-4c05-aeb3-4ede00001e4e",
      "isFrozen": false,
      "name": "EntropyLossLayer/e2a3bda5-e7e7-4c05-aeb3-4ede00001e4e"
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
    [[ 0.6288984682189375, 0.3233357484145689, 0.18529431961000165, 0.7013051289208733 ],
    [ 0.10074258316009665, 0.5017369257735899, 0.8527438317391771, 0.6281794321892405 ]]
    --------------------
    Output: 
    [ 2.2736657005464607 ]
```



### Batch Execution
Code from [LayerTestBase.java:178](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L178) executed in 0.00 seconds: 
```java
    BatchingTester batchingTester = getBatchingTester();
    return batchingTester == null ? null : batchingTester.test(layer, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (90#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (90#)}
```



### Differential Validation
Code from [LayerTestBase.java:186](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L186) executed in 0.00 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [ 0.6288984682189375, 0.3233357484145689, 0.18529431961000165, 0.7013051289208733 ],
    [ 0.10074258316009665, 0.5017369257735899, 0.8527438317391771, 0.6281794321892405 ]
    Inputs Statistics: {meanExponent=-0.39449915469773533, negative=0, min=0.7013051289208733, max=0.7013051289208733, mean=0.45970841629109527, count=4.0, positive=4, stdDev=0.21265867383316603, zeros=0},
    {meanExponent=-0.39185213842730376, negative=0, min=0.6281794321892405, max=0.6281794321892405, mean=0.520850693215526, count=4.0, positive=4, stdDev=0.27318888788205514, zeros=0}
    Output: [ 2.2736657005464607 ]
    Outputs Statistics: {meanExponent=0.35672661027222596, negative=0, min=2.2736657005464607, max=2.2736657005464607, mean=2.2736657005464607, count=1.0, positive=1, stdDev=0.0, zeros=0}
    Feedback for input 0
    Inputs Values: [ 0.6288984682189375, 0.3233357484145689, 0.18529431961000165, 0.7013051289208733 ]
    Value Statistics: {meanExponent=-0.39449915469773533, negative=0, min=0.7013051289208733, max=0.7013051289208733, mean=0.459
```
...[skipping 1541 bytes](etc/54.txt)...
```
    : {meanExponent=-0.12603929393152172, negative=0, min=0.35481221001831365, max=0.35481221001831365, mean=0.9083678728057574, count=4.0, positive=4, stdDev=0.5378661878803489, zeros=0}
    Measured Feedback: [ [ 0.4637854544853326 ], [ 1.1290639978511763 ], [ 1.6858098028649238 ], [ 0.35481217963706513 ] ]
    Measured Statistics: {meanExponent=-0.12603930558686707, negative=0, min=0.35481217963706513, max=0.35481217963706513, mean=0.9083678587096244, count=4.0, positive=4, stdDev=0.537866193012559, zeros=0}
    Feedback Error: [ [ 1.3984506885833525E-9 ], [ -2.900612816603143E-8 ], [ 1.60439439511606E-9 ], [ -3.038124851695301E-8 ] ]
    Error Statistics: {meanExponent=-8.175986581983155, negative=2, min=-3.038124851695301E-8, max=-3.038124851695301E-8, mean=-1.4096132899821257E-8, count=4.0, positive=2, stdDev=1.5605300601229117E-8, zeros=0}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 2.7965e-08 +- 3.7058e-08 [1.3985e-09 - 1.2114e-07] (8#)
    relativeTol: 1.9956e-08 +- 2.7119e-08 [4.7585e-10 - 8.2930e-08] (8#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=2.7965e-08 +- 3.7058e-08 [1.3985e-09 - 1.2114e-07] (8#), relativeTol=1.9956e-08 +- 2.7119e-08 [4.7585e-10 - 8.2930e-08] (8#)}
```



### Performance
Code from [LayerTestBase.java:192](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L192) executed in 0.00 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 0.1740 +- 0.0436 [0.1396 - 0.5215]
    Learning performance: 0.0037 +- 0.0040 [0.0000 - 0.0371]
    
```

