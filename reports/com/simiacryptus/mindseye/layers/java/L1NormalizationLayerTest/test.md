# L1NormalizationLayer
## L1NormalizationLayerTest
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
      "class": "com.simiacryptus.mindseye.layers.java.L1NormalizationLayer",
      "id": "e2a3bda5-e7e7-4c05-aeb3-4ede00001eaf",
      "isFrozen": false,
      "name": "L1NormalizationLayer/e2a3bda5-e7e7-4c05-aeb3-4ede00001eaf"
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
    [[ 133.20000000000002, 56.8, -74.0, -35.6 ]]
    --------------------
    Output: 
    [ 1.6567164179104479, 0.7064676616915422, -0.9203980099502487, -0.4427860696517413 ]
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
    Inputs: [ 133.20000000000002, 56.8, -74.0, -35.6 ]
    Inputs Statistics: {meanExponent=1.824883569562288, negative=2, min=-35.6, max=-35.6, mean=20.1, count=4.0, positive=2, stdDev=80.77097250869275, zeros=0}
    Output: [ 1.6567164179104479, 0.7064676616915422, -0.9203980099502487, -0.4427860696517413 ]
    Outputs Statistics: {meanExponent=-0.08037247918616315, negative=2, min=-0.4427860696517413, max=-0.4427860696517413, mean=0.2500000000000001, count=4.0, positive=2, stdDev=1.0046140859290142, zeros=0}
    Feedback for input 0
    Inputs Values: [ 133.20000000000002, 56.8, -74.0, -35.6 ]
    Value Statistics: {meanExponent=1.824883569562288, negative=2, min=-35.6, max=-35.6, mean=20.1, count=4.0, positive=2, stdDev=80.77097250869275, zeros=0}
    Implemented Feedback: [ [ -0.00816811465062746, -0.008786911215068933, 0.011447736442167272, 0.00550728942352912 ], [ -0.020605925595901093, 0.0036508997302046988, 0.011447736442167272, 0.00550728942352912 ], [ -0.020605925595901093, -0.008786911215068933, 0.023885547387440902, 0.00
```
...[skipping 703 bytes](etc/64.txt)...
```
    9641516 ] ]
    Measured Statistics: {meanExponent=-1.982559259765501, negative=7, min=0.017945078049641516, max=0.017945078049641516, mean=-6.938893903907228E-13, count=16.0, positive=9, stdDev=0.013606456311290437, zeros=0}
    Feedback Error: [ [ 1.0153408185401003E-8, 1.0926575661623916E-8, -1.423531962793656E-8, -6.8485499979398234E-9 ], [ 2.562290920063437E-8, -4.542925352742089E-9, -1.423531962793656E-8, -6.8485499979398234E-9 ], [ 2.562735009273287E-8, 1.0928796107673167E-8, -2.970815131050908E-8, -6.849660220964449E-9 ], [ 2.562735009273287E-8, 1.0928796107673167E-8, -1.4238650297010436E-8, -2.2319161234463092E-8 ] ]
    Error Statistics: {meanExponent=-7.887860308185501, negative=9, min=-2.2319161234463092E-8, max=-2.2319161234463092E-8, mean=-6.938886856593107E-13, count=16.0, positive=7, stdDev=1.6921736854183007E-8, zeros=0}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 1.4978e-08 +- 7.8751e-09 [4.5429e-09 - 2.9708e-08] (16#)
    relativeTol: 6.2183e-07 +- 1.2599e-10 [6.2153e-07 - 6.2217e-07] (16#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=1.4978e-08 +- 7.8751e-09 [4.5429e-09 - 2.9708e-08] (16#), relativeTol=6.2183e-07 +- 1.2599e-10 [6.2153e-07 - 6.2217e-07] (16#)}
```



### Performance
Code from [LayerTestBase.java:192](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L192) executed in 0.00 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 0.1751 +- 0.1197 [0.1112 - 1.1627]
    Learning performance: 0.0034 +- 0.0045 [0.0000 - 0.0371]
    
```

