# CrossDifferenceLayer
## CrossDifferenceLayerTest
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
      "class": "com.simiacryptus.mindseye.layers.java.CrossDifferenceLayer",
      "id": "a864e734-2f23-44db-97c1-504000002b96",
      "isFrozen": false,
      "name": "CrossDifferenceLayer/a864e734-2f23-44db-97c1-504000002b96"
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
    [[ 0.824, 0.252, 1.22, 0.7 ]]
    --------------------
    Output: 
    [ 0.572, -0.396, 0.124, -0.968, -0.44799999999999995, 0.52 ]
```



### Batch Execution
Code from [LayerTestBase.java:138](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L138) executed in 0.00 seconds: 
```java
    BatchingTester batchingTester = getBatchingTester();
    return batchingTester==null?null:batchingTester.test(layer, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (100#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (100#)}
```



### Differential Validation
Code from [LayerTestBase.java:144](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.00 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [ 0.824, 0.252, 1.22, 0.7 ]
    Inputs Statistics: {meanExponent=-0.1878035942080838, negative=0, min=0.7, max=0.7, mean=0.7490000000000001, count=4.0, positive=4, stdDev=0.3452810449474453, zeros=0}
    Output: [ 0.572, -0.396, 0.124, -0.968, -0.44799999999999995, 0.52 ]
    Outputs Statistics: {meanExponent=-0.36638839752964864, negative=3, min=0.52, max=0.52, mean=-0.09933333333333334, count=6.0, positive=3, stdDev=0.5550227222575868, zeros=0}
    Feedback for input 0
    Inputs Values: [ 0.824, 0.252, 1.22, 0.7 ]
    Value Statistics: {meanExponent=-0.1878035942080838, negative=0, min=0.7, max=0.7, mean=0.7490000000000001, count=4.0, positive=4, stdDev=0.3452810449474453, zeros=0}
    Implemented Feedback: [ [ 1.0, 1.0, 1.0, 0.0, 0.0, 0.0 ], [ -1.0, 0.0, 0.0, 1.0, 1.0, 0.0 ], [ 0.0, -1.0, 0.0, -1.0, 0.0, 1.0 ], [ 0.0, 0.0, -1.0, 0.0, -1.0, -1.0 ] ]
    Implemented Statistics: {meanExponent=0.0, negative=6, min=-1.0, max=-1.0, mean=0.0, count=24.0, positive=6, stdDev=0.7071067811865476, zeros=12}
    Measured Feedback: [ [ 0.9999999999998899, 0.9999999999998899, 0.9999999999998899, 0.0, 0.0, 0.0 ], [ -0.9999999999998899, 0.0, 0.0, 0.9999999999998899, 0.9999999999998899, 0.0 ], [ 0.0, -0.9999999999998899, 0.0, -0.9999999999998899, 0.0, 0.9999999999998899 ], [ 0.0, 0.0, -0.9999999999998899, 0.0, -0.9999999999998899, -0.9999999999998899 ] ]
    Measured Statistics: {meanExponent=-4.783064234104566E-14, negative=6, min=-0.9999999999998899, max=-0.9999999999998899, mean=0.0, count=24.0, positive=6, stdDev=0.7071067811864696, zeros=12}
    Feedback Error: [ [ -1.1013412404281553E-13, -1.1013412404281553E-13, -1.1013412404281553E-13, 0.0, 0.0, 0.0 ], [ 1.1013412404281553E-13, 0.0, 0.0, -1.1013412404281553E-13, -1.1013412404281553E-13, 0.0 ], [ 0.0, 1.1013412404281553E-13, 0.0, 1.1013412404281553E-13, 0.0, -1.1013412404281553E-13 ], [ 0.0, 0.0, 1.1013412404281553E-13, 0.0, 1.1013412404281553E-13, 1.1013412404281553E-13 ] ]
    Error Statistics: {meanExponent=-12.958078098036824, negative=6, min=1.1013412404281553E-13, max=1.1013412404281553E-13, mean=0.0, count=24.0, positive=6, stdDev=7.787658595071524E-14, zeros=12}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 5.5067e-14 +- 5.5067e-14 [0.0000e+00 - 1.1013e-13] (24#)
    relativeTol: 5.5067e-14 +- 0.0000e+00 [5.5067e-14 - 5.5067e-14] (12#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=5.5067e-14 +- 5.5067e-14 [0.0000e+00 - 1.1013e-13] (24#), relativeTol=5.5067e-14 +- 0.0000e+00 [5.5067e-14 - 5.5067e-14] (12#)}
```



### Performance
Code from [LayerTestBase.java:149](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L149) executed in 0.00 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 0.1994 +- 0.1083 [0.1254 - 0.7922]
    Learning performance: 0.0031 +- 0.0047 [0.0000 - 0.0456]
    
```

