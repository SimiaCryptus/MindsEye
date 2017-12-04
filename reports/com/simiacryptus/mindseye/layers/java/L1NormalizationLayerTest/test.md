# L1NormalizationLayer
## L1NormalizationLayerTest
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
      "class": "com.simiacryptus.mindseye.layers.java.L1NormalizationLayer",
      "id": "a864e734-2f23-44db-97c1-504000002c06",
      "isFrozen": false,
      "name": "L1NormalizationLayer/a864e734-2f23-44db-97c1-504000002c06"
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
    [[ -1.824, -1.004, 1.18, 0.956 ]]
    --------------------
    Output: 
    [ 2.635838150289016, 1.4508670520231206, -1.7052023121387272, -1.3815028901734097 ]
```



### Batch Execution
Code from [LayerTestBase.java:138](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L138) executed in 0.00 seconds: 
```java
    BatchingTester batchingTester = getBatchingTester();
    return batchingTester==null?null:batchingTester.test(layer, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (80#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (40#)}
```



### Differential Validation
Code from [LayerTestBase.java:144](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.00 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [ -1.824, -1.004, 1.18, 0.956 ]
    Inputs Statistics: {meanExponent=0.07877461159590585, negative=2, min=0.956, max=0.956, mean=-0.1730000000000001, count=4.0, positive=2, stdDev=1.2768723507069921, zeros=0}
    Output: [ 2.635838150289016, 1.4508670520231206, -1.7052023121387272, -1.3815028901734097 ]
    Outputs Statistics: {meanExponent=0.23866851713914777, negative=2, min=-1.3815028901734097, max=-1.3815028901734097, mean=0.25000000000000006, count=4.0, positive=2, stdDev=1.8451912582470975, zeros=0}
    Feedback for input 0
    Inputs Values: [ -1.824, -1.004, 1.18, 0.956 ]
    Value Statistics: {meanExponent=0.07877461159590585, negative=2, min=0.956, max=0.956, mean=-0.1730000000000001, count=4.0, positive=2, stdDev=1.2768723507069921, zeros=0}
    Implemented Feedback: [ [ 2.3639279628453997, 2.096628687894683, -2.464165190951918, -1.9963914597881645 ], [ 3.8090146680477113, 0.6515419826923713, -2.464165190951918, -1.9963914597881645 ], [ 3.8090146680477113, 2.096628687894683, -3.90925189615423, -1.9963914597881645 ], [ 3.8090146680477113, 2.096628687894683, -2.464165190951918, -3.441478164990476 ] ]
    Implemented Statistics: {meanExponent=0.381197774323031, negative=8, min=-3.441478164990476, max=-3.441478164990476, mean=0.0, count=16.0, positive=8, stdDev=2.7388990186795885, zeros=0}
    Measured Feedback: [ [ 2.3642696203118874, 2.0969317127139675, -2.464521335658887, -1.996679997362527 ], [ 3.8095651832570354, 0.6516361497710399, -2.464521335658887, -1.996679997362527 ], [ 3.8095651832481536, 2.096931712707306, -3.9098168985973736, -1.9966799973603067 ], [ 3.8095651832481536, 2.096931712707306, -2.464521335654446, -3.441975560301014 ] ]
    Measured Statistics: {meanExponent=0.3812605381768833, negative=8, min=-3.441975560301014, max=-3.441975560301014, mean=5.551115123125783E-13, count=16.0, positive=8, stdDev=2.7392948705418347, zeros=0}
    Feedback Error: [ [ 3.416574664876215E-4, 3.030248192845697E-4, -3.5614470696865297E-4, -2.8853757436264615E-4 ], [ 5.505152093241428E-4, 9.416707866860552E-5, -3.5614470696865297E-4, -2.8853757436264615E-4 ], [ 5.505152004423586E-4, 3.0302481262323155E-4, -5.65002443143392E-4, -2.885375721422001E-4 ], [ 5.505152004423586E-4, 3.0302481262323155E-4, -3.5614470252776087E-4, -4.973953105378293E-4 ] ]
    Error Statistics: {meanExponent=-3.458845552207805, negative=8, min=-4.973953105378293E-4, max=-4.973953105378293E-4, mean=5.551462067820978E-13, count=16.0, positive=8, stdDev=3.9585186224610136E-4, zeros=0}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 3.7456e-04 +- 1.2809e-04 [9.4167e-05 - 5.6500e-04] (16#)
    relativeTol: 7.2260e-05 +- 9.0949e-13 [7.2260e-05 - 7.2260e-05] (16#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=3.7456e-04 +- 1.2809e-04 [9.4167e-05 - 5.6500e-04] (16#), relativeTol=7.2260e-05 +- 9.0949e-13 [7.2260e-05 - 7.2260e-05] (16#)}
```



### Performance
Code from [LayerTestBase.java:149](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L149) executed in 0.00 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 0.1585 +- 0.0725 [0.1282 - 0.7837]
    Learning performance: 0.0023 +- 0.0033 [0.0000 - 0.0171]
    
```

