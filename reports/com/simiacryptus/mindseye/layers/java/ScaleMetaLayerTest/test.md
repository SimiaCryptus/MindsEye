# ScaleMetaLayer
## ScaleMetaLayerTest
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
      "class": "com.simiacryptus.mindseye.layers.java.ScaleMetaLayer",
      "id": "a864e734-2f23-44db-97c1-504000002c86",
      "isFrozen": false,
      "name": "ScaleMetaLayer/a864e734-2f23-44db-97c1-504000002c86"
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
    [[ -1.508, 1.272, 1.144 ],
    [ 1.96, -1.868, 0.784 ]]
    --------------------
    Output: 
    [ -2.95568, -2.376096, 0.8968959999999999 ]
```



### Batch Execution
Code from [LayerTestBase.java:138](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L138) executed in 0.00 seconds: 
```java
    BatchingTester batchingTester = getBatchingTester();
    return batchingTester==null?null:batchingTester.test(layer, inputPrototype);
```

### Differential Validation
Code from [LayerTestBase.java:144](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.00 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [ -1.508, 1.272, 1.144 ],
    [ 1.96, -1.868, 0.784 ]
    Inputs Statistics: {meanExponent=0.11377149243438522, negative=1, min=1.144, max=1.144, mean=0.30266666666666664, count=3.0, positive=2, stdDev=1.2814006225827874, zeros=0},
    {meanExponent=0.15264966864499635, negative=1, min=0.784, max=0.784, mean=0.292, count=3.0, positive=2, stdDev=1.6010296686820018, zeros=0}
    Output: [ -2.95568, -2.376096, 0.8968959999999999 ]
    Outputs Statistics: {meanExponent=0.2664211610793816, negative=2, min=0.8968959999999999, max=0.8968959999999999, mean=-1.4782933333333332, count=3.0, positive=1, stdDev=1.6960980082037975, zeros=0}
    Feedback for input 0
    Inputs Values: [ -1.508, 1.272, 1.144 ]
    Value Statistics: {meanExponent=0.11377149243438522, negative=1, min=1.144, max=1.144, mean=0.30266666666666664, count=3.0, positive=2, stdDev=1.2814006225827874, zeros=0}
    Implemented Feedback: [ [ 1.96, 0.0, 0.0 ], [ 0.0, -1.868, 0.0 ], [ 0.0, 0.0, 0.784 ] ]
    Implemented Statistics: {meanExponent=0.15264966864499635, negative=1, min=0.784, max=0.784, mean=0.09733333333333333, count=9.0, positive=2, stdDev=0.9345477813122, zeros=6}
    Measured Feedback: [ [ 1.9600000000030704, 0.0, 0.0 ], [ 0.0, -1.8680000000026453, 0.0 ], [ 0.0, 0.0, 0.78400000000034 ] ]
    Measured Statistics: {meanExponent=0.15264966864549087, negative=1, min=0.78400000000034, max=0.78400000000034, mean=0.09733333333341834, count=9.0, positive=2, stdDev=0.9345477813135258, zeros=6}
    Feedback Error: [ [ 3.070432796903333E-12, 0.0, 0.0 ], [ 0.0, -2.645217378471898E-12, 0.0 ], [ 0.0, 0.0, 3.3995029014022293E-13 ] ]
    Error Statistics: {meanExponent=-11.852974540076298, negative=1, min=3.3995029014022293E-13, max=3.3995029014022293E-13, mean=8.501841206351755E-14, count=9.0, positive=2, stdDev=1.3529903328030908E-12, zeros=6}
    Feedback for input 1
    Inputs Values: [ 1.96, -1.868, 0.784 ]
    Value Statistics: {meanExponent=0.15264966864499635, negative=1, min=0.784, max=0.784, mean=0.292, count=3.0, positive=2, stdDev=1.6010296686820018, zeros=0}
    Implemented Feedback: [ [ -1.508, 0.0, 0.0 ], [ 0.0, 1.272, 0.0 ], [ 0.0, 0.0, 1.144 ] ]
    Implemented Statistics: {meanExponent=0.11377149243438522, negative=1, min=1.144, max=1.144, mean=0.10088888888888888, count=9.0, positive=2, stdDev=0.7534496141001429, zeros=6}
    Measured Feedback: [ [ -1.5079999999967342, 0.0, 0.0 ], [ 0.0, 1.2719999999966092, 0.0 ], [ 0.0, 0.0, 1.1440000000007 ] ]
    Measured Statistics: {meanExponent=0.1137714924337744, negative=1, min=1.1440000000007, max=1.1440000000007, mean=0.10088888888895278, count=9.0, positive=2, stdDev=0.7534496140988902, zeros=6}
    Feedback Error: [ [ 3.2658320492373605E-12, 0.0, 0.0 ], [ 0.0, -3.390843161810153E-12, 0.0 ], [ 0.0, 0.0, 7.001066393286237E-13 ] ]
    Error Statistics: {meanExponent=-11.703511418161392, negative=1, min=7.001066393286237E-13, max=7.001066393286237E-13, mean=6.389950297287012E-14, count=9.0, positive=2, stdDev=1.5852401609125692E-12, zeros=6}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 7.4513e-13 +- 1.2737e-12 [0.0000e+00 - 3.3908e-12] (18#)
    relativeTol: 7.3830e-13 +- 3.9455e-13 [2.1681e-13 - 1.3329e-12] (6#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=7.4513e-13 +- 1.2737e-12 [0.0000e+00 - 3.3908e-12] (18#), relativeTol=7.3830e-13 +- 3.9455e-13 [2.1681e-13 - 1.3329e-12] (6#)}
```



### Performance
Code from [LayerTestBase.java:149](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L149) executed in 0.00 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 0.1878 +- 0.1024 [0.1168 - 0.7495]
    Learning performance: 0.0019 +- 0.0015 [0.0000 - 0.0057]
    
```

