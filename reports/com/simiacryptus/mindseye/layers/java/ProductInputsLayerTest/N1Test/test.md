# ProductInputsLayer
## N1Test
### Json Serialization
Code from [LayerTestBase.java:83](../../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L83) executed in 0.00 seconds: 
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
      "class": "com.simiacryptus.mindseye.layers.java.ProductInputsLayer",
      "id": "a864e734-2f23-44db-97c1-504000002c6e",
      "isFrozen": false,
      "name": "ProductInputsLayer/a864e734-2f23-44db-97c1-504000002c6e"
    }
```



### Example Input/Output Pair
Code from [LayerTestBase.java:120](../../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L120) executed in 0.00 seconds: 
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
    [[ -0.02, -0.42, 0.304 ],
    [ -1.524 ]]
    --------------------
    Output: 
    [ 0.03048, 0.64008, -0.463296 ]
```



### Batch Execution
Code from [LayerTestBase.java:138](../../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L138) executed in 0.00 seconds: 
```java
    BatchingTester batchingTester = getBatchingTester();
    return batchingTester==null?null:batchingTester.test(layer, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (70#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (70#)}
```



### Differential Validation
Code from [LayerTestBase.java:144](../../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.00 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [ -0.02, -0.42, 0.304 ],
    [ -1.524 ]
    Inputs Statistics: {meanExponent=-0.8642823767764548, negative=2, min=0.304, max=0.304, mean=-0.04533333333333334, count=3.0, positive=1, stdDev=0.29611409212591616, zeros=0},
    {meanExponent=0.1829849670035817, negative=1, min=-1.524, max=-1.524, mean=-1.524, count=1.0, positive=0, stdDev=0.0, zeros=0}
    Output: [ 0.03048, 0.64008, -0.463296 ]
    Outputs Statistics: {meanExponent=-0.6812974097728732, negative=1, min=-0.463296, max=-0.463296, mean=0.06908799999999998, count=3.0, positive=2, stdDev=0.4512778763998962, zeros=0}
    Feedback for input 0
    Inputs Values: [ -0.02, -0.42, 0.304 ]
    Value Statistics: {meanExponent=-0.8642823767764548, negative=2, min=0.304, max=0.304, mean=-0.04533333333333334, count=3.0, positive=1, stdDev=0.29611409212591616, zeros=0}
    Implemented Feedback: [ [ -1.524, 0.0, 0.0 ], [ 0.0, -1.524, 0.0 ], [ 0.0, 0.0, -1.524 ] ]
    Implemented Statistics: {meanExponent=0.1829849670035817, negative=3, min=-1.524, max=-1.524, mean=-0.508, count=9.0, positive=0, stdDev=0.7184204896855324, zeros=6}
    Measured Feedback: [ [ -1.5239999999999698, 0.0, 0.0 ], [ 0.0, -1.5239999999994147, 0.0 ], [ 0.0, 0.0, -1.5239999999999698 ] ]
    Measured Statistics: {meanExponent=0.18298496700352032, negative=3, min=-1.5239999999999698, max=-1.5239999999999698, mean=-0.5079999999999283, count=9.0, positive=0, stdDev=0.7184204896854308, zeros=6}
    Feedback Error: [ [ 3.019806626980426E-14, 0.0, 0.0 ], [ 0.0, 5.853095785823825E-13, 0.0 ], [ 0.0, 0.0, 3.019806626980426E-14 ] ]
    Error Statistics: {meanExponent=-13.090885366972886, negative=0, min=3.019806626980426E-14, max=3.019806626980426E-14, mean=7.174507901355455E-14, count=9.0, positive=3, stdDev=1.8199051926747497E-13, zeros=6}
    Feedback for input 1
    Inputs Values: [ -1.524 ]
    Value Statistics: {meanExponent=0.1829849670035817, negative=1, min=-1.524, max=-1.524, mean=-1.524, count=1.0, positive=0, stdDev=0.0, zeros=0}
    Implemented Feedback: [ [ -0.02, -0.42, 0.304 ] ]
    Implemented Statistics: {meanExponent=-0.8642823767764548, negative=2, min=0.304, max=0.304, mean=-0.04533333333333334, count=3.0, positive=1, stdDev=0.29611409212591616, zeros=0}
    Measured Feedback: [ [ -0.019999999999985307, -0.4199999999998649, 0.30399999999985994 ] ]
    Measured Statistics: {meanExponent=-0.8642823767766745, negative=2, min=0.30399999999985994, max=0.30399999999985994, mean=-0.045333333333330096, count=3.0, positive=1, stdDev=0.29611409212580453, zeros=0}
    Feedback Error: [ [ 1.4693107841523556E-14, 1.350586309456503E-13, -1.400546345564635E-13 ] ]
    Error Statistics: {meanExponent=-13.185355502035252, negative=1, min=-1.400546345564635E-13, max=-1.400546345564635E-13, mean=3.2323680769034504E-15, count=3.0, positive=2, stdDev=1.1260650848795967E-13, zeros=0}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 7.7959e-14 +- 1.6061e-13 [0.0000e+00 - 5.8531e-13] (12#)
    relativeTol: 1.6172e-13 +- 1.2517e-13 [9.9075e-15 - 3.6733e-13] (6#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=7.7959e-14 +- 1.6061e-13 [0.0000e+00 - 5.8531e-13] (12#), relativeTol=1.6172e-13 +- 1.2517e-13 [9.9075e-15 - 3.6733e-13] (6#)}
```



### Performance
Code from [LayerTestBase.java:149](../../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L149) executed in 0.01 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 0.2364 +- 0.0645 [0.1624 - 0.5700]
    Learning performance: 0.0182 +- 0.0108 [0.0086 - 0.0912]
    
```

