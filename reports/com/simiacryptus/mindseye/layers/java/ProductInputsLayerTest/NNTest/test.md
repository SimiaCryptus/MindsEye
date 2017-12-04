# ProductInputsLayer
## NNTest
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
      "id": "a864e734-2f23-44db-97c1-504000002c75",
      "isFrozen": false,
      "name": "ProductInputsLayer/a864e734-2f23-44db-97c1-504000002c75"
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
    [[ 1.5, 1.508, -1.916 ],
    [ 1.092, -1.416, -0.456 ]]
    --------------------
    Output: 
    [ 1.6380000000000001, -2.135328, 0.873696 ]
```



### Batch Execution
Code from [LayerTestBase.java:138](../../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L138) executed in 0.00 seconds: 
```java
    BatchingTester batchingTester = getBatchingTester();
    return batchingTester==null?null:batchingTester.test(layer, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (90#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (90#)}
```



### Differential Validation
Code from [LayerTestBase.java:144](../../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.00 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [ 1.5, 1.508, -1.916 ],
    [ 1.092, -1.416, -0.456 ]
    Inputs Statistics: {meanExponent=0.21229603511065406, negative=1, min=-1.916, max=-1.916, mean=0.36400000000000005, count=3.0, positive=2, stdDev=1.6122067692038344, zeros=0},
    {meanExponent=-0.050583088537698793, negative=2, min=-0.456, max=-0.456, mean=-0.25999999999999995, count=3.0, positive=1, stdDev=1.0332240802459067, zeros=0}
    Output: [ 1.6380000000000001, -2.135328, 0.873696 ]
    Outputs Statistics: {meanExponent=0.16171294657295524, negative=1, min=0.873696, max=0.873696, mean=0.1254560000000001, count=3.0, positive=2, stdDev=1.6287824434417262, zeros=0}
    Feedback for input 0
    Inputs Values: [ 1.5, 1.508, -1.916 ]
    Value Statistics: {meanExponent=0.21229603511065406, negative=1, min=-1.916, max=-1.916, mean=0.36400000000000005, count=3.0, positive=2, stdDev=1.6122067692038344, zeros=0}
    Implemented Feedback: [ [ 1.092, 0.0, 0.0 ], [ 0.0, -1.416, 0.0 ], [ 0.0, 0.0, -0.456 ] ]
    Implemented Statistics: {meanExponent=-0.050583088537698793, negative=2, min=-0.456, max=-0.456, mean=-0.08666666666666664, count=9.0, positive=1, stdDev=0.6089933405948614, zeros=6}
    Measured Feedback: [ [ 1.092000000000315, 0.0, 0.0 ], [ 0.0, -1.41600000000075, 0.0 ], [ 0.0, 0.0, -0.4559999999997899 ] ]
    Measured Statistics: {meanExponent=-0.050583088537647064, negative=2, min=-0.4559999999997899, max=-0.4559999999997899, mean=-0.08666666666669166, count=9.0, positive=1, stdDev=0.6089933405950969, zeros=6}
    Feedback Error: [ [ 3.148592497836944E-13, 0.0, 0.0 ], [ 0.0, -7.500666754367558E-13, 0.0 ], [ 0.0, 0.0, 2.1010970741031088E-13 ] ]
    Error Statistics: {meanExponent=-12.434779184996968, negative=1, min=2.1010970741031088E-13, max=2.1010970741031088E-13, mean=-2.501085758252783E-14, count=9.0, positive=2, stdDev=2.789369835146011E-13, zeros=6}
    Feedback for input 1
    Inputs Values: [ 1.092, -1.416, -0.456 ]
    Value Statistics: {meanExponent=-0.050583088537698793, negative=2, min=-0.456, max=-0.456, mean=-0.25999999999999995, count=3.0, positive=1, stdDev=1.0332240802459067, zeros=0}
    Implemented Feedback: [ [ 1.5, 0.0, 0.0 ], [ 0.0, 1.508, 0.0 ], [ 0.0, 0.0, -1.916 ] ]
    Implemented Statistics: {meanExponent=0.21229603511065406, negative=1, min=-1.916, max=-1.916, mean=0.12133333333333335, count=9.0, positive=2, stdDev=0.9464920026662196, zeros=6}
    Measured Feedback: [ [ 1.4999999999987246, 0.0, 0.0 ], [ 0.0, 1.508000000001175, 0.0 ], [ 0.0, 0.0, -1.9159999999995847 ] ]
    Measured Statistics: {meanExponent=0.21229603511061237, negative=1, min=-1.9159999999995847, max=-1.9159999999995847, mean=0.12133333333336832, count=9.0, positive=2, stdDev=0.9464920026661052, zeros=6}
    Feedback Error: [ [ -1.2754242106893798E-12, 0.0, 0.0 ], [ 0.0, 1.1750600492632657E-12, 0.0 ], [ 0.0, 0.0, 4.1522341120980855E-13 ] ]
    Error Statistics: {meanExponent=-12.068667816764952, negative=1, min=4.1522341120980855E-13, max=4.1522341120980855E-13, mean=3.498436108707715E-14, count=9.0, positive=2, stdDev=5.933771864054363E-13, zeros=6}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 2.3004e-13 +- 4.0368e-13 [0.0000e+00 - 1.2754e-12] (18#)
    relativeTol: 2.6042e-13 +- 1.1646e-13 [1.0836e-13 - 4.2514e-13] (6#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=2.3004e-13 +- 4.0368e-13 [0.0000e+00 - 1.2754e-12] (18#), relativeTol=2.6042e-13 +- 1.1646e-13 [1.0836e-13 - 4.2514e-13] (6#)}
```



### Performance
Code from [LayerTestBase.java:149](../../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L149) executed in 0.00 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 0.1880 +- 0.1544 [0.1368 - 1.6557]
    Learning performance: 0.0201 +- 0.0170 [0.0086 - 0.1453]
    
```

