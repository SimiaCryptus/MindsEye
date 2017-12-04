# CrossProductLayer
## CrossProductLayerTest
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
      "class": "com.simiacryptus.mindseye.layers.java.CrossProductLayer",
      "id": "a864e734-2f23-44db-97c1-504000002b9e",
      "isFrozen": false,
      "name": "CrossProductLayer/a864e734-2f23-44db-97c1-504000002b9e"
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
    [[ 1.444, -1.6, 1.468, -1.86 ]]
    --------------------
    Output: 
    [ -2.3104, 2.119792, -2.6858400000000002, -2.3488, 2.9760000000000004, -2.73048 ]
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
    Inputs: [ 1.444, -1.6, 1.468, -1.86 ]
    Inputs Statistics: {meanExponent=0.19998154392187828, negative=2, min=-1.86, max=-1.86, mean=-0.13700000000000007, count=4.0, positive=2, stdDev=1.595672585463572, zeros=0}
    Output: [ -2.3104, 2.119792, -2.6858400000000002, -2.3488, 2.9760000000000004, -2.73048 ]
    Outputs Statistics: {meanExponent=0.39996308784375656, negative=4, min=-2.73048, max=-2.73048, mean=-0.8299546666666666, count=6.0, positive=2, stdDev=2.4062838834647553, zeros=0}
    Feedback for input 0
    Inputs Values: [ 1.444, -1.6, 1.468, -1.86 ]
    Value Statistics: {meanExponent=0.19998154392187828, negative=2, min=-1.86, max=-1.86, mean=-0.13700000000000007, count=4.0, positive=2, stdDev=1.595672585463572, zeros=0}
    Implemented Feedback: [ [ -1.6, 1.468, -1.86, 0.0, 0.0, 0.0 ], [ 1.444, 0.0, 0.0, 1.468, -1.86, 0.0 ], [ 0.0, 1.444, 0.0, -1.6, 0.0, -1.86 ], [ 0.0, 0.0, 1.444, 0.0, -1.6, 1.468 ] ]
    Implemented Statistics: {meanExponent=0.19998154392187828, negative=6, min=1.468, max=1.468, mean=-0.06850000000000002, count=24.0, positive=6, stdDev=1.1303883182340484, zeros=12}
    Measured Feedback: [ [ -1.6000000000016001, 1.4680000000000248, -1.8599999999979744, 0.0, 0.0, 0.0 ], [ 1.4439999999993347, 0.0, 0.0, 1.4680000000000248, -1.8600000000024153, 0.0 ], [ 0.0, 1.4439999999993347, 0.0, -1.5999999999971593, 0.0, -1.8600000000024153 ], [ 0.0, 0.0, 1.4440000000037756, 0.0, -1.6000000000016001, 1.4680000000000248 ] ]
    Measured Statistics: {meanExponent=0.1999815439220041, negative=6, min=1.4680000000000248, max=1.4680000000000248, mean=-0.06850000000002687, count=24.0, positive=6, stdDev=1.1303883182343943, zeros=12}
    Feedback Error: [ [ -1.6000534230897756E-12, 2.4868995751603507E-14, 2.0257129307310606E-12, 0.0, 0.0, 0.0 ], [ -6.652456363553938E-13, 0.0, 0.0, 2.4868995751603507E-14, -2.4151791677695655E-12, 0.0 ], [ 0.0, -6.652456363553938E-13, 0.0, 2.8408386754108506E-12, 0.0, -2.4151791677695655E-12 ], [ 0.0, 0.0, 3.775646462145232E-12, 0.0, -1.6000534230897756E-12, 2.4868995751603507E-14 ] ]
    Error Statistics: {meanExponent=-12.221323143173324, negative=6, min=2.4868995751603507E-14, max=2.4868995751603507E-14, mean=-2.683964162031316E-14, count=24.0, positive=6, stdDev=1.355322475840209E-12, zeros=12}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 7.5324e-13 +- 1.1271e-12 [0.0000e+00 - 3.7756e-12] (24#)
    relativeTol: 4.6036e-13 +- 3.7640e-13 [8.4704e-15 - 1.3074e-12] (12#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=7.5324e-13 +- 1.1271e-12 [0.0000e+00 - 3.7756e-12] (24#), relativeTol=4.6036e-13 +- 3.7640e-13 [8.4704e-15 - 1.3074e-12] (12#)}
```



### Performance
Code from [LayerTestBase.java:149](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L149) executed in 0.00 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 0.2370 +- 0.0936 [0.1738 - 0.7694]
    Learning performance: 0.0028 +- 0.0022 [0.0000 - 0.0142]
    
```

