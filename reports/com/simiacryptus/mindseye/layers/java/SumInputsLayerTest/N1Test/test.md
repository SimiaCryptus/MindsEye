# SumInputsLayer
## N1Test
### Json Serialization
Code from [LayerTestBase.java:83](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L83) executed in 0.00 seconds: 
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
      "class": "com.simiacryptus.mindseye.layers.java.SumInputsLayer",
      "id": "370a9587-74a1-4959-b406-fa4500002cad",
      "isFrozen": false,
      "name": "SumInputsLayer/370a9587-74a1-4959-b406-fa4500002cad"
    }
```



### Example Input/Output Pair
Code from [LayerTestBase.java:120](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L120) executed in 0.00 seconds: 
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
    [[ 1.488, 0.856, -0.116 ],
    [ -1.612 ]]
    --------------------
    Output: 
    [ -0.12400000000000011, -0.7560000000000001, -1.7280000000000002 ]
```



### Batch Execution
Code from [LayerTestBase.java:138](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L138) executed in 0.00 seconds: 
```java
    BatchingTester batchingTester = getBatchingTester();
    return batchingTester==null?null:batchingTester.test(layer, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (70#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (70#)}
```



### Differential Validation
Code from [LayerTestBase.java:144](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.00 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [ 1.488, 0.856, -0.116 ],
    [ -1.612 ]
    Inputs Statistics: {meanExponent=-0.2768217716286895, negative=1, min=-0.116, max=-0.116, mean=0.7426666666666666, count=3.0, positive=2, stdDev=0.6597157637110767, zeros=0},
    {meanExponent=0.20736503746907187, negative=1, min=-1.612, max=-1.612, mean=-1.612, count=1.0, positive=0, stdDev=0.0, zeros=0}
    Output: [ -0.12400000000000011, -0.7560000000000001, -1.7280000000000002 ]
    Outputs Statistics: {meanExponent=-0.26350426039789443, negative=3, min=-1.7280000000000002, max=-1.7280000000000002, mean=-0.8693333333333335, count=3.0, positive=0, stdDev=0.6597157637110764, zeros=0}
    Feedback for input 0
    Inputs Values: [ 1.488, 0.856, -0.116 ]
    Value Statistics: {meanExponent=-0.2768217716286895, negative=1, min=-0.116, max=-0.116, mean=0.7426666666666666, count=3.0, positive=2, stdDev=0.6597157637110767, zeros=0}
    Implemented Feedback: [ [ 1.0, 0.0, 0.0 ], [ 0.0, 1.0, 0.0 ], [ 0.0, 0.0, 1.0 ] ]
    Implemented Statistics: {meanExponent=0.0, negative=0, min=1.0, max=1.0, mean=0.3333333333333333, count=9.0, positive=3, stdDev=0.4714045207910317, zeros=6}
    Measured Feedback: [ [ 0.9999999999998899, 0.0, 0.0 ], [ 0.0, 0.9999999999998899, 0.0 ], [ 0.0, 0.0, 1.0000000000021103 ] ]
    Measured Statistics: {meanExponent=2.7361184650972856E-13, negative=0, min=1.0000000000021103, max=1.0000000000021103, mean=0.3333333333335433, count=9.0, positive=3, stdDev=0.4714045207913287, zeros=6}
    Feedback Error: [ [ -1.1013412404281553E-13, 0.0, 0.0 ], [ 0.0, -1.1013412404281553E-13, 0.0 ], [ 0.0, 0.0, 2.1103119252074976E-12 ] ]
    Error Statistics: {meanExponent=-12.530603180987852, negative=2, min=2.1103119252074976E-12, max=2.1103119252074976E-12, mean=2.1000485301354073E-13, count=9.0, positive=1, stdDev=6.733627986644662E-13, zeros=6}
    Feedback for input 1
    Inputs Values: [ -1.612 ]
    Value Statistics: {meanExponent=0.20736503746907187, negative=1, min=-1.612, max=-1.612, mean=-1.612, count=1.0, positive=0, stdDev=0.0, zeros=0}
    Implemented Feedback: [ [ 1.0, 1.0, 1.0 ] ]
    Implemented Statistics: {meanExponent=0.0, negative=0, min=1.0, max=1.0, mean=1.0, count=3.0, positive=3, stdDev=0.0, zeros=0}
    Measured Feedback: [ [ 0.9999999999998899, 0.9999999999998899, 0.9999999999998899 ] ]
    Measured Statistics: {meanExponent=-4.7830642341045674E-14, negative=0, min=0.9999999999998899, max=0.9999999999998899, mean=0.9999999999998899, count=3.0, positive=3, stdDev=0.0, zeros=0}
    Feedback Error: [ [ -1.1013412404281553E-13, -1.1013412404281553E-13, -1.1013412404281553E-13 ] ]
    Error Statistics: {meanExponent=-12.958078098036827, negative=3, min=-1.1013412404281553E-13, max=-1.1013412404281553E-13, mean=-1.1013412404281553E-13, count=3.0, positive=0, stdDev=0.0, zeros=0}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 2.2175e-13 +- 5.7184e-13 [0.0000e+00 - 2.1103e-12] (12#)
    relativeTol: 2.2175e-13 +- 3.7271e-13 [5.5067e-14 - 1.0552e-12] (6#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=2.2175e-13 +- 5.7184e-13 [0.0000e+00 - 2.1103e-12] (12#), relativeTol=2.2175e-13 +- 3.7271e-13 [5.5067e-14 - 1.0552e-12] (6#)}
```



### Performance
Code from [LayerTestBase.java:149](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L149) executed in 0.01 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 0.2694 +- 0.1074 [0.1738 - 1.1114]
    Learning performance: 0.0199 +- 0.0115 [0.0114 - 0.1026]
    
```

