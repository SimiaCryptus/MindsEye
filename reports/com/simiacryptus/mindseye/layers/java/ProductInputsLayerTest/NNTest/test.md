# ProductInputsLayer
## NNTest
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
      "class": "com.simiacryptus.mindseye.layers.java.ProductInputsLayer",
      "id": "370a9587-74a1-4959-b406-fa4500002c75",
      "isFrozen": false,
      "name": "ProductInputsLayer/370a9587-74a1-4959-b406-fa4500002c75"
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
    [[ 0.732, 1.312, 1.876 ],
    [ 0.652, -0.012, -1.02 ]]
    --------------------
    Output: 
    [ 0.477264, -0.015744, -1.9135199999999999 ]
```



### Batch Execution
Code from [LayerTestBase.java:138](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L138) executed in 0.00 seconds: 
```java
    BatchingTester batchingTester = getBatchingTester();
    return batchingTester==null?null:batchingTester.test(layer, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (90#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (90#)}
```



### Differential Validation
Code from [LayerTestBase.java:144](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.00 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [ 0.732, 1.312, 1.876 ],
    [ 0.652, -0.012, -1.02 ]
    Inputs Statistics: {meanExponent=0.08522591671369299, negative=0, min=1.876, max=1.876, mean=1.3066666666666666, count=3.0, positive=3, stdDev=0.4670512700859391, zeros=0},
    {meanExponent=-0.6993236621528457, negative=2, min=-1.02, max=-1.02, mean=-0.12666666666666668, count=3.0, positive=1, stdDev=0.6873899103775738, zeros=0}
    Output: [ 0.477264, -0.015744, -1.9135199999999999 ]
    Outputs Statistics: {meanExponent=-0.6140977454391529, negative=2, min=-1.9135199999999999, max=-1.9135199999999999, mean=-0.484, count=3.0, positive=1, stdDev=1.0306663846968136, zeros=0}
    Feedback for input 0
    Inputs Values: [ 0.732, 1.312, 1.876 ]
    Value Statistics: {meanExponent=0.08522591671369299, negative=0, min=1.876, max=1.876, mean=1.3066666666666666, count=3.0, positive=3, stdDev=0.4670512700859391, zeros=0}
    Implemented Feedback: [ [ 0.652, 0.0, 0.0 ], [ 0.0, -0.012, 0.0 ], [ 0.0, 0.0, -1.02 ] ]
    Implemented Statistics: {meanExponent=-0.6993236621528457, negative=2, min=-1.02, max=-1.02, mean=-0.042222222222222223, count=9.0, positive=1, stdDev=0.4013316106767508, zeros=6}
    Measured Feedback: [ [ 0.6519999999998749, 0.0, 0.0 ], [ 0.0, -0.011999999999998123, 0.0 ], [ 0.0, 0.0, -1.020000000000465 ] ]
    Measured Statistics: {meanExponent=-0.6993236621528301, negative=2, min=-1.020000000000465, max=-1.020000000000465, mean=-0.04222222222228758, count=9.0, positive=1, stdDev=0.40133161067685263, zeros=6}
    Feedback Error: [ [ -1.2512213487525514E-13, 0.0, 0.0 ], [ 0.0, 1.8769708010069053E-15, 0.0 ], [ 0.0, 0.0, -4.649614027130156E-13 ] ]
    Error Statistics: {meanExponent=-13.320597144911146, negative=2, min=-4.649614027130156E-13, max=-4.649614027130156E-13, mean=-6.535628519858487E-14, count=9.0, positive=1, stdDev=1.4659285321876127E-13, zeros=6}
    Feedback for input 1
    Inputs Values: [ 0.652, -0.012, -1.02 ]
    Value Statistics: {meanExponent=-0.6993236621528457, negative=2, min=-1.02, max=-1.02, mean=-0.12666666666666668, count=3.0, positive=1, stdDev=0.6873899103775738, zeros=0}
    Implemented Feedback: [ [ 0.732, 0.0, 0.0 ], [ 0.0, 1.312, 0.0 ], [ 0.0, 0.0, 1.876 ] ]
    Implemented Statistics: {meanExponent=0.08522591671369299, negative=0, min=1.876, max=1.876, mean=0.43555555555555553, count=9.0, positive=3, stdDev=0.6724058151495372, zeros=6}
    Measured Feedback: [ [ 0.7319999999999549, 0.0, 0.0 ], [ 0.0, 1.3119999999999972, 0.0 ], [ 0.0, 0.0, 1.8759999999984345 ] ]
    Measured Statistics: {meanExponent=0.08522591671356294, negative=0, min=1.8759999999984345, max=1.8759999999984345, mean=0.4355555555553763, count=9.0, positive=3, stdDev=0.672405815149162, zeros=6}
    Feedback Error: [ [ -4.5075054799781356E-14, 0.0, 0.0 ], [ 0.0, -2.886579864025407E-15, 0.0 ], [ 0.0, 0.0, -1.5654144647214707E-12 ] ]
    Error Statistics: {meanExponent=-13.230350272123205, negative=3, min=-1.5654144647214707E-12, max=-1.5654144647214707E-12, mean=-1.7926401104280862E-13, count=9.0, positive=0, stdDev=4.902768229810263E-13, zeros=6}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 1.2252e-13 +- 3.6623e-13 [0.0000e+00 - 1.5654e-12] (18#)
    relativeTol: 1.4187e-13 +- 1.4233e-13 [1.1001e-15 - 4.1722e-13] (6#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=1.2252e-13 +- 3.6623e-13 [0.0000e+00 - 1.5654e-12] (18#), relativeTol=1.4187e-13 +- 1.4233e-13 [1.1001e-15 - 4.1722e-13] (6#)}
```



### Performance
Code from [LayerTestBase.java:149](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L149) executed in 0.00 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 0.1888 +- 0.0546 [0.1282 - 0.4645]
    Learning performance: 0.0185 +- 0.0104 [0.0114 - 0.0883]
    
```

