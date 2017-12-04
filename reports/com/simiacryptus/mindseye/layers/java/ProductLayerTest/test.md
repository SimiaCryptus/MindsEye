# ProductLayer
## ProductLayerTest
### Json Serialization
Code from [LayerTestBase.java:83](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L83) executed in 0.00 seconds: 
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
      "class": "com.simiacryptus.mindseye.layers.java.ProductLayer",
      "id": "370a9587-74a1-4959-b406-fa4500002c7c",
      "isFrozen": false,
      "name": "ProductLayer/370a9587-74a1-4959-b406-fa4500002c7c"
    }
```



### Example Input/Output Pair
Code from [LayerTestBase.java:120](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L120) executed in 0.00 seconds: 
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
    [[ -1.76, 1.868, 1.936 ]]
    --------------------
    Output: 
    [ -6.364948480000001 ]
```



### Batch Execution
Code from [LayerTestBase.java:138](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L138) executed in 0.00 seconds: 
```java
    BatchingTester batchingTester = getBatchingTester();
    return batchingTester==null?null:batchingTester.test(layer, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (40#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (40#)}
```



### Differential Validation
Code from [LayerTestBase.java:144](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.00 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [ -1.76, 1.868, 1.936 ]
    Inputs Statistics: {meanExponent=0.26793163089353306, negative=1, min=1.936, max=1.936, mean=0.6813333333333333, count=3.0, positive=2, stdDev=1.7265065562832043, zeros=0}
    Output: [ -6.364948480000001 ]
    Outputs Statistics: {meanExponent=0.8037948926805993, negative=1, min=-6.364948480000001, max=-6.364948480000001, mean=-6.364948480000001, count=1.0, positive=0, stdDev=0.0, zeros=0}
    Feedback for input 0
    Inputs Values: [ -1.76, 1.868, 1.936 ]
    Value Statistics: {meanExponent=0.26793163089353306, negative=1, min=1.936, max=1.936, mean=0.6813333333333333, count=3.0, positive=2, stdDev=1.7265065562832043, zeros=0}
    Implemented Feedback: [ [ 3.6164480000000006 ], [ -3.40736 ], [ -3.2876800000000004 ] ]
    Implemented Statistics: {meanExponent=0.5358632617870662, negative=2, min=-3.2876800000000004, max=-3.2876800000000004, mean=-1.0261973333333334, count=3.0, positive=1, stdDev=3.2832095672721775, zeros=0}
    Measured Feedback: [ [ 3.6164480000078214 ], [ -3.4073599999917548 ], [ -3.2876800000014583 ] ]
    Measured Statistics: {meanExponent=0.5358632617870932, negative=2, min=-3.2876800000014583, max=-3.2876800000014583, mean=-1.026197333328464, count=3.0, positive=1, stdDev=3.283209567274205, zeros=0}
    Feedback Error: [ [ 7.820855074669453E-12 ], [ 8.245404359286113E-12 ], [ -1.4579448759377556E-12 ] ]
    Error Statistics: {meanExponent=-11.34226423312326, negative=1, min=-1.4579448759377556E-12, max=-1.4579448759377556E-12, mean=4.8694381860059366E-12, count=3.0, positive=2, stdDev=4.477491323884844E-12, zeros=0}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 5.8414e-12 +- 3.1044e-12 [1.4579e-12 - 8.2454e-12] (3#)
    relativeTol: 8.3765e-13 +- 4.3868e-13 [2.2173e-13 - 1.2099e-12] (3#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=5.8414e-12 +- 3.1044e-12 [1.4579e-12 - 8.2454e-12] (3#), relativeTol=8.3765e-13 +- 4.3868e-13 [2.2173e-13 - 1.2099e-12] (3#)}
```



### Performance
Code from [LayerTestBase.java:149](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L149) executed in 0.00 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 0.1638 +- 0.0417 [0.1140 - 0.3762]
    Learning performance: 0.0022 +- 0.0014 [0.0000 - 0.0057]
    
```

