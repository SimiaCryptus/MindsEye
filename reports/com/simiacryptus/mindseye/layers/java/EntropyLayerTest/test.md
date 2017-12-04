# EntropyLayer
## EntropyLayerTest
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
      "class": "com.simiacryptus.mindseye.layers.java.EntropyLayer",
      "id": "a864e734-2f23-44db-97c1-504000002ba6",
      "isFrozen": true,
      "name": "EntropyLayer/a864e734-2f23-44db-97c1-504000002ba6"
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
    [[
    	[ [ 1.6 ], [ -0.284 ], [ 1.62 ] ],
    	[ [ 0.9 ], [ -1.8 ], [ 1.884 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ -0.7520058067931771 ], [ -0.35749381559314436 ], [ -0.7815303617757543 ] ],
    	[ [ 0.09482446409204366 ], [ 1.0580159968238143 ], [ -1.1933202798744587 ] ]
    ]
```



### Batch Execution
Code from [LayerTestBase.java:138](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L138) executed in 0.00 seconds: 
```java
    BatchingTester batchingTester = getBatchingTester();
    return batchingTester==null?null:batchingTester.test(layer, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (120#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (120#)}
```



### Differential Validation
Code from [LayerTestBase.java:144](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.00 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ 1.6 ], [ -0.284 ], [ 1.62 ] ],
    	[ [ 0.9 ], [ -1.8 ], [ 1.884 ] ]
    ]
    Inputs Statistics: {meanExponent=0.058591541707513815, negative=2, min=1.884, max=1.884, mean=0.6533333333333334, count=6.0, positive=4, stdDev=1.310435381424391, zeros=0}
    Output: [
    	[ [ -0.7520058067931771 ], [ -0.35749381559314436 ], [ -0.7815303617757543 ] ],
    	[ [ 0.09482446409204366 ], [ 1.0580159968238143 ], [ -1.1933202798744587 ] ]
    ]
    Outputs Statistics: {meanExponent=-0.2665657942809248, negative=4, min=-1.1933202798744587, max=-1.1933202798744587, mean=-0.3219183005201127, count=6.0, positive=2, stdDev=0.7342477708105858, zeros=0}
    Feedback for input 0
    Inputs Values: [
    	[ [ 1.6 ], [ -0.284 ], [ 1.62 ] ],
    	[ [ 0.9 ], [ -1.8 ], [ 1.884 ] ]
    ]
    Value Statistics: {meanExponent=0.058591541707513815, negative=2, min=1.884, max=1.884, mean=0.6533333333333334, count=6.0, positive=4, stdDev=1.310435381424391, zeros=0}
    Implemented Feedback: [ [ -1.4700036292457357, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, -0.8946394843421737, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.258781040820931, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, -1.587786664902119, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, -1.4824261492442927, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, -1.6333971761541712 ] ]
    Implemented Statistics: {meanExponent=0.01945932358962257, negative=5, min=-1.6333971761541712, max=-1.6333971761541712, mean=-0.1891520017518767, count=36.0, positive=1, stdDev=0.5035119150277819, zeros=30}
    Measured Feedback: [ [ -1.4700348785945394, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, -0.8946950378402319, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.25895711782630837, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, -1.5877588866097803, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, -1.4824570128069148, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, -1.6334237149617792 ] ]
    Measured Statistics: {meanExponent=0.01951600636070587, negative=5, min=-1.6334237149617792, max=-1.6334237149617792, mean=-0.1891503448051927, count=36.0, positive=1, stdDev=0.5035228105691476, zeros=30}
    Feedback Error: [ [ -3.12493488037191E-5, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, -5.5553498058191764E-5, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 1.7607700537736193E-4, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 2.777829233879814E-5, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, -3.086356262205214E-5, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, -2.653880760794536E-5 ] ]
    Error Statistics: {meanExponent=-4.359618661415856, negative=4, min=-2.653880760794536E-5, max=-2.653880760794536E-5, mean=1.6569466840069917E-6, count=36.0, positive=2, stdDev=3.2229855508158136E-5, zeros=30}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 9.6683e-06 +- 3.0790e-05 [0.0000e+00 - 1.7608e-04] (36#)
    relativeTol: 6.8174e-05 +- 1.2186e-04 [8.1237e-06 - 3.4009e-04] (6#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=9.6683e-06 +- 3.0790e-05 [0.0000e+00 - 1.7608e-04] (36#), relativeTol=6.8174e-05 +- 1.2186e-04 [8.1237e-06 - 3.4009e-04] (6#)}
```



### Performance
Code from [LayerTestBase.java:149](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L149) executed in 0.00 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 0.1510 +- 0.0737 [0.1111 - 0.5899]
    Learning performance: 0.0027 +- 0.0028 [0.0000 - 0.0200]
    
```

### Function Plots
Code from [ActivationLayerTestBase.java:74](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/java/ActivationLayerTestBase.java#L74) executed in 0.01 seconds: 
```java
    return plot("Value Plot", plotData, x -> new double[]{x[0], x[1]});
```

Returns: 

![Result](etc/test.1.png)



Code from [ActivationLayerTestBase.java:78](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/java/ActivationLayerTestBase.java#L78) executed in 0.01 seconds: 
```java
    return plot("Derivative Plot", plotData, x -> new double[]{x[0], x[2]});
```

Returns: 

![Result](etc/test.2.png)



