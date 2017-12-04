# MeanSqLossLayer
## MeanSqLossLayerTest
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
      "class": "com.simiacryptus.mindseye.layers.java.MeanSqLossLayer",
      "id": "a864e734-2f23-44db-97c1-504000002c2f",
      "isFrozen": false,
      "name": "MeanSqLossLayer/a864e734-2f23-44db-97c1-504000002c2f"
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
    	[ [ 0.484 ], [ 0.752 ], [ -0.848 ] ],
    	[ [ 0.08 ], [ 0.068 ], [ -1.076 ] ]
    ],
    [
    	[ [ -1.316 ], [ 0.644 ], [ 0.304 ] ],
    	[ [ -1.332 ], [ -1.34 ], [ 0.384 ] ]
    ]]
    --------------------
    Output: 
    [ 1.7810959999999998 ]
```



### Batch Execution
Code from [LayerTestBase.java:138](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L138) executed in 0.00 seconds: 
```java
    BatchingTester batchingTester = getBatchingTester();
    return batchingTester==null?null:batchingTester.test(layer, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (130#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (130#)}
```



### Differential Validation
Code from [LayerTestBase.java:144](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.00 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ 0.484 ], [ 0.752 ], [ -0.848 ] ],
    	[ [ 0.08 ], [ 0.068 ], [ -1.076 ] ]
    ],
    [
    	[ [ -1.316 ], [ 0.644 ], [ 0.304 ] ],
    	[ [ -1.332 ], [ -1.34 ], [ 0.384 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.4571882957464468, negative=2, min=-1.076, max=-1.076, mean=-0.09000000000000002, count=6.0, positive=4, stdDev=0.6632073582221476, zeros=0},
    {meanExponent=-0.1255074020311461, negative=3, min=0.384, max=0.384, mean=-0.44266666666666676, count=6.0, positive=3, stdDev=0.8926146362730609, zeros=0}
    Output: [ 1.7810959999999998 ]
    Outputs Statistics: {meanExponent=0.2506873283047081, negative=0, min=1.7810959999999998, max=1.7810959999999998, mean=1.7810959999999998, count=1.0, positive=1, stdDev=0.0, zeros=0}
    Feedback for input 0
    Inputs Values: [
    	[ [ 0.484 ], [ 0.752 ], [ -0.848 ] ],
    	[ [ 0.08 ], [ 0.068 ], [ -1.076 ] ]
    ]
    Value Statistics: {meanExponent=-0.4571882957464468, negative=2, min=-1.076, max=-1.076, mean=-0.09000000000000002, count=6.0, positive=4, stdDev=0.6632073582221476, zeros=0}
    Implemented Feedback: [ [ 0.6 ], [ 0.4706666666666667 ], [ 0.03599999999999999 ], [ 0.4693333333333334 ], [ -0.38399999999999995 ], [ -0.48666666666666664 ] ]
    Implemented Statistics: {meanExponent=-0.508298096889035, negative=2, min=-0.48666666666666664, max=-0.48666666666666664, mean=0.1175555555555556, count=6.0, positive=4, stdDev=0.42904573988513184, zeros=0}
    Measured Feedback: [ [ 0.600016666669756 ], [ 0.4706833333334437 ], [ 0.03601666666908088 ], [ 0.4693500000030326 ], [ -0.38398333333189427 ], [ -0.486649999995592 ] ]
    Measured Statistics: {meanExponent=-0.508263070801923, negative=2, min=-0.486649999995592, max=-0.486649999995592, mean=0.11757222222463781, count=6.0, positive=4, stdDev=0.42904573988474887, zeros=0}
    Feedback Error: [ [ 1.666666975597142E-5 ], [ 1.6666666777021E-5 ], [ 1.666666908088643E-5 ], [ 1.6666669699239023E-5 ], [ 1.6666668105680404E-5 ], [ 1.6666671074638817E-5 ] ]
    Error Statistics: {meanExponent=-4.778151187439452, negative=0, min=1.6666671074638817E-5, max=1.6666671074638817E-5, mean=1.6666669082239516E-5, count=6.0, positive=6, stdDev=1.3642420526593924E-12, zeros=0}
    Feedback for input 1
    Inputs Values: [
    	[ [ -1.316 ], [ 0.644 ], [ 0.304 ] ],
    	[ [ -1.332 ], [ -1.34 ], [ 0.384 ] ]
    ]
    Value Statistics: {meanExponent=-0.1255074020311461, negative=3, min=0.384, max=0.384, mean=-0.44266666666666676, count=6.0, positive=3, stdDev=0.8926146362730609, zeros=0}
    Implemented Feedback: [ [ -0.6 ], [ -0.4706666666666667 ], [ -0.03599999999999999 ], [ -0.4693333333333334 ], [ 0.38399999999999995 ], [ 0.48666666666666664 ] ]
    Implemented Statistics: {meanExponent=-0.508298096889035, negative=4, min=0.48666666666666664, max=0.48666666666666664, mean=-0.1175555555555556, count=6.0, positive=2, stdDev=0.42904573988513184, zeros=0}
    Measured Feedback: [ [ -0.5999833333336646 ], [ -0.47064999999735235 ], [ -0.03598333333298953 ], [ -0.46931666666027994 ], [ 0.3840166666679856 ], [ 0.4866833333383447 ] ]
    Measured Statistics: {meanExponent=-0.5083331389456168, negative=4, min=0.4866833333383447, max=0.4866833333383447, mean=-0.11753888888632602, count=6.0, positive=2, stdDev=0.4290457398854022, zeros=0}
    Feedback Error: [ [ 1.666666633537428E-5 ], [ 1.66666693143247E-5 ], [ 1.666666701045927E-5 ], [ 1.6666673053444825E-5 ], [ 1.6666667985665296E-5 ], [ 1.666667167804503E-5 ] ]
    Error Statistics: {meanExponent=-4.77815118360083, negative=0, min=1.666667167804503E-5, max=1.666667167804503E-5, mean=1.6666669229552233E-5, count=6.0, positive=6, stdDev=2.427686525159359E-12, zeros=0}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 1.6667e-05 +- 1.9822e-12 [1.6667e-05 - 1.6667e-05] (12#)
    relativeTol: 5.3276e-05 +- 7.9728e-05 [1.3889e-05 - 2.3154e-04] (12#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=1.6667e-05 +- 1.9822e-12 [1.6667e-05 - 1.6667e-05] (12#), relativeTol=5.3276e-05 +- 7.9728e-05 [1.3889e-05 - 2.3154e-04] (12#)}
```



### Performance
Code from [LayerTestBase.java:149](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L149) executed in 0.01 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 0.3147 +- 0.1355 [0.1710 - 0.9347]
    Learning performance: 0.0072 +- 0.0388 [0.0000 - 0.3904]
    
```

