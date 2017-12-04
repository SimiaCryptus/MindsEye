# FullyConnectedLayer
## FullyConnectedLayerTest
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
      "class": "com.simiacryptus.mindseye.layers.java.FullyConnectedLayer",
      "id": "a864e734-2f23-44db-97c1-504000002bb0",
      "isFrozen": false,
      "name": "FullyConnectedLayer/a864e734-2f23-44db-97c1-504000002bb0",
      "outputDims": [
        3
      ],
      "inputDims": [
        3
      ],
      "weights": {
        "dimensions": [
          3,
          3
        ],
        "data": [
          0.42944534033353243,
          0.7445356536206575,
          0.03359549416622668,
          0.6060379412860019,
          -0.19596057978186052,
          0.23178488222464302,
          -0.15086333868773924,
          0.1563046678815048,
          -0.8319612036364962
        ]
      }
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
    [[ 0.716, -0.012, 1.092 ]]
    --------------------
    Output: 
    [ 0.13546764253636595, 0.7061237522763764, -0.8872286791347314 ]
```



### Batch Execution
Code from [LayerTestBase.java:138](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L138) executed in 0.00 seconds: 
```java
    BatchingTester batchingTester = getBatchingTester();
    return batchingTester==null?null:batchingTester.test(layer, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (60#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (60#)}
```



### Differential Validation
Code from [LayerTestBase.java:144](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.00 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [ 0.716, -0.012, 1.092 ]
    Inputs Statistics: {meanExponent=-0.675894364425267, negative=1, min=1.092, max=1.092, mean=0.5986666666666667, count=3.0, positive=2, stdDev=0.45827890585925457, zeros=0}
    Output: [ 0.13546764253636595, 0.7061237522763764, -0.8872286791347314 ]
    Outputs Statistics: {meanExponent=-0.357082678293098, negative=1, min=-0.8872286791347314, max=-0.8872286791347314, mean=-0.0152124281073297, count=3.0, positive=2, stdDev=0.6591516544962998, zeros=0}
    Feedback for input 0
    Inputs Values: [ 0.716, -0.012, 1.092 ]
    Value Statistics: {meanExponent=-0.675894364425267, negative=1, min=1.092, max=1.092, mean=0.5986666666666667, count=3.0, positive=2, stdDev=0.45827890585925457, zeros=0}
    Implemented Feedback: [ [ 0.42944534033353243, 0.7445356536206575, 0.03359549416622668 ], [ 0.6060379412860019, -0.19596057978186052, 0.23178488222464302 ], [ -0.15086333868773924, 0.1563046678815048, -0.8319612036364962 ] ]
    Implemented Statistics: {meanExponent=-0.581834799810355, negative=3, min=-0.8319612036364962, max=-0.8319612036364962, mean=0.11365765082294116, count=9.0, positive=6, stdDev=0.44998516642364583, zeros=0}
    Measured Feedback: [ [ 0.4294453403330678, 0.7445356536206482, 0.03359549416659391 ], [ 0.6060379412858863, -0.19596057978299442, 0.23178488222530547 ], [ -0.15086333868791213, 0.15630466788096875, -0.8319612036367108 ] ]
    Measured Statistics: {meanExponent=-0.5818347998095701, negative=3, min=-0.8319612036367108, max=-0.8319612036367108, mean=0.11365765082276146, count=9.0, positive=6, stdDev=0.44998516642374864, zeros=0}
    Feedback Error: [ [ -4.64628335805628E-13, -9.325873406851315E-15, 3.6722708207648225E-13 ], [ -1.155742168634788E-13, -1.133898530625288E-12, 6.624423232182153E-13 ], [ -1.728894805097525E-13, -5.360434318646412E-13, -2.1460611066004276E-13 ] ]
    Error Statistics: {meanExponent=-12.61789735939584, negative=7, min=-2.1460611066004276E-13, max=-2.1460611066004276E-13, mean=-1.7969961938233168E-13, count=9.0, positive=2, stdDev=4.900563276355356E-13, zeros=0}
    Learning Gradient for weight set 0
    Weights: [ 0.42944534033353243, 0.7445356536206575, 0.03359549416622668, 0.6060379412860019, -0.19596057978186052, 0.23178488222464302, -0.15086333868773924, 0.1563046678815048, -0.8319612036364962 ]
    Implemented Gradient: [ [ 0.716, 0.0, 0.0 ], [ 0.0, 0.716, 0.0 ], [ 0.0, 0.0, 0.716 ], [ -0.012, 0.0, 0.0 ], [ 0.0, -0.012, 0.0 ], [ 0.0, 0.0, -0.012 ], [ 1.092, 0.0, 0.0 ], [ 0.0, 1.092, 0.0 ], [ 0.0, 0.0, 1.092 ] ]
    Implemented Statistics: {meanExponent=-0.6758943644252671, negative=3, min=1.092, max=1.092, mean=0.19955555555555554, count=27.0, positive=6, stdDev=0.3868479779250389, zeros=18}
    Measured Gradient: [ [ 0.7159999999994948, 0.0, 0.0 ], [ 0.0, 0.7159999999983846, 0.0 ], [ 0.0, 0.0, 0.716000000000605 ], [ -0.011999999999789956, 0.0, 0.0 ], [ 0.0, -0.012000000000345068, 0.0 ], [ 0.0, 0.0, -0.011999999999234845 ], [ 1.0919999999997598, 0.0, 0.0 ], [ 0.0, 1.0919999999992047, 0.0 ], [ 0.0, 0.0, 1.092000000000315 ] ]
    Measured Statistics: {meanExponent=-0.675894364427935, negative=3, min=1.092000000000315, max=1.092000000000315, mean=0.1995555555554961, count=27.0, positive=6, stdDev=0.38684797792488956, zeros=18}
    Gradient Error: [ [ -5.051514762044462E-13, 0.0, 0.0 ], [ 0.0, -1.6153745008296028E-12, 0.0 ], [ 0.0, 0.0, 6.050715484207103E-13 ], [ 2.1004378791822376E-13, 0.0, 0.0 ], [ 0.0, -3.450677243943545E-13, 0.0 ], [ 0.0, 0.0, 7.65155300230802E-13 ], [ -2.402522625288839E-13, 0.0, 0.0 ], [ 0.0, -7.953637748414621E-13, 0.0 ], [ 0.0, 0.0, 3.148592497836944E-13 ] ]
    Error Statistics: {meanExponent=-12.309242766249405, negative=5, min=3.148592497836944E-13, max=3.148592497836944E-13, mean=-5.948443897945626E-14, count=27.0, positive=4, stdDev=4.1603513933939385E-13, zeros=18}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 2.5203e-13 +- 3.7022e-13 [0.0000e+00 - 1.6154e-12] (36#)
    relativeTol: 3.9100e-12 +- 7.7106e-12 [6.2629e-15 - 3.1881e-11] (18#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=2.5203e-13 +- 3.7022e-13 [0.0000e+00 - 1.6154e-12] (36#), relativeTol=3.9100e-12 +- 7.7106e-12 [6.2629e-15 - 3.1881e-11] (18#)}
```



### Performance
Code from [LayerTestBase.java:149](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L149) executed in 0.01 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 0.2612 +- 0.0831 [0.1653 - 0.7267]
    Learning performance: 0.4659 +- 0.3728 [0.2764 - 2.2941]
    
```

