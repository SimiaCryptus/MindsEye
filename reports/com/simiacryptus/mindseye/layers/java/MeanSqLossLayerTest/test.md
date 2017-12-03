# MeanSqLossLayer
## MeanSqLossLayerTest
### Json Serialization
Code from [LayerTestBase.java:84](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L84) executed in 0.05 seconds: 
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
      "id": "5eacdeb9-0f89-49b5-a080-71a400000001",
      "isFrozen": false,
      "name": "MeanSqLossLayer/5eacdeb9-0f89-49b5-a080-71a400000001"
    }
```



### Example Input/Output Pair
Code from [LayerTestBase.java:121](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L121) executed in 0.01 seconds: 
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
    	[ [ -0.94 ], [ 1.22 ], [ -1.544 ] ],
    	[ [ 0.088 ], [ 1.296 ], [ 0.868 ] ]
    ],
    [
    	[ [ 1.864 ], [ -1.192 ], [ 1.66 ] ],
    	[ [ -1.968 ], [ 0.604 ], [ -1.88 ] ]
    ]]
    --------------------
    Output: 
    [ 6.03388 ]
```



### Differential Validation
Code from [LayerTestBase.java:139](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L139) executed in 0.03 seconds: 
```java
    getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ -0.94 ], [ 1.22 ], [ -1.544 ] ],
    	[ [ 0.088 ], [ 1.296 ], [ 0.868 ] ]
    ],
    [
    	[ [ 1.864 ], [ -1.192 ], [ 1.66 ] ],
    	[ [ -1.968 ], [ 0.604 ], [ -1.88 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.1260429368107668, negative=2, min=0.868, max=0.868, mean=0.16466666666666668, count=6.0, positive=4, stdDev=1.08274568677147, zeros=0},
    {meanExponent=0.15267502224039492, negative=3, min=-1.88, max=-1.88, mean=-0.15199999999999997, count=6.0, positive=3, stdDev=1.5960668323517448, zeros=0}
    Output: [ 6.03388 ]
    Outputs Statistics: {meanExponent=0.780596668806147, negative=0, min=6.03388, max=6.03388, mean=6.03388, count=1.0, positive=1, stdDev=0.0, zeros=0}
    Feedback for input 0
    Inputs Values: [
    	[ [ -0.94 ], [ 1.22 ], [ -1.544 ] ],
    	[ [ 0.088 ], [ 1.296 ], [ 0.868 ] ]
    ]
    Value Statistics: {meanExponent=-0.1260429368107668, negative=2, min=0.868, max=0.868, mean=0.16466666666666668, count=6.0, positive=4, stdDev=1.08274568677147, zeros=0}
    Implemented Feedback: [ [ -0.9346666666666668 ], [ 0.6853333333333333 ], [ 0.8039999999999999 ], [ 0.2306666666666667 ], [ -1.0679999999999998 ], [ 0.9159999999999999 ] ]
    Implemented Statistics: {meanExponent=-0.15578896249592186, negative=2, min=0.9159999999999999, max=0.9159999999999999, mean=0.10555555555555556, count=6.0, positive=4, stdDev=0.8119662159243259, zeros=0}
    Measured Feedback: [ [ -0.9346499999995928 ], [ 0.6853499999959212 ], [ 0.8040166666578585 ], [ 0.23068333333320368 ], [ -1.0679833333337996 ], [ 0.916016666669961 ] ]
    Measured Statistics: {meanExponent=-0.15578157533540216, negative=2, min=0.916016666669961, max=0.916016666669961, mean=0.105572222220592, count=6.0, positive=4, stdDev=0.8119662159231478, zeros=0}
    Feedback Error: [ [ 1.666666707400566E-5 ], [ 1.666666258781646E-5 ], [ 1.6666657858599443E-5 ], [ 1.6666666536990782E-5 ], [ 1.666666620026014E-5 ], [ 1.6666669961029612E-5 ] ]
    Error Statistics: {meanExponent=-4.778151292863295, negative=0, min=1.6666669961029612E-5, max=1.6666669961029612E-5, mean=1.666666503645035E-5, count=6.0, positive=6, stdDev=3.851954340029219E-12, zeros=0}
    Feedback for input 1
    Inputs Values: [
    	[ [ 1.864 ], [ -1.192 ], [ 1.66 ] ],
    	[ [ -1.968 ], [ 0.604 ], [ -1.88 ] ]
    ]
    Value Statistics: {meanExponent=0.15267502224039492, negative=3, min=-1.88, max=-1.88, mean=-0.15199999999999997, count=6.0, positive=3, stdDev=1.5960668323517448, zeros=0}
    Implemented Feedback: [ [ 0.9346666666666668 ], [ -0.6853333333333333 ], [ -0.8039999999999999 ], [ -0.2306666666666667 ], [ 1.0679999999999998 ], [ -0.9159999999999999 ] ]
    Implemented Statistics: {meanExponent=-0.15578896249592186, negative=4, min=-0.9159999999999999, max=-0.9159999999999999, mean=-0.10555555555555556, count=6.0, positive=2, stdDev=0.8119662159243259, zeros=0}
    Measured Feedback: [ [ 0.9346833333268023 ], [ -0.6853166666687116 ], [ -0.8039833333395308 ], [ -0.23064999999711233 ], [ 1.0680166666787727 ], [ -0.9159833333338696 ] ]
    Measured Statistics: {meanExponent=-0.15579635017362484, negative=4, min=-0.9159833333338696, max=-0.9159833333338696, mean=-0.10553888888894154, count=6.0, positive=2, stdDev=0.8119662159269946, zeros=0}
    Feedback Error: [ [ 1.6666660135555844E-5 ], [ 1.6666664621745042E-5 ], [ 1.6666660469177863E-5 ], [ 1.6666669554354918E-5 ], [ 1.666667877286976E-5 ], [ 1.6666666130316088E-5 ] ]
    Error Statistics: {meanExponent=-4.77815125175596, negative=0, min=1.6666666130316088E-5, max=1.6666666130316088E-5, mean=1.666666661400325E-5, count=6.0, positive=6, stdDev=6.337977476853827E-12, zeros=0}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 1.6667e-05 +- 5.3032e-12 [1.6667e-05 - 1.6667e-05] (12#)
    relativeTol: 1.4078e-05 +- 9.9538e-06 [7.8027e-06 - 3.6128e-05] (12#)
    
```

### Performance
Code from [LayerTestBase.java:144](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.03 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 1.2124 +- 0.8224 [0.8663 - 9.2276]
    Learning performance: 0.0212 +- 0.0166 [0.0142 - 0.1653]
    
```

