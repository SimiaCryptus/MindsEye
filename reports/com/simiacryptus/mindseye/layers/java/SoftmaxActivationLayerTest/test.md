# SoftmaxActivationLayer
## SoftmaxActivationLayerTest
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
      "class": "com.simiacryptus.mindseye.layers.java.SoftmaxActivationLayer",
      "id": "a864e734-2f23-44db-97c1-504000002c94",
      "isFrozen": false,
      "name": "SoftmaxActivationLayer/a864e734-2f23-44db-97c1-504000002c94"
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
    [[ -1.212, 0.312, -1.216, -0.016 ]]
    --------------------
    Output: 
    [ 0.10107726437650613, 0.4640003090162907, 0.10067376286003495, 0.33424866374716816 ]
```



### Batch Execution
Code from [LayerTestBase.java:138](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L138) executed in 0.00 seconds: 
```java
    BatchingTester batchingTester = getBatchingTester();
    return batchingTester==null?null:batchingTester.test(layer, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (80#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (40#)}
```



### Differential Validation
Code from [LayerTestBase.java:144](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.00 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [ -1.212, 0.312, -1.216, -0.016 ]
    Inputs Statistics: {meanExponent=-0.5333223071396622, negative=3, min=-0.016, max=-0.016, mean=-0.5329999999999999, count=4.0, positive=1, stdDev=0.6908046033430872, zeros=0}
    Output: [ 0.10107726437650613, 0.4640003090162907, 0.10067376286003495, 0.33424866374716816 ]
    Outputs Statistics: {meanExponent=-0.7004605674205541, negative=0, min=0.33424866374716816, max=0.33424866374716816, mean=0.25, count=4.0, positive=4, stdDev=0.1560210529754057, zeros=0}
    Feedback for input 0
    Inputs Values: [ -1.212, 0.312, -1.216, -0.016 ]
    Value Statistics: {meanExponent=-0.5333223071396622, negative=3, min=-0.016, max=-0.016, mean=-0.5329999999999999, count=4.0, positive=1, stdDev=0.6908046033430872, zeros=0}
    Implemented Feedback: [ [ 0.090860651002668, -0.04689988190522015, -0.010175828544381435, -0.03378494055306642 ], [ -0.04689988190522015, 0.24870402224907742, -0.04671265707688898, -0.15509148326696826 ], [ -0.010175828544381435, -0.04671265707688898, 0.09053855633163638, -0.03365007071036596 ], [ -0.03378494055306642, -0.15509148326696826, -0.03365007071036596, 0.22252649453040066 ] ]
    Implemented Statistics: {meanExponent=-1.2595487295679724, negative=12, min=0.22252649453040066, max=0.22252649453040066, mean=3.469446951953614E-18, count=16.0, positive=4, stdDev=0.10881443272634966, zeros=0}
    Measured Feedback: [ [ 0.09086427570970113, -0.04690175288313281, -0.01017623448895666, -0.033786288336501435 ], [ -0.0469000507045958, 0.24870491737272182, -0.04671282520268827, -0.1550920414644663 ], [ -0.0101762348996004, -0.04671452247140273, 0.09054217184265023, -0.0336514144716471 ], [ -0.033785500523908496, -0.15509405384139807, -0.03365062844581934, 0.22253018281293002 ] ]
    Measured Statistics: {meanExponent=-1.2595378723804993, negative=12, min=0.22253018281293002, max=0.22253018281293002, mean=2.42861286636753E-13, count=16.0, positive=4, stdDev=0.10881587593319705, zeros=0}
    Feedback Error: [ [ 3.6247070331202336E-6, -1.87097791265467E-6, -4.0594457522379057E-7, -1.3477834350170137E-6 ], [ -1.6879937564384662E-7, 8.951236444010302E-7, -1.681257992894314E-7, -5.581974980573001E-7 ], [ -4.063552189650238E-7, -1.865394513751728E-6, 3.615511013854711E-6, -1.3437612811431632E-6 ], [ -5.599708420775396E-7, -2.5705744298165634E-6, -5.577354533839451E-7, 3.6882825293627075E-6 ] ]
    Error Statistics: {meanExponent=-6.015079942522566, negative=12, min=3.6882825293627075E-6, max=3.6882825293627075E-6, mean=2.4285716666849755E-13, count=16.0, positive=4, stdDev=1.9226985192390776E-6, zeros=0}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 1.4780e-06 +- 1.2298e-06 [1.6813e-07 - 3.6883e-06] (16#)
    relativeTol: 1.2500e-05 +- 7.8012e-06 [1.7996e-06 - 1.9966e-05] (16#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=1.4780e-06 +- 1.2298e-06 [1.6813e-07 - 3.6883e-06] (16#), relativeTol=1.2500e-05 +- 7.8012e-06 [1.7996e-06 - 1.9966e-05] (16#)}
```



### Performance
Code from [LayerTestBase.java:149](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L149) executed in 0.01 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 0.2764 +- 0.1024 [0.2080 - 0.8236]
    Learning performance: 0.0017 +- 0.0017 [0.0000 - 0.0086]
    
```

