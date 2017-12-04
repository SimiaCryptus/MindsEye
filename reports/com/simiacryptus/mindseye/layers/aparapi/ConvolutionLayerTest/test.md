# ConvolutionLayer
## ConvolutionLayerTest
### Json Serialization
Code from [LayerTestBase.java:83](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L83) executed in 0.04 seconds: 
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
      "class": "com.simiacryptus.mindseye.layers.aparapi.ConvolutionLayer",
      "id": "a864e734-2f23-44db-97c1-504000000001",
      "isFrozen": false,
      "name": "ConvolutionLayer/a864e734-2f23-44db-97c1-504000000001",
      "filter": {
        "dimensions": [
          3,
          3,
          4
        ],
        "data": [
          1.836,
          -0.216,
          0.28,
          -0.892,
          -1.1,
          1.332,
          1.992,
          0.684,
          0.376,
          -0.972,
          -1.688,
          0.56,
          -1.312,
          -0.32,
          0.472,
          -0.952,
          0.248,
          0.88,
          0.468,
          -1.56,
          0.588,
          -1.116,
          1.02,
          0.436,
          -1.516,
          -1.5,
          -0.316,
          -1.856,
          0.644,
          -1.628,
          1.908,
          1.36,
          -1.784,
          0.516,
          -1.188,
          -0.296
        ]
      },
      "skip": {
        "dimensions": [
          1,
          1
        ]
      },
      "simple": true
    }
```



### Example Input/Output Pair
Code from [LayerTestBase.java:120](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L120) executed in 0.31 seconds: 
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
    	[ [ 1.6, -1.912 ], [ 1.516, 1.136 ], [ -0.128, 0.788 ] ],
    	[ [ -0.276, 1.324 ], [ -0.04, -1.612 ], [ 1.128, -1.088 ] ],
    	[ [ 1.596, 0.616 ], [ 0.296, 0.812 ], [ 1.704, -0.724 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ -7.869104, 0.9793120000000005 ], [ 3.091408, 3.2972640000000006 ], [ 2.849648000000001, -4.210512 ] ],
    	[ [ 5.379840000000001, 1.370704 ], [ 5.691376000000001, -7.470415999999999 ], [ -0.929152, -3.883728 ] ],
    	[ [ -3.2073600000000004, 0.4599679999999999 ], [ -0.17049599999999998, 1.9560800000000003 ], [ -2.1059360000000003, 0.494192 ] ]
    ]
```



### Batch Execution
Code from [LayerTestBase.java:138](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L138) executed in 0.12 seconds: 
```java
    BatchingTester batchingTester = getBatchingTester();
    return batchingTester==null?null:batchingTester.test(layer, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (360#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (360#)}
```



### Differential Validation
Code from [LayerTestBase.java:144](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.60 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ 1.6, -1.912 ], [ 1.516, 1.136 ], [ -0.128, 0.788 ] ],
    	[ [ -0.276, 1.324 ], [ -0.04, -1.612 ], [ 1.128, -1.088 ] ],
    	[ [ 1.596, 0.616 ], [ 0.296, 0.812 ], [ 1.704, -0.724 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.13048604569516883, negative=7, min=-0.724, max=-0.724, mean=0.3742222222222221, count=18.0, positive=11, stdDev=1.1032253499804592, zeros=0}
    Output: [
    	[ [ -7.869104, 0.9793120000000005 ], [ 3.091408, 3.2972640000000006 ], [ 2.849648000000001, -4.210512 ] ],
    	[ [ 5.379840000000001, 1.370704 ], [ 5.691376000000001, -7.470415999999999 ], [ -0.929152, -3.883728 ] ],
    	[ [ -3.2073600000000004, 0.4599679999999999 ], [ -0.17049599999999998, 1.9560800000000003 ], [ -2.1059360000000003, 0.494192 ] ]
    ]
    Outputs Statistics: {meanExponent=0.3187339014779412, negative=8, min=0.494192, max=0.494192, mean=-0.23760622222222208, count=18.0, positive=10, stdDev=3.8162195984450307, zeros=0}
    Feedback for input 0
    Inputs Values: [
    	[ [ 1.6, -1.912 ], [ 1.516, 1.136 ], [ -0.128, 0.788 ] ],
    	[ [ -0.276, 1.324 ], [ -0.04, -1.612 ], [ 1.128, -1.088 ] ],
    	[ [ 1.596, 0.616 ], [ 0.296, 0.812 ], [ 1.704, -0.724 ] ]
    ]
    Value Statistics: {meanExponent=-0.13048604569516883, negative=7, min=-0.724, max=-0.724, mean=0.3742222222222221, count=18.0, positive=11, stdDev=1.1032253499804592, zeros=0}
    Implemented Feedback: [ [ -1.1, 1.332, 0.0, 0.684, 0.376, 0.0, 0.0, 0.0, ... ], [ -0.892, -1.1, 1.332, 1.992, 0.684, 0.376, 0.0, 0.0, ... ], [ 0.0, -0.892, -1.1, 0.0, 1.992, 0.684, 0.0, 0.0, ... ], [ -0.216, 0.28, 0.0, -1.1, 1.332, 0.0, 0.684, 0.376, ... ], [ 1.836, -0.216, 0.28, -0.892, -1.1, 1.332, 1.992, 0.684, ... ], [ 0.0, 1.836, -0.216, 0.0, -0.892, -1.1, 0.0, 1.992, ... ], [ 0.0, 0.0, 0.0, -0.216, 0.28, 0.0, -1.1, 1.332, ... ], [ 0.0, 0.0, 0.0, 1.836, -0.216, 0.28, -0.892, -1.1, ... ], ... ]
    Implemented Statistics: {meanExponent=-0.0811555018602731, negative=100, min=1.36, max=1.36, mean=-0.0762716049382716, count=324.0, positive=96, stdDev=0.8816054520840593, zeros=128}
    Measured Feedback: [ [ -1.1000000000027654, 1.33
```
...[skipping 4332 bytes](etc/1.txt)...
```
    752, 1.5959999999992647, 0.0, -0.04000000000559112, 0.2959999999996299, ... ], [ 0.0, 0.0, 0.0, 1.6000000000016001, -0.276000000010157, 1.5960000000014851, 1.5159999999969642, -0.04000000000115023, ... ], ... ]
    Measured Statistics: {meanExponent=-0.17153621913880102, negative=84, min=-1.6120000000002799, max=-1.6120000000002799, mean=0.0875925925923688, count=648.0, positive=112, stdDev=0.6223003775872968, zeros=452}
    Gradient Error: [ [ -5.591117846481808E-12, 4.07079925324183E-12, 0.0, 1.3502532425491154E-12, 1.4988010832439613E-13, 0.0, 0.0, 0.0, ... ], [ -3.035793838535028E-12, 3.2906663505194444E-12, -4.810984943759422E-12, 7.601697049608447E-13, 1.3502532425491154E-12, 1.4988010832439613E-13, 0.0, 0.0, ... ], [ 0.0, 5.845990358466224E-12, -5.591117846481808E-12, 0.0, -3.6807223935397815E-12, 1.3502532425491154E-12, 0.0, 0.0, ... ], [ -1.2752021660844548E-12, 8.146372465489549E-12, 0.0, -1.1502257479811817E-12, -4.810984943759422E-12, 0.0, 1.3502532425491154E-12, 1.4988010832439613E-13, ... ], [ 1.6000534230897756E-12, 7.606582030916798E-12, -7.354117315117037E-13, 1.4050982599655981E-12, -5.591117846481808E-12, -3.7009284525879593E-13, -3.6807223935397815E-12, 1.3502532425491154E-12, ... ], [ 0.0, 1.6000534230897756E-12, -1.2752021660844548E-12, 0.0, 5.845990358466224E-12, 1.0702203012691314E-12, 0.0, 7.601697049608447E-13, ... ], [ 0.0, 0.0, 0.0, -1.2752021660844548E-12, -7.354117315117037E-13, 0.0, -5.591117846481808E-12, -3.7009284525879593E-13, ... ], [ 0.0, 0.0, 0.0, 1.6000534230897756E-12, -1.0156986363085707E-11, 1.4850343177386094E-12, -3.035793838535028E-12, -1.1502257479811817E-12, ... ], ... ]
    Error Statistics: {meanExponent=-11.900049015209907, negative=104, min=-2.7977620220553945E-13, max=-2.7977620220553945E-13, mean=-2.2379005798695212E-13, count=648.0, positive=92, stdDev=2.0292252933222763E-12, zeros=452}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 1.0164e-12 +- 2.1247e-12 [0.0000e+00 - 1.4688e-11] (972#)
    relativeTol: 3.6344e-12 +- 9.7855e-12 [1.7861e-14 - 6.9889e-11] (392#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=1.0164e-12 +- 2.1247e-12 [0.0000e+00 - 1.4688e-11] (972#), relativeTol=3.6344e-12 +- 9.7855e-12 [1.7861e-14 - 6.9889e-11] (392#)}
```



### Performance
Code from [LayerTestBase.java:149](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L149) executed in 0.86 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 26.3927 +- 2.6087 [22.1058 - 37.0758]
    Learning performance: 25.8048 +- 4.6041 [19.1848 - 51.4643]
    
```

