# ConvolutionLayer
## DownsizeTest
### Json Serialization
Code from [LayerTestBase.java:83](../../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L83) executed in 0.00 seconds: 
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
      "id": "a864e734-2f23-44db-97c1-504000000014",
      "isFrozen": false,
      "name": "ConvolutionLayer/a864e734-2f23-44db-97c1-504000000014",
      "filter": {
        "dimensions": [
          3,
          3,
          21
        ],
        "data": [
          1.704,
          -1.1,
          1.592,
          -0.568,
          -0.216,
          1.868,
          1.384,
          0.064,
          0.052,
          -1.36,
          1.856,
          -0.988,
          -1.584,
          0.792,
          1.152,
          0.98,
          1.184,
          -0.212,
          -1.972,
          1.496,
          -1.648,
          0.824,
          -0.28,
          -1.752,
          1.824,
          -0.26,
          1.792,
          -1.668,
          0.712,
          1.932,
          -0.988,
          0.572,
          -0.688,
          1.02,
          1.0,
          -0.032,
          0.8,
          -1.156,
          0.704,
          1.5,
          1.344,
          -0.44,
          -0.216,
          0.456,
          1.052,
          -0.328,
          0.74,
          0.98,
          1.208,
          -0.42,
          1.088,
          -0.464,
          0.704,
          -0.272,
          1.348,
          -0.204,
          -0.548,
          -0.844,
          0.492,
          -0.444,
          1.392,
          1.932,
          1.336,
          1.492,
          -0.704,
          0.648,
          -1.792,
          1.464,
          1.312,
          1.156,
          -1.204,
          0.42,
          0.116,
          -0.176,
          0.408,
          1.108,
          0.852,
          -0.496,
          -1.22,
          1.208,
          -0.784,
          1.12,
          1.784,
          -0.024,
          -0.16,
          0.18,
          -1.36,
          0.632,
          -0.06,
          1.936,
          0.532,
          -0.072,
          1.932,
          1.292,
          -1.764,
          -0.424,
          1.044,
          1.044,
          -1.908,
          -0.14,
          -0.92,
          0.376,
          -1.804,
          -1.716,
          1.836,
          0.212,
          -0.288,
          0.784,
          -1.392,
          1.568,
          0.764,
          -1.248,
          0.068,
          -0.28,
          0.692,
          -0.08,
          0.684,
          1.416,
          1.684,
          -0.552,
          -0.764,
          -1.24,
          -0.308,
          0.492,
          0.928,
          -0.108,
          -1.756,
          0.344,
          -1.944,
          -1.96,
          -0.724,
          -0.612,
          0.332,
          -0.292,
          -0.548,
          -1.844,
          -1.1,
          1.704,
          1.06,
          0.376,
          1.408,
          -1.616,
          -1.824,
          -1.708,
          0.428,
          0.408,
          -0.74,
          -0.096,
          0.384,
          1.728,
          1.264,
          0.972,
          -1.108,
          -0.216,
          0.016,
          -1.484,
          -1.116,
          0.132,
          1.892,
          1.684,
          -1.284,
          0.924,
          -1.572,
          -1.944,
          1.848,
          -1.208,
          0.844,
          -1.888,
          -0.292,
          0.824,
          -1.924,
          -0.044,
          -0.568,
          0.356,
          1.096,
          -1.844,
          0.516,
          -0.636,
          1.104,
          0.312,
          0.848,
          -1.02,
          -1.576,
          1.9,
          0.268,
          1.272,
          -1.308,
          -0.72,
          -0.692
        ]
      },
      "skip": {
        "dimensions": [
          1,
          1
        ]
      },
      "simple": false
    }
```



### Example Input/Output Pair
Code from [LayerTestBase.java:120](../../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L120) executed in 0.01 seconds: 
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
    	[ [ -2.0, -0.596, 0.62, 1.812, -0.112, -1.908, 1.06 ], [ -0.344, 1.268, -1.408, 0.628, -1.908, 0.988, 1.836 ], [ 1.996, -1.248, -1.776, -0.768, -1.176, 1.612, -0.58 ] ],
    	[ [ -1.208, 1.384, -0.324, -0.48, 0.964, 0.212, 0.508 ], [ -0.04, 1.052, 1.016, 1.668, 1.44, 1.316, -0.064 ], [ 0.512, -1.212, 0.58, -0.228, -1.216, 1.292, -1.544 ] ],
    	[ [ 1.316, -0.572, 0.812, -1.284, -1.116, -0.284, -0.34 ], [ -0.6, -0.532, 1.416, -0.3, 1.116, 1.368, -0.16 ], [ 0.436, -0.632, 0.88, -0.804, 1.612, -0.332, 0.804 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 2.4592639999999997, 3.1103680000000007, 5.465408000000001 ] ]
    ]
```



### Batch Execution
Code from [LayerTestBase.java:138](../../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L138) executed in 0.09 seconds: 
```java
    BatchingTester batchingTester = getBatchingTester();
    return batchingTester==null?null:batchingTester.test(layer, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (660#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (100#)}
```



### Differential Validation
Code from [LayerTestBase.java:144](../../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 1.18 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ -2.0, -0.596, 0.62, 1.812, -0.112, -1.908, 1.06 ], [ -0.344, 1.268, -1.408, 0.628, -1.908, 0.988, 1.836 ], [ 1.996, -1.248, -1.776, -0.768, -1.176, 1.612, -0.58 ] ],
    	[ [ -1.208, 1.384, -0.324, -0.48, 0.964, 0.212, 0.508 ], [ -0.04, 1.052, 1.016, 1.668, 1.44, 1.316, -0.064 ], [ 0.512, -1.212, 0.58, -0.228, -1.216, 1.292, -1.544 ] ],
    	[ [ 1.316, -0.572, 0.812, -1.284, -1.116, -0.284, -0.34 ], [ -0.6, -0.532, 1.416, -0.3, 1.116, 1.368, -0.16 ], [ 0.436, -0.632, 0.88, -0.804, 1.612, -0.332, 0.804 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.12352352203343947, negative=33, min=0.804, max=0.804, mean=0.10203174603174604, count=63.0, positive=30, stdDev=1.0999174279001762, zeros=0}
    Output: [
    	[ [ 2.4592639999999997, 3.1103680000000007, 5.465408000000001 ] ]
    ]
    Outputs Statistics: {meanExponent=0.5404131719112862, negative=0, min=5.465408000000001, max=5.465408000000001, mean=3.6783466666666675, count=3.0, positive=3, stdDev=1.2912978643017343, zeros=0}
    Feedback for input 0
    Inputs Values: [
    	[ [ -2.0, -0.596, 0.62, 1.812, -0.112, -1.908, 1.06 ], [ -0.344, 1.268, -1.408, 0.628, -1.908, 0.988, 1.836 ], [ 1.996, -1.248, -1.776, -0.768, -1.176, 1.612, -0.58 ] ],
    	[ [ -1.208, 1.384, -0.324, -0.48, 0.964, 0.212, 0.508 ], [ -0.04, 1.052, 1.016, 1.668, 1.44, 1.316, -0.064 ], [ 0.512, -1.212, 0.58, -0.228, -1.216, 1.292, -1.544 ] ],
    	[ [ 1.316, -0.572, 0.812, -1.284, -1.116, -0.284, -0.34 ], [ -0.6, -0.532, 1.416, -0.3, 1.116, 1.368, -0.16 ], [ 0.436, -0.632, 0.88, -0.804, 1.612, -0.332, 0.804 ] ]
    ]
    Value Statistics: {meanExponent=-0.12352352203343947, negative=33, min=0.804, max=0.804, mean=0.10203174603174604, count=63.0, positive=30, stdDev=1.0999174279001762, zeros=0}
    Implemented Feedback: [ [ 1.704, -1.36, -1.972 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], ... ]
    Implemented Statistics: {meanExponent=-0.1366935148026594, negative=11, min=0.0, max=0.0, mean=-0.013164021164021166, count=189.0, positive=10, stdDev=0.40
```
...[skipping 217 bytes](etc/1.txt)...
```
    0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], ... ]
    Measured Statistics: {meanExponent=-0.13669351480288283, negative=11, min=0.0, max=0.0, mean=-0.013164021164023756, count=189.0, positive=10, stdDev=0.40890458852298456, zeros=168}
    Feedback Error: [ [ 1.4988010832439613E-13, -3.58046925441613E-12, -5.635936162207145E-12 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], ... ]
    Error Statistics: {meanExponent=-11.690564188778255, negative=10, min=0.0, max=0.0, mean=-2.5918420848689666E-15, count=189.0, positive=11, stdDev=1.1803378081684277E-12, zeros=168}
    Learning Gradient for weight set 0
    Weights: [ 1.704, -1.1, 1.592, -0.568, -0.216, 1.868, 1.384, 0.064, ... ]
    Implemented Gradient: [ [ -2.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], ... ]
    Implemented Statistics: {meanExponent=-0.07401022906481516, negative=12, min=0.0, max=0.0, mean=-0.005947089947089945, count=567.0, positive=9, stdDev=0.2601676654231714, zeros=546}
    Measured Gradient: [ [ -1.9999999999953388, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], ... ]
    Measured Statistics: {meanExponent=-0.07401022906518306, negative=12, min=0.0, max=0.0, mean=-0.005947089947084734, count=567.0, positive=9, stdDev=0.2601676654232654, zeros=546}
    Gradient Error: [ [ 4.661160346586257E-12, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], ... ]
    Error Statistics: {meanExponent=-11.648683190992742, negative=10, min=0.0, max=0.0, mean=5.212247447289467E-15, count=567.0, positive=11, stdDev=6.236466788800083E-13, zeros=546}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 1.5803e-13 +- 7.8425e-13 [0.0000e+00 - 6.9267e-12] (756#)
    relativeTol: 3.0911e-12 +- 4.8678e-12 [4.3979e-14 - 2.5273e-11] (42#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=1.5803e-13 +- 7.8425e-13 [0.0000e+00 - 6.9267e-12] (756#), relativeTol=3.0911e-12 +- 4.8678e-12 [4.3979e-14 - 2.5273e-11] (42#)}
```



### Performance
Code from [LayerTestBase.java:149](../../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L149) executed in 0.82 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 30.5074 +- 4.0261 [25.0525 - 45.4285]
    Learning performance: 21.7806 +- 2.5183 [18.9938 - 33.6332]
    
```

