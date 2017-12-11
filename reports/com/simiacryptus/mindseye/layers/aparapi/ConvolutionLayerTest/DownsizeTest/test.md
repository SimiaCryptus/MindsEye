# ConvolutionLayer
## DownsizeTest
### Json Serialization
Code from [LayerTestBase.java:121](../../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L121) executed in 0.00 seconds: 
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
      "id": "e2a3bda5-e7e7-4c05-aeb3-4ede00000014",
      "isFrozen": false,
      "name": "ConvolutionLayer/e2a3bda5-e7e7-4c05-aeb3-4ede00000014",
      "filter": [
        [
          [
            -0.812,
            -1.396,
            -0.248
          ],
          [
            1.108,
            -1.452,
            1.116
          ],
          [
            0.84,
            0.04,
            1.848
          ]
        ],
        [
          [
            0.588,
            0.576,
            0.26
          ],
          [
            -0.884,
            -1.768,
            -1.284
          ],
          [
            -1.052,
            1.864,
            1.764
          ]
        ],
        [
          [
            0.712,
            -1.296,
            1.72
          ],
          [
            -1.304,
            1.944,
            -0.6
          ],
          [
            0.912,
            1.884,
            0.552
          ]
        ],
        [
          [
            0.2,
            -1.996,
            -0.296
          ],
          [
            0.188,
            -0.2,
            -0.04
          ],
          [
            -1.58,
            1.876,
            -0.192
          ]
        ],
        [
          [
     
```
...[skipping 2384 bytes](etc/2.txt)...
```
    
            -0.372,
            0.984
          ],
          [
            -1.876,
            -0.532,
            0.556
          ],
          [
            1.6,
            -1.592,
            -0.848
          ]
        ],
        [
          [
            1.944,
            1.864,
            -0.288
          ],
          [
            1.128,
            0.248,
            -0.056
          ],
          [
            -0.852,
            0.46,
            0.872
          ]
        ],
        [
          [
            0.672,
            -1.164,
            0.788
          ],
          [
            -0.344,
            0.064,
            1.744
          ],
          [
            1.964,
            0.464,
            0.444
          ]
        ],
        [
          [
            0.2,
            -0.492,
            1.672
          ],
          [
            1.872,
            0.464,
            -1.916
          ],
          [
            -0.984,
            -1.604,
            -0.088
          ]
        ],
        [
          [
            -0.856,
            0.936,
            -1.488
          ],
          [
            -0.408,
            0.828,
            -1.896
          ],
          [
            1.664,
            0.144,
            -1.152
          ]
        ]
      ],
      "skip": [
        [
          0.0
        ]
      ],
      "simple": false
    }
```



### Example Input/Output Pair
Code from [LayerTestBase.java:159](../../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L159) executed in 0.01 seconds: 
```java
    SimpleEval eval = SimpleEval.run(layer, inputPrototype);
    return String.format("--------------------\nInput: \n[%s]\n--------------------\nOutput: \n%s",
      Arrays.stream(inputPrototype).map(t -> t.prettyPrint()).reduce((a, b) -> a + ",\n" + b).get(),
      eval.getOutput().prettyPrint());
```

Returns: 

```
    --------------------
    Input: 
    [[
    	[ [ -1.912, -0.176, -1.416, -0.332, -0.196, 1.388, -1.308 ], [ 1.152, 0.944, 0.856, -0.616, 1.324, 0.984, 1.176 ], [ -0.32, -0.196, -1.1, 0.952, 0.36, 0.576, 1.036 ] ],
    	[ [ 1.196, -0.02, 0.512, 1.668, 0.052, -1.32, -1.792 ], [ 1.732, 0.5, -0.552, 0.004, -0.868, 0.5, -1.908 ], [ -0.88, -0.824, -0.404, -0.764, 1.444, 0.388, 1.18 ] ],
    	[ [ 1.588, -1.828, -0.136, 0.516, 1.896, 1.252, -1.868 ], [ -0.488, 0.344, -0.496, -1.404, -0.588, 0.764, 1.02 ], [ 0.184, -1.044, 1.584, 0.772, -0.556, 0.664, -0.908 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ -0.9431839999999999, 2.652384, 4.23576 ] ]
    ]
```



### Batch Execution
Code from [LayerTestBase.java:178](../../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L178) executed in 0.12 seconds: 
```java
    BatchingTester batchingTester = getBatchingTester();
    return batchingTester == null ? null : batchingTester.test(layer, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (660#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (100#)}
```



### Differential Validation
Code from [LayerTestBase.java:186](../../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L186) executed in 1.37 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ -1.912, -0.176, -1.416, -0.332, -0.196, 1.388, -1.308 ], [ 1.152, 0.944, 0.856, -0.616, 1.324, 0.984, 1.176 ], [ -0.32, -0.196, -1.1, 0.952, 0.36, 0.576, 1.036 ] ],
    	[ [ 1.196, -0.02, 0.512, 1.668, 0.052, -1.32, -1.792 ], [ 1.732, 0.5, -0.552, 0.004, -0.868, 0.5, -1.908 ], [ -0.88, -0.824, -0.404, -0.764, 1.444, 0.388, 1.18 ] ],
    	[ [ 1.588, -1.828, -0.136, 0.516, 1.896, 1.252, -1.868 ], [ -0.488, 0.344, -0.496, -1.404, -0.588, 0.764, 1.02 ], [ 0.184, -1.044, 1.584, 0.772, -0.556, 0.664, -0.908 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.18738745607548185, negative=30, min=-0.908, max=-0.908, mean=0.06806349206349205, count=63.0, positive=33, stdDev=1.0466686058977468, zeros=0}
    Output: [
    	[ [ -0.9431839999999999, 2.652384, 4.23576 ] ]
    ]
    Outputs Statistics: {meanExponent=0.3417213897086057, negative=1, min=4.23576, max=4.23576, mean=1.9816533333333333, count=3.0, positive=2, stdDev=2.1668371665001094, zeros=0}
    Feedback for input 0
    Inputs Values: [
    	[ [ -1.912, -0.176, -1.416, -0.332, -0.196, 1
```
...[skipping 2227 bytes](etc/3.txt)...
```
    an=-0.02091005291005291, count=567.0, positive=3, stdDev=0.2229573650241582, zeros=546}
    Measured Gradient: [ [ -1.9120000000016901, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], ... ]
    Measured Statistics: {meanExponent=-0.17850640367388232, negative=18, min=0.0, max=0.0, mean=-0.020910052910103977, count=567.0, positive=3, stdDev=0.22295736502422145, zeros=546}
    Gradient Error: [ [ -1.6902035326893383E-12, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], ... ]
    Error Statistics: {meanExponent=-12.098486640040306, negative=14, min=0.0, max=0.0, mean=-5.1066049292364E-14, count=567.0, positive=7, stdDev=5.030378199858743E-13, zeros=546}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 1.0599e-13 +- 6.2808e-13 [0.0000e+00 - 6.7164e-12] (756#)
    relativeTol: 2.9609e-12 +- 6.7534e-12 [1.9837e-14 - 4.1133e-11] (42#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=1.0599e-13 +- 6.2808e-13 [0.0000e+00 - 6.7164e-12] (756#), relativeTol=2.9609e-12 +- 6.7534e-12 [1.9837e-14 - 4.1133e-11] (42#)}
```



### Performance
Code from [LayerTestBase.java:192](../../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L192) executed in 0.80 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 28.2874 +- 3.7638 [22.8554 - 45.4143]
    Learning performance: 22.0409 +- 3.7580 [17.5319 - 40.7122]
    
```

