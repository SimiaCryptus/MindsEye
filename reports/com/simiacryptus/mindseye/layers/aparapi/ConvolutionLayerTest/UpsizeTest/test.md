# ConvolutionLayer
## UpsizeTest
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
      "id": "e2a3bda5-e7e7-4c05-aeb3-4ede00000018",
      "isFrozen": false,
      "name": "ConvolutionLayer/e2a3bda5-e7e7-4c05-aeb3-4ede00000018",
      "filter": [
        [
          [
            -0.744,
            -0.812,
            0.744
          ],
          [
            -1.356,
            -0.104,
            1.536
          ],
          [
            -1.592,
            -0.648,
            -0.84
          ]
        ],
        [
          [
            -1.204,
            0.512,
            -0.088
          ],
          [
            -0.076,
            -1.552,
            1.664
          ],
          [
            1.424,
            0.104,
            -0.856
          ]
        ],
        [
          [
            1.68,
            0.264,
            -0.308
          ],
          [
            1.744,
            1.868,
            1.312
          ],
          [
            -0.128,
            0.408,
            -1.384
          ]
        ],
        [
          [
            -1.308,
            1.012,
            -1.356
          ],
          [
            -1.748,
            0.18,
            0.132
          ],
          [
            -1.504,
            -0.44,
            -1.808
          ]
        ],
        [
          [
            0.076,
            0.748,
            -0.144
          ],
          [
            -0.572,
            -1.416,
            -1.528
          ],
          [
            -1.328,
            1.4,
            -0.552
          ]
        ],
        [
          [
            -0.892,
            -1.124,
            1.908
          ],
          [
            -0.456,
            1.324,
            -0.764
          ],
          [
            -1.66,
            -0.2,
            -0.888
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
    	[ [ -0.8, -0.792 ], [ 1.78, -0.068 ], [ 0.424, 0.072 ] ],
    	[ [ 0.656, -0.356 ], [ -0.044, -1.396 ], [ 1.012, 1.0 ] ],
    	[ [ -1.984, -1.744 ], [ -0.928, 1.82 ], [ -0.096, -0.656 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 1.6311360000000001, 0.903008, -0.637536 ] ]
    ]
```



### Batch Execution
Code from [LayerTestBase.java:178](../../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L178) executed in 0.08 seconds: 
```java
    BatchingTester batchingTester = getBatchingTester();
    return batchingTester == null ? null : batchingTester.test(layer, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (210#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (50#)}
```



### Differential Validation
Code from [LayerTestBase.java:186](../../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L186) executed in 0.33 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ -0.8, -0.792 ], [ 1.78, -0.068 ], [ 0.424, 0.072 ] ],
    	[ [ 0.656, -0.356 ], [ -0.044, -1.396 ], [ 1.012, 1.0 ] ],
    	[ [ -1.984, -1.744 ], [ -0.928, 1.82 ], [ -0.096, -0.656 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.2723779185399951, negative=11, min=-0.656, max=-0.656, mean=-0.11666666666666667, count=18.0, positive=7, stdDev=1.0672692048192693, zeros=0}
    Output: [
    	[ [ 1.6311360000000001, 0.903008, -0.637536 ] ]
    ]
    Outputs Statistics: {meanExponent=-0.009104505309338243, negative=1, min=-0.637536, max=-0.637536, mean=0.6322026666666668, count=3.0, positive=2, stdDev=0.9457694600402833, zeros=0}
    Feedback for input 0
    Inputs Values: [
    	[ [ -0.8, -0.792 ], [ 1.78, -0.068 ], [ 0.424, 0.072 ] ],
    	[ [ 0.656, -0.356 ], [ -0.044, -1.396 ], [ 1.012, 1.0 ] ],
    	[ [ -1.984, -1.744 ], [ -0.928, 1.82 ], [ -0.096, -0.656 ] ]
    ]
    Value Statistics: {meanExponent=-0.2723779185399951, negative=11, min=-0.656, max=-0.656, mean=-0.11666666666666667, count=18.0, positive=7, stdDev=1.0672692048192693, zeros=0}
    Impl
```
...[skipping 1569 bytes](etc/4.txt)...
```
    an=-0.02948148148148148, count=162.0, positive=0, stdDev=0.1503286203672106, zeros=156}
    Measured Gradient: [ [ -0.8000000000008001, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], ... ]
    Measured Statistics: {meanExponent=-0.09909241570921039, negative=6, min=0.0, max=0.0, mean=-0.029481481481486296, count=162.0, positive=0, stdDev=0.15032862036723504, zeros=156}
    Gradient Error: [ [ -8.000267115448878E-13, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], ... ]
    Error Statistics: {meanExponent=-12.132843320987623, negative=4, min=0.0, max=0.0, mean=-4.810966440042345E-15, count=162.0, positive=2, stdDev=1.6169538007097543E-13, zeros=156}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 3.0284e-14 +- 1.5312e-13 [0.0000e+00 - 1.4204e-12] (216#)
    relativeTol: 3.4045e-13 +- 2.3488e-13 [5.1692e-14 - 8.8776e-13] (12#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=3.0284e-14 +- 1.5312e-13 [0.0000e+00 - 1.4204e-12] (216#), relativeTol=3.4045e-13 +- 2.3488e-13 [5.1692e-14 - 8.8776e-13] (12#)}
```



### Performance
Code from [LayerTestBase.java:192](../../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L192) executed in 0.81 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 30.1331 +- 5.5078 [22.8012 - 48.7657]
    Learning performance: 21.9947 +- 3.0412 [17.8454 - 32.4364]
    
```

