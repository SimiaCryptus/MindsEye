# ConvolutionLayer
## UpsizeTest
### Json Serialization
Code from [StandardLayerTests.java:68](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L68) executed in 0.00 seconds: 
```java
    JsonObject json = layer.getJson();
    NNLayer echo = NNLayer.fromJson(json);
    if ((echo == null)) throw new AssertionError("Failed to deserialize");
    if ((layer == echo)) throw new AssertionError("Serialization did not copy");
    if ((!layer.equals(echo))) throw new AssertionError("Serialization not equal");
    return new GsonBuilder().setPrettyPrinting().create().toJson(json);
```

Returns: 

```
    {
      "class": "com.simiacryptus.mindseye.layers.aparapi.ConvolutionLayer",
      "id": "96873c4f-cde9-4a09-9930-f60d871cecab",
      "isFrozen": false,
      "name": "ConvolutionLayer/96873c4f-cde9-4a09-9930-f60d871cecab",
      "filter": [
        [
          [
            0.0,
            0.0,
            0.0
          ],
          [
            0.0,
            0.0,
            0.0
          ],
          [
            0.0,
            0.0,
            0.0
          ]
        ],
        [
          [
            0.0,
            0.0,
            0.0
          ],
          [
            0.0,
            0.0,
            0.0
          ],
          [
            0.0,
            0.0,
            0.0
          ]
        ],
        [
          [
            0.0,
            0.0,
            0.0
          ],
          [
            0.0,
            0.0,
            0.0
          ],
          [
            0.0,
            0.0,
            0.0
          ]
        ],
        [
          [
            0.0,
            0.0,
            0.0
          ],
          [
            0.0,
            0.0,
            0.0
          ],
          [
            0.0,
            0.0,
            0.0
          ]
        ],
        [
          [
            0.0,
            0.0,
            0.0
          ],
          [
            0.0,
            0.0,
            0.0
          ],
          [
            0.0,
            0.0,
            0.0
          ]
        ],
        [
          [
            0.0,
            0.0,
            0.0
          ],
          [
            0.0,
            0.0,
            0.0
          ],
          [
            0.0,
            0.0,
            1.968
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
Code from [StandardLayerTests.java:152](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L152) executed in 0.01 seconds: 
```java
    SimpleEval eval = SimpleEval.run(layer, inputPrototype);
    return String.format("--------------------\nInput: \n[%s]\n--------------------\nOutput: \n%s\n--------------------\nDerivative: \n%s",
      Arrays.stream(inputPrototype).map(t -> t.prettyPrint()).reduce((a, b) -> a + ",\n" + b).get(),
      eval.getOutput().prettyPrint(),
      Arrays.stream(eval.getDerivative()).map(t -> t.prettyPrint()).reduce((a, b) -> a + ",\n" + b).get());
```

Returns: 

```
    --------------------
    Input: 
    [[
    	[ [ -1.464, 0.26 ], [ 1.848, 1.632 ], [ -1.7, -1.816 ] ],
    	[ [ 0.384, 1.052 ], [ -1.728, 1.724 ], [ -0.776, 0.032 ] ],
    	[ [ 0.26, -0.18 ], [ 1.064, -1.908 ], [ -1.988, -0.128 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 0.0, 0.0, 0.0 ] ]
    ]
    --------------------
    Derivative: 
    [
    	[ [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ] ],
    	[ [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ] ],
    	[ [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ] ]
    ]
```



### Batch Execution
Code from [StandardLayerTests.java:101](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L101) executed in 0.08 seconds: 
```java
    return getBatchingTester().test(layer, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (210#), relativeTol=0.0000e+00 +- 0.0000e+00 [Infinity - -Infinity] (0#)}
```



### Differential Validation
Code from [StandardLayerTests.java:109](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L109) executed in 0.31 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ -1.524, -1.956 ], [ -1.316, 0.792 ], [ 1.204, 1.684 ] ],
    	[ [ 1.196, 0.648 ], [ 0.676, -0.612 ], [ -1.804, -1.372 ] ],
    	[ [ 0.58, 0.404 ], [ -0.408, 0.372 ], [ 1.992, 1.38 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.017273372626160813, negative=7, min=1.38, max=1.38, mean=0.10755555555555554, count=18.0, positive=11, stdDev=1.2265680314611453, zeros=0}
    Output: [
    	[ [ 0.0, 0.0, 0.0 ] ]
    ]
    Outputs Statistics: {meanExponent=NaN, negative=0, min=0.0, max=0.0, mean=0.0, count=3.0, positive=0, stdDev=0.0, zeros=3}
    Feedback for input 0
    Inputs Values: [
    	[ [ -1.524, -1.956 ], [ -1.316, 0.792 ], [ 1.204, 1.684 ] ],
    	[ [ 1.196, 0.648 ], [ 0.676, -0.612 ], [ -1.804, -1.372 ] ],
    	[ [ 0.58, 0.404 ], [ -0.408, 0.372 ], [ 1.992, 1.38 ] ]
    ]
    Value Statistics: {meanExponent=-0.017273372626160813, negative=7, min=1.38, max=1.38, mean=0.10755555555555554, count=18.0, positive=11, stdDev=1.2265680314611453, zeros=0}
    Implemented Feedback: [ [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0
```
...[skipping 1191 bytes](etc/42.txt)...
```
    0, mean=-0.06444444444444443, count=162.0, positive=0, stdDev=0.33122235267804656, zeros=156}
    Measured Gradient: [ [ -1.5240000000000002, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], ... ]
    Measured Statistics: {meanExponent=0.2371769087275822, negative=6, min=0.0, max=0.0, mean=-0.06444444444444446, count=162.0, positive=0, stdDev=0.33122235267804656, zeros=156}
    Gradient Error: [ [ -2.220446049250313E-16, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], ... ]
    Error Statistics: {meanExponent=-15.653559774527023, negative=6, min=0.0, max=0.0, mean=-8.22387425648264E-18, count=162.0, positive=0, stdDev=4.193369531113834E-17, zeros=156}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 6.1679e-18 +- 3.6490e-17 [0.0000e+00 - 2.2204e-16] (216#)
    relativeTol: 6.4805e-17 +- 8.0447e-18 [5.6760e-17 - 7.2849e-17] (6#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=6.1679e-18 +- 3.6490e-17 [0.0000e+00 - 2.2204e-16] (216#), relativeTol=6.4805e-17 +- 8.0447e-18 [5.6760e-17 - 7.2849e-17] (6#)}
```



### Performance
Code from [StandardLayerTests.java:119](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L119) executed in 2.00 seconds: 
```java
    getPerformanceTester().test(layer, permPrototype);
```
Logging: 
```
    100 batches
    Input Dimensions:
    	[100, 100, 3]
    Performance:
    	Evaluation performance: 0.078171s +- 0.002749s [0.075218s - 0.082255s]
    	Learning performance: 0.243510s +- 0.001137s [0.241890s - 0.245232s]
    
```

