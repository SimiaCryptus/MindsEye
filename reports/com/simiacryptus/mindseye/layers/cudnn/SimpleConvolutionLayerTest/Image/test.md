# SimpleConvolutionLayer
## Image
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
      "class": "com.simiacryptus.mindseye.layers.cudnn.SimpleConvolutionLayer",
      "id": "fcfb3745-f7c4-402b-9f9b-665165be0281",
      "isFrozen": false,
      "name": "SimpleConvolutionLayer/fcfb3745-f7c4-402b-9f9b-665165be0281",
      "filter": [
        [
          [
            1.688,
            -1.076,
            0.412
          ],
          [
            -1.428,
            0.84,
            -0.836
          ],
          [
            -0.552,
            1.28,
            -0.804
          ]
        ],
        [
          [
            0.004,
            -0.236,
            -1.804
          ],
          [
            -1.672,
            1.808,
            0.252
          ],
          [
            -0.008,
            0.364,
            -0.64
          ]
        ],
        [
          [
            1.276,
            1.744,
            -0.384
          ],
          [
            -0.372,
            0.308,
            1.196
          ],
          [
            1.912,
            1.572,
            0.496
          ]
        ],
        [
          [
            -1.728,
            1.212,
            -0.988
          ],
          [
            0.052,
            1.476,
            -0.756
          ],
          [
            -0.592,
            0.536,
            0.968
          ]
        
```
...[skipping 17 bytes](etc/83.txt)...
```
            0.672,
            -1.036,
            1.496
          ],
          [
            0.252,
            -0.46,
            -0.116
          ],
          [
            1.76,
            -1.136,
            0.856
          ]
        ],
        [
          [
            0.384,
            -0.852,
            0.332
          ],
          [
            -0.18,
            0.052,
            1.56
          ],
          [
            0.612,
            1.188,
            -1.788
          ]
        ],
        [
          [
            -1.588,
            1.216,
            -0.052
          ],
          [
            -0.628,
            1.716,
            -1.252
          ],
          [
            1.168,
            1.592,
            1.04
          ]
        ],
        [
          [
            -0.14,
            0.32,
            0.232
          ],
          [
            -0.884,
            -0.56,
            1.916
          ],
          [
            1.236,
            -0.676,
            0.568
          ]
        ],
        [
          [
            0.176,
            1.66,
            -1.66
          ],
          [
            0.628,
            -0.776,
            0.504
          ],
          [
            1.728,
            -1.744,
            0.536
          ]
        ]
      ],
      "strideX": 1,
      "strideY": 1,
      "simple": false
    }
```



### Example Input/Output Pair
Code from [StandardLayerTests.java:152](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L152) executed in 0.00 seconds: 
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
    	[ [ 1.68, 0.112, 1.06 ], [ -1.388, -1.936, -1.488 ], [ 0.628, 0.464, -1.968 ] ],
    	[ [ 1.944, 1.72, -1.056 ], [ -0.58, 1.788, -1.28 ], [ 0.832, 0.404, -0.668 ] ],
    	[ [ 1.896, -1.46, -1.484 ], [ 1.944, 0.056, -1.488 ], [ -0.164, 1.556, 0.172 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ -6.568719999999999, 6.511359999999999, -5.738848 ], [ -9.72184, 3.212368, -2.691296 ], [ -7.816784, 3.270927999999999, 2.06064 ] ],
    	[ [ 7.439536000000001, -4.985136, -2.515984 ], [ -6.41592, -7.374976, 1.5156000000000003 ], [ -8.375039999999998, -8.40448, -3.80568 ] ],
    	[ [ -11.631856, 6.4651999999999985, 8.033648 ], [ -2.9647840000000003, 4.757807999999999, 18.735632000000003 ], [ 0.19247999999999987, -0.1984639999999999, 3.48784 ] ]
    ]
    --------------------
    Derivative: 
    [
    	[ [ 5.8, 2.1759999999999997, 3.104 ], [ 6.523999999999999, 1.1480000000000001, 3.944 ], [ 1.912, 1.812, 3.684 ] ],
    	[ [ 3.82, 2.86, 7.432 ], [ 2.916, 2.368, 10.108 ], [ -1.7200000000000006, 0.04400000000000026, 5.596 ] ],
    	[ [ 5.460000000000001, 0.024000000000000243, 4.928 ], [ 5.184000000000001, -0.3919999999999997, 9.316 ], [ 1.7519999999999998, -1.932, 4.048 ] ]
    ]
```



[GPU Log](etc/cuda.log)

### Reference Implementation
Code from [StandardLayerTests.java:92](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L92) executed in 0.01 seconds: 
```java
    System.out.println(new GsonBuilder().setPrettyPrinting().create().toJson(referenceLayer.getJson()));
    getEquivalencyTester().test(referenceLayer, layer, inputPrototype);
```
Logging: 
```
    {
      "class": "com.simiacryptus.mindseye.layers.aparapi.ConvolutionLayer",
      "id": "a00c0c9c-5a58-4dbb-bf8f-e2b31f986f38",
      "isFrozen": false,
      "name": "ConvolutionLayer/a00c0c9c-5a58-4dbb-bf8f-e2b31f986f38",
      "filter": [
        [
          [
            1.688,
            -1.076,
            0.412
          ],
          [
            -1.428,
            0.84,
            -0.836
          ],
          [
            -0.552,
            1.28,
            -0.804
          ]
        ],
        [
          [
            -1.728,
            1.212,
            -0.988
          ],
          [
            0.052,
            1.476,
            -0.756
          ],
          [
            -0.592,
            0.536,
            0.968
          ]
        ],
        [
          [
            -1.588,
            1.216,
            -0.052
          ],
          [
            -0.628,
            1.716,
            -1.252
          ],
          [
            1.168,
            1.592,
            1.04
          ]
        ],
        [
          [
            0.004,
            -0.236,
            -1.804
          ],
          [
            -1.672,
            1.808,
            0.252
          ],
          [
            -0.008,
            0.364,
            -0.64
          ]
        ],
        [
```
...[skipping 615 bytes](etc/84.txt)...
```
          -0.852,
            0.332
          ],
          [
            -0.18,
            0.052,
            1.56
          ],
          [
            0.612,
            1.188,
            -1.788
          ]
        ],
        [
          [
            0.176,
            1.66,
            -1.66
          ],
          [
            0.628,
            -0.776,
            0.504
          ],
          [
            1.728,
            -1.744,
            0.536
          ]
        ]
      ],
      "skip": [
        [
          0.0
        ]
      ],
      "simple": true
    }
    Inputs: [
    	[ [ 1.776, -0.604, 1.176 ], [ 0.468, 0.312, 0.36 ], [ 1.9, 1.148, -1.46 ] ],
    	[ [ -1.36, -1.232, -0.724 ], [ -0.04, 1.636, 0.292 ], [ 1.428, 1.036, 1.88 ] ],
    	[ [ 0.62, 0.032, 1.984 ], [ 1.44, -0.884, -0.364 ], [ 0.388, -1.908, 1.4 ] ]
    ]
    Error: [
    	[ [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ] ],
    	[ [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ] ],
    	[ [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ] ]
    ]
    Accuracy:
    absoluteTol: 0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (27#)
    relativeTol: 0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (27#)
    
```

### Batch Execution
Code from [StandardLayerTests.java:101](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L101) executed in 0.01 seconds: 
```java
    return getBatchingTester().test(layer, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=4.5231e-18 +- 5.6349e-17 [0.0000e+00 - 8.8818e-16] (540#), relativeTol=3.4119e-18 +- 5.1787e-17 [0.0000e+00 - 8.4966e-16] (540#)}
```



### Differential Validation
Code from [StandardLayerTests.java:109](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L109) executed in 0.12 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ 0.212, -0.18, 0.604 ], [ 1.084, -1.292, 1.516 ], [ 1.624, -0.256, -0.052 ] ],
    	[ [ -1.048, 1.3, -0.616 ], [ 0.976, 0.756, 0.036 ], [ -0.196, -1.512, 0.884 ] ],
    	[ [ 0.7, -1.32, -1.552 ], [ 1.244, 1.648, 1.248 ], [ 0.004, 0.804, -0.648 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.27132103217026154, negative=11, min=-0.648, max=-0.648, mean=0.22103703703703706, count=27.0, positive=16, stdDev=0.9869288496055901, zeros=0}
    Output: [
    	[ [ 3.069647999999999, 1.0071200000000002, 0.8898559999999998 ], [ -4.070576, 8.014896000000002, 1.3577760000000005 ], [ 6.188816, 6.586288000000001, 7.077152000000001 ] ],
    	[ [ 8.293264, -4.085344, -5.972336 ], [ -4.709136, -5.585711999999999, -7.371296000000001 ], [ -1.2447359999999987, -0.6398720000000004, 3.986752 ] ],
    	[ [ -2.336080000000001, 0.15596800000000013, 10.813808000000002 ], [ 2.490671999999999, -0.44656000000000023, 1.8982080000000001 ], [ 5.000864, 3.109792, -1.9929599999999998 ] ]
    ]
    Outputs Statistics: {meanExponent=0.4463980288638909, negative=11, mi
```
...[skipping 6701 bytes](etc/85.txt)...
```
    ], [ 3.875760823390806E-12, 9.832135106080386E-12, 1.4499512701604544E-13, 2.305267088331675E-12, 1.4201972931005002E-12, 6.906253346983249E-12, -2.1507240433038533E-12, 1.0252354520901008E-12, ... ], [ 0.0, 8.316652921891432E-12, 9.50350909079134E-13, 0.0, 2.305267088331675E-12, 1.4201972931005002E-12, 0.0, -2.1507240433038533E-12, ... ], [ 0.0, 0.0, 0.0, -7.931433287922118E-12, 1.4499512701604544E-13, 0.0, 1.4201972931005002E-12, -1.9755308500180035E-12, ... ], [ 0.0, 0.0, 0.0, -5.651312751098203E-13, 9.50350909079134E-13, 1.4499512701604544E-13, 2.305267088331675E-12, 1.4201972931005002E-12, ... ], ... ]
    Error Statistics: {meanExponent=-11.744030247325908, negative=185, min=1.0352066426300155E-12, max=1.0352066426300155E-12, mean=-7.978835483386697E-14, count=2187.0, positive=256, stdDev=1.954519938796392E-12, zeros=1746}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 9.0858e-13 +- 2.1453e-12 [0.0000e+00 - 1.7404e-11] (2916#)
    relativeTol: 1.4072e-11 +- 7.3262e-11 [3.9124e-14 - 1.4022e-09] (882#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=9.0858e-13 +- 2.1453e-12 [0.0000e+00 - 1.7404e-11] (2916#), relativeTol=1.4072e-11 +- 7.3262e-11 [3.9124e-14 - 1.4022e-09] (882#)}
```



### Performance
Code from [StandardLayerTests.java:119](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L119) executed in 0.90 seconds: 
```java
    getPerformanceTester().test(layer, permPrototype);
```
Logging: 
```
    100 batches
    Input Dimensions:
    	[100, 100, 3]
    Performance:
    	Evaluation performance: 0.031727s +- 0.006348s [0.022685s - 0.038399s]
    	Learning performance: 0.095202s +- 0.013085s [0.078138s - 0.111017s]
    
```

