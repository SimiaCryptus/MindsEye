# ConvolutionLayer
## DownsizeTest
### Json Serialization
Code from [StandardLayerTests.java:69](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L69) executed in 0.00 seconds: 
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
      "id": "a2303731-f558-4038-80f4-747b2e9e7e80",
      "isFrozen": false,
      "name": "ConvolutionLayer/a2303731-f558-4038-80f4-747b2e9e7e80",
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
            0.
```
...[skipping 1957 bytes](etc/2.txt)...
```
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
            0.432
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
Code from [StandardLayerTests.java:153](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L153) executed in 0.01 seconds: 
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
    	[ [ -1.084, -0.456, -0.244, 0.212, 1.88, 1.996, -1.788 ], [ -1.668, 0.648, 1.128, 1.508, 0.02, -1.964, 0.312 ], [ -0.368, 1.844, -1.78, -0.852, 1.352, 1.188, 0.96 ] ],
    	[ [ 0.372, -1.568, 1.636, -0.588, 0.66, -1.312, 1.012 ], [ -0.556, -1.372, -1.32, -0.24, 0.008, -1.684, -0.276 ], [ 1.68, -0.232, 1.604, 0.528, 0.728, -1.572, -0.536 ] ],
    	[ [ -0.204, -0.432, -0.508, -0.564, -0.364, 1.332, 0.384 ], [ -0.564, 1.46, 0.128, 1.172, -0.712, -0.42, -1.832 ], [ 0.72, 0.744, -1.372, 0.556, -0.16, 1.98, -0.032 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 0.0, 0.0, 0.0 ] ]
    ]
    --------------------
    Derivative: 
    [
    	[ [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ] ],
    	[ [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ] ],
    	[ [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ] ]
    ]
```



### Batch Execution
Code from [StandardLayerTests.java:102](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L102) executed in 0.08 seconds: 
```java
    return getBatchingTester().test(layer, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (660#), relativeTol=0.0000e+00 +- 0.0000e+00 [Infinity - -Infinity] (0#)}
```



### Differential Validation
Code from [StandardLayerTests.java:110](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L110) executed in 1.03 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ -1.72, -1.284, 1.26, 0.468, -0.276, -1.64, -0.896 ], [ -0.48, 1.832, -1.032, -1.624, 0.22, -1.948, -0.312 ], [ 1.34, 0.464, -1.6, 0.768, -1.476, 0.724, 1.788 ] ],
    	[ [ 0.004, 0.62, -0.764, -0.448, 1.276, -0.82, 1.636 ], [ 1.052, 1.148, 0.632, 1.68, 0.456, -0.924, -1.696 ], [ 0.328, 1.164, 1.12, 1.04, -0.192, 1.848, 0.844 ] ],
    	[ [ 1.028, -1.144, -0.008, -1.656, -0.504, -0.372, 0.556 ], [ 1.96, 1.168, 1.152, -0.608, 0.988, -0.708, -1.328 ], [ -1.128, -1.764, -1.86, 0.412, 0.308, -0.804, -1.688 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.11716721799123919, negative=31, min=-1.688, max=-1.688, mean=-0.02253968253968254, count=63.0, positive=32, stdDev=1.1511931260673256, zeros=0}
    Output: [
    	[ [ 0.0, 0.0, 0.0 ] ]
    ]
    Outputs Statistics: {meanExponent=NaN, negative=0, min=0.0, max=0.0, mean=0.0, count=3.0, positive=0, stdDev=0.0, zeros=3}
    Feedback for input 0
    Inputs Values: [
    	[ [ -1.72, -1.284, 1.26, 0.468, -0.276, -1.64, -0.896 ], [ -0.48, 1.832, -1.032, -1.624, 0.22, -1.948, -0.312 ], [ 1.34, 0.
```
...[skipping 1836 bytes](etc/3.txt)...
```
    tive=15, min=0.0, max=0.0, mean=-0.021629629629629634, count=567.0, positive=6, stdDev=0.2287977558682318, zeros=546}
    Measured Gradient: [ [ -1.72, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], ... ]
    Measured Statistics: {meanExponent=-0.039604170198984094, negative=15, min=0.0, max=0.0, mean=-0.021629629629629634, count=567.0, positive=6, stdDev=0.22879775586823187, zeros=546}
    Gradient Error: [ [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], ... ]
    Error Statistics: {meanExponent=-15.954589770191003, negative=6, min=0.0, max=0.0, mean=-1.4685489743719001E-18, count=567.0, positive=3, stdDev=1.8445349266921272E-17, zeros=558}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 1.5420e-18 +- 1.5950e-17 [0.0000e+00 - 2.2204e-16] (756#)
    relativeTol: 2.6994e-17 +- 3.1254e-17 [0.0000e+00 - 6.7697e-17] (21#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=1.5420e-18 +- 1.5950e-17 [0.0000e+00 - 2.2204e-16] (756#), relativeTol=2.6994e-17 +- 3.1254e-17 [0.0000e+00 - 6.7697e-17] (21#)}
```



### Performance
Code from [StandardLayerTests.java:120](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L120) executed in 0.04 seconds: 
```java
    getPerformanceTester().test(layer, permPrototype);
```
Logging: 
```
    Evaluation performance: 0.002018s +- 0.000599s [0.001558s - 0.003190s]
    Learning performance: 0.001852s +- 0.000348s [0.001563s - 0.002509s]
    
```

