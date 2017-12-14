# ConvolutionLayer
## DownsizeTest
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
      "id": "81b975a7-b3fe-429e-b768-6d5744ba39f9",
      "isFrozen": false,
      "name": "ConvolutionLayer/81b975a7-b3fe-429e-b768-6d5744ba39f9",
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
...[skipping 1958 bytes](etc/40.txt)...
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
            -1.176
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
    	[ [ 1.916, -0.6, -0.968, -1.384, 0.22, 1.708, 1.4 ], [ -1.54, 1.092, -1.992, -0.088, 1.092, -0.944, 0.964 ], [ 0.852, -1.76, -0.628, 1.072, 1.832, -1.464, -0.696 ] ],
    	[ [ 1.196, 1.676, -1.808, 1.736, -0.14, -0.208, -1.516 ], [ 1.08, 1.828, -1.904, 1.616, 1.588, -1.66, -1.06 ], [ -1.968, 1.868, 0.704, 1.332, -1.264, -1.784, 1.652 ] ],
    	[ [ -0.052, -0.84, -0.94, -0.02, -0.692, -1.832, -1.284 ], [ 0.308, 1.804, -0.072, -0.232, 0.968, 0.292, 1.576 ], [ 0.636, 0.156, -1.772, 1.396, -0.696, 1.736, 1.756 ] ]
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
Code from [StandardLayerTests.java:101](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L101) executed in 0.08 seconds: 
```java
    return getBatchingTester().test(layer, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (660#), relativeTol=0.0000e+00 +- 0.0000e+00 [Infinity - -Infinity] (0#)}
```



### Differential Validation
Code from [StandardLayerTests.java:109](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L109) executed in 1.38 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ 0.576, -1.664, 0.42, 1.392, -0.952, -1.772, 1.956 ], [ -0.456, 1.12, -0.644, 0.032, 0.64, -0.588, 1.62 ], [ 0.016, 0.968, -0.948, 1.696, -1.608, 1.396, -0.744 ] ],
    	[ [ -0.596, -1.228, 1.984, 1.392, -0.848, -1.496, 1.86 ], [ 1.136, 0.104, -1.672, 1.764, 1.104, -1.64, -1.128 ], [ -0.34, -0.876, -1.084, 1.46, 1.576, -1.188, -0.248 ] ],
    	[ [ 1.568, 1.912, 0.56, -1.52, 0.628, -1.74, 0.912 ], [ 1.056, 0.524, -0.872, -1.272, -0.892, -1.664, 0.46 ], [ -1.78, -0.048, 0.62, 0.752, -0.6, -0.624, 0.428 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.08893940892814015, negative=31, min=0.428, max=0.428, mean=0.014285714285714316, count=63.0, positive=32, stdDev=1.1846327597657442, zeros=0}
    Output: [
    	[ [ 0.0, 0.0, 0.0 ] ]
    ]
    Outputs Statistics: {meanExponent=NaN, negative=0, min=0.0, max=0.0, mean=0.0, count=3.0, positive=0, stdDev=0.0, zeros=3}
    Feedback for input 0
    Inputs Values: [
    	[ [ 0.576, -1.664, 0.42, 1.392, -0.952, -1.772, 1.956 ], [ -0.456, 1.12, -0.644, 0.032, 0.64, -0.588, 1.62 ], [ 0.016, 0.968, 
```
...[skipping 1810 bytes](etc/41.txt)...
```
    3340677677853, negative=9, min=0.0, max=0.0, mean=-2.3280423280423458E-4, count=567.0, positive=12, stdDev=0.26314087558827454, zeros=546}
    Measured Gradient: [ [ 0.576, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], ... ]
    Measured Statistics: {meanExponent=0.038133406776778546, negative=9, min=0.0, max=0.0, mean=-2.328042328042342E-4, count=567.0, positive=12, stdDev=0.26314087558827454, zeros=546}
    Gradient Error: [ [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], ... ]
    Error Statistics: {meanExponent=-15.653559774527023, negative=3, min=0.0, max=0.0, mean=0.0, count=567.0, positive=3, stdDev=2.2841484253961028E-17, zeros=561}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 1.7623e-18 +- 1.9703e-17 [0.0000e+00 - 2.2204e-16] (756#)
    relativeTol: 1.7640e-17 +- 2.8018e-17 [0.0000e+00 - 6.6720e-17] (21#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=1.7623e-18 +- 1.9703e-17 [0.0000e+00 - 2.2204e-16] (756#), relativeTol=1.7640e-17 +- 2.8018e-17 [0.0000e+00 - 6.6720e-17] (21#)}
```



### Performance
Code from [StandardLayerTests.java:119](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L119) executed in 2.55 seconds: 
```java
    getPerformanceTester().test(layer, permPrototype);
```
Logging: 
```
    100 batches
    Input Dimensions:
    	[100, 100, 3]
    Performance:
    	Evaluation performance: 0.155002s +- 0.026213s [0.137739s - 0.207188s]
    	Learning performance: 0.206948s +- 0.062918s [0.173720s - 0.332753s]
    
```

