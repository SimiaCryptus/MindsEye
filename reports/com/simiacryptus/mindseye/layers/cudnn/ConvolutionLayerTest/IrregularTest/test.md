# ConvolutionLayer
## IrregularTest
### Batch Execution
Code from [BatchingTester.java:66](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/BatchingTester.java#L66) executed in 0.02 seconds: 
```java
    return test(reference, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=5.4401e-17 +- 3.3978e-16 [0.0000e+00 - 3.5527e-15] (3000#), relativeTol=1.7965e-17 +- 2.0648e-16 [0.0000e+00 - 5.0465e-15] (3000#)}
```



Code from [SingleDerivativeTester.java:77](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/SingleDerivativeTester.java#L77) executed in 1.00 seconds: 
```java
    return test(component, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ 0.472, 1.5, 1.284, -0.288, -0.352, -1.056, -0.672 ], [ -0.868, -0.752, -0.52, 0.412, -1.672, -0.068, 0.744 ], [ -1.476, -0.956, -0.444, 1.172, 0.88, 1.616, 1.68 ], [ -0.044, -0.328, 0.336, 0.956, 0.624, -0.432, 1.02 ], [ -0.924, -1.9, 0.84, -0.896, 0.304, 1.68, 1.168 ] ],
    	[ [ -1.648, -1.784, 1.548, -0.66, 1.796, 1.16, 0.3 ], [ 1.556, 0.076, 0.652, -0.64, -1.56, -1.416, -1.952 ], [ -1.08, 0.416, 1.824, -1.98, 1.868, -0.076, -1.252 ], [ 1.08, 1.188, -0.176, 0.168, -1.188, -1.488, -0.2 ], [ 0.692, -1.756, -1.476, -0.736, -0.02, -0.956, 1.788 ] ],
    	[ [ 0.408, -0.572, 1.576, 1.176, 0.904, -0.088, 0.808 ], [ 0.584, 0.784, 0.868, 0.6, 0.16, -1.356, 1.824 ], [ 1.5, -0.936, -0.436, 0.66, -1.932, 1.804, -1.76 ], [ -0.072, 0.004, -0.308, -0.228, 0.212, 1.036, 0.208 ], [ 1.392, -0.108, 0.548, -0.704, 0.456, 1.42, 0.344 ] ],
    	[ [ 0.684, -1.528, -0.836, 0.98, -0.048, -1.28, 1.712 ], [ -1.968, 0.648, 0.616, -0.588, 0.716, -0.576, -1.716 ], [ -1.456, -1.984, 1.392, -0.632, -0.408, 0.416, 0.16 ], [ 1.536, 0.40
```
...[skipping 10839 bytes](etc/61.txt)...
```
    .581313189395587E-12, ... ], [ 2.5002222514558525E-13, -2.9805047319086952E-12, 7.291445225376947E-12, -3.1519231669108194E-13, 2.620459405022757E-12, -5.350164755668629E-13, 1.1437073510478513E-11, 3.470668197280702E-12, ... ], [ 0.0, 2.5002222514558525E-13, -2.9805047319086952E-12, -1.5903389716243055E-12, 8.56659188031017E-12, 0.0, -5.350164755668629E-13, -6.326494883523992E-12, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, -2.9805047319086952E-12, 7.291445225376947E-12, -3.1519231669108194E-13, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 9.131806422146838E-12, 1.478306366209381E-11, -1.0472123168625558E-11, ... ], ... ]
    Error Statistics: {meanExponent=-11.578950264932008, negative=2792, min=4.350408921993676E-13, max=4.350408921993676E-13, mean=3.396501025758226E-14, count=39375.0, positive=3123, stdDev=2.8234695601243672E-12, zeros=33460}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 1.0090e-12 +- 3.0996e-12 [0.0000e+00 - 3.5282e-11] (61250#)
    relativeTol: 1.0132e-11 +- 7.4497e-11 [2.4328e-15 - 3.0387e-09] (11830#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=1.0090e-12 +- 3.0996e-12 [0.0000e+00 - 3.5282e-11] (61250#), relativeTol=1.0132e-11 +- 7.4497e-11 [2.4328e-15 - 3.0387e-09] (11830#)}
```



### Reference Implementation
Code from [EquivalencyTester.java:61](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/EquivalencyTester.java#L61) executed in 0.00 seconds: 
```java
    System.out.println(new GsonBuilder().setPrettyPrinting().create().toJson(this.reference.getJson()));
```
Logging: 
```
    {
      "class": "com.simiacryptus.mindseye.layers.aparapi.ConvolutionLayer",
      "id": "bb21bec7-2ae0-4eca-a8a8-11e55cf69bdd",
      "isFrozen": false,
      "name": "ConvolutionLayer/bb21bec7-2ae0-4eca-a8a8-11e55cf69bdd",
      "filter": [
        [
          [
            1.908,
            -0.676,
            0.98
          ],
          [
            0.96,
            -1.188,
            1.6
          ],
          [
            -1.236,
            0.964,
            -1.536
          ]
        ],
        [
          [
            -1.576,
            1.096,
            -0.252
          ],
          [
            -0.38,
            -1.904,
            0.744
          ],
          [
            -0.96,
            -1.18,
            0.992
          ]
        ],
        [
          [
            -1.1,
            1.208,
            -1.816
          ],
          [
            -0.712,
            1.632,
            1.152
          ],
          [
            -1.98,
            1.256,
            -1.676
          ]
        ],
        [
          [
            -0.54,
            -0.284,
            -0.752
          ],
          [
            1.488,
            -1.876,
            0.036
          ],
          [
            1.056,
            -1.76,
            1.944
          ]
        ],
        [
          
```
...[skipping 5145 bytes](etc/62.txt)...
```
        -0.652,
            1.364,
            1.484
          ],
          [
            0.18,
            -0.264,
            -1.856
          ],
          [
            -0.232,
            -1.832,
            -1.576
          ]
        ],
        [
          [
            -0.992,
            0.9,
            1.68
          ],
          [
            -0.148,
            0.764,
            1.816
          ],
          [
            -1.532,
            0.212,
            -0.196
          ]
        ],
        [
          [
            1.38,
            1.704,
            0.54
          ],
          [
            1.808,
            1.064,
            -0.06
          ],
          [
            1.324,
            1.408,
            1.736
          ]
        ],
        [
          [
            0.688,
            1.892,
            -1.568
          ],
          [
            0.76,
            1.956,
            0.316
          ],
          [
            1.484,
            1.676,
            1.956
          ]
        ],
        [
          [
            -0.392,
            1.24,
            -1.06
          ],
          [
            1.136,
            -1.064,
            -1.632
          ],
          [
            1.28,
            0.98,
            -1.54
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
    
```

Code from [EquivalencyTester.java:64](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/EquivalencyTester.java#L64) executed in 0.02 seconds: 
```java
    return test(subject, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ -0.012, -0.032, 0.22, 1.904, 1.168, 1.524, -1.312 ], [ -1.336, 0.952, 0.532, -0.668, -0.564, -1.252, 0.828 ], [ 0.684, 1.588, 1.244, 0.668, -0.764, 0.216, 1.112 ], [ 1.564, -1.46, -1.848, -1.08, -1.436, -1.884, -0.42 ], [ 0.316, -1.236, 1.756, 0.412, 1.228, -0.8, -0.356 ] ],
    	[ [ -0.596, 0.636, 0.34, -1.712, 0.796, 0.864, 1.588 ], [ 0.904, -1.984, 1.404, -0.732, -0.504, -1.136, 0.032 ], [ 0.464, -1.48, 0.712, -0.76, 0.168, 1.74, 0.592 ], [ -0.196, -1.372, 1.116, -1.508, -1.348, 0.156, 0.748 ], [ 0.152, 1.288, 0.4, 1.268, 0.804, 0.768, 0.8 ] ],
    	[ [ 1.36, 1.616, -1.272, -1.636, 1.464, 0.512, -1.7 ], [ 0.112, 1.548, 1.336, -0.796, -0.32, 1.224, -0.564 ], [ 1.488, -1.652, 1.156, 0.228, 0.8, -1.584, 1.528 ], [ -1.54, -1.752, -1.292, -1.532, 1.328, -0.532, 0.996 ], [ 1.76, 1.688, -1.556, 0.024, -1.904, -1.136, 1.164 ] ],
    	[ [ 0.848, 0.212, -0.324, 0.9, -1.344, -1.54, -1.208 ], [ 0.316, 0.392, -0.212, -0.02, 1.428, 1.312, -1.244 ], [ 1.328, -1.496, 1.596, 1.608, 0.564, 0.756, -1.024 ], [ -1.48, 1.72,
```
...[skipping 267 bytes](etc/63.txt)...
```
     1.016, -0.476, -0.992, -1.268, -0.692, 1.484 ], [ 1.416, -0.492, 0.908, -1.18, -0.128, 0.556, -1.24 ] ]
    ]
    Error: [
    	[ [ 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0 ] ],
    	[ [ 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0 ] ],
    	[ [ 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0 ] ],
    	[ [ 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0 ] ],
    	[ [ 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0 ] ]
    ]
    Accuracy:
    absoluteTol: 0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (125#)
    relativeTol: 0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (125#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (125#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (125#)}
```



### Json Serialization
Code from [JsonTest.java:36](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/JsonTest.java#L36) executed in 0.00 seconds: 
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
      "class": "com.simiacryptus.mindseye.layers.cudnn.ConvolutionLayer",
      "id": "d4d5c81b-538d-45e8-916a-a4f14245062a",
      "isFrozen": false,
      "name": "ConvolutionLayer/d4d5c81b-538d-45e8-916a-a4f14245062a",
      "filter": [
        [
          [
            1.908,
            -0.676,
            0.98
          ],
          [
            0.96,
            -1.188,
            1.6
          ],
          [
            -1.236,
            0.964,
            -1.536
          ]
        ],
        [
          [
            -1.576,
            1.096,
            -0.252
          ],
          [
            -0.38,
            -1.904,
            0.744
          ],
          [
            -0.96,
            -1.18,
            0.992
          ]
        ],
        [
          [
            -1.1,
            1.208,
            -1.816
          ],
          [
            -0.712,
            1.632,
            1.152
          ],
          [
            -1.98,
            1.256,
            -1.676
          ]
        ],
        [
          [
            -0.54,
            -0.284,
            -0.752
          ],
          [
            1.488,
            -1.876,
            0.036
          ],
          [
            1.056,
            -1.76,
            1.944
          ]
        ],
        [
          [
```
...[skipping 5116 bytes](etc/64.txt)...
```
    ]
        ],
        [
          [
            -0.652,
            1.364,
            1.484
          ],
          [
            0.18,
            -0.264,
            -1.856
          ],
          [
            -0.232,
            -1.832,
            -1.576
          ]
        ],
        [
          [
            -0.992,
            0.9,
            1.68
          ],
          [
            -0.148,
            0.764,
            1.816
          ],
          [
            -1.532,
            0.212,
            -0.196
          ]
        ],
        [
          [
            1.38,
            1.704,
            0.54
          ],
          [
            1.808,
            1.064,
            -0.06
          ],
          [
            1.324,
            1.408,
            1.736
          ]
        ],
        [
          [
            0.688,
            1.892,
            -1.568
          ],
          [
            0.76,
            1.956,
            0.316
          ],
          [
            1.484,
            1.676,
            1.956
          ]
        ],
        [
          [
            -0.392,
            1.24,
            -1.06
          ],
          [
            1.136,
            -1.064,
            -1.632
          ],
          [
            1.28,
            0.98,
            -1.54
          ]
        ]
      ],
      "strideX": 1,
      "strideY": 1
    }
```



### Example Input/Output Pair
Code from [ReferenceIO.java:68](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/ReferenceIO.java#L68) executed in 0.00 seconds: 
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
    	[ [ 1.924, -0.028, 1.516, -1.184, -1.804, -1.58, -0.52 ], [ -0.684, -0.208, 1.812, 1.48, 0.876, 0.016, 1.252 ], [ 1.796, 1.976, -1.78, -0.92, 0.32, 1.572, -1.98 ], [ -0.916, 1.976, -0.132, 1.444, 0.268, -0.004, 1.116 ], [ -1.388, -1.092, -1.5, -1.464, 1.632, 0.272, -1.512 ] ],
    	[ [ -1.772, 0.92, -1.66, 0.656, -0.608, -0.844, -0.336 ], [ 0.852, 1.048, 1.896, -0.248, -0.196, 0.62, -1.928 ], [ -0.428, 1.964, -0.444, 1.62, 0.564, 1.94, 0.668 ], [ 1.7, -1.88, -0.512, -0.984, 0.604, -0.456, 1.504 ], [ 1.524, 1.5, 1.848, -1.0, -0.808, 1.14, -1.62 ] ],
    	[ [ 0.904, -1.264, 0.928, -0.596, -1.556, -1.3, 1.016 ], [ -1.888, 0.972, -1.868, -0.98, 1.7, 1.044, 1.492 ], [ 0.936, 0.536, 0.532, -1.148, -0.524, -0.592, 1.428 ], [ 1.188, -1.932, 0.424, -0.096, -1.372, -0.684, -0.676 ], [ 1.152, 1.592, -0.784, -1.232, 1.184, 0.04, -0.548 ] ],
    	[ [ -0.036, -1.428, 0.644, 1.368, 1.632, -1.288, 1.228 ], [ 1.536, 1.052, -0.996, -1.84, -0.684, 0.4, 0.276 ], [ 0.484, 1.28, 0.912, -0.364, 1.008, -1.136, -
```
...[skipping 4670 bytes](etc/65.txt)...
```
     8.012, -4.184, 9.924 ], [ -5.283999999999999, -6.904, 19.392, -0.1319999999999999, 21.956, -9.028, 18.131999999999998 ], [ -5.283999999999999, -6.904, 19.392000000000003, -0.13200000000000012, 21.955999999999996, -9.028, 18.131999999999998 ], [ -5.283999999999999, -6.904, 19.392, -0.13200000000000012, 21.956, -9.028000000000002, 18.131999999999998 ], [ -0.5199999999999991, -9.232, 13.052, 0.4119999999999999, 20.103999999999996, -7.832000000000001, 12.983999999999998 ] ],
    	[ [ -6.4559999999999995, -1.9480000000000004, 9.288, -1.4199999999999997, 4.644, -2.8800000000000003, 10.959999999999999 ], [ -5.927999999999999, 0.252, 17.763999999999996, 0.7240000000000002, 12.624, -6.468, 18.092 ], [ -5.927999999999999, 0.252, 17.764, 0.7240000000000002, 12.624000000000002, -6.468, 18.092 ], [ -5.927999999999999, 0.252, 17.764, 0.7240000000000002, 12.624, -6.468, 18.092 ], [ -1.5039999999999996, -1.5519999999999998, 14.247999999999998, 0.04400000000000004, 11.524000000000001, -5.203999999999999, 13.323999999999998 ] ]
    ]
```



[GPU Log](etc/cuda.log)

### Input Learning
In this test, we use a network to learn this target input, given it's pre-evaluated output:

Code from [LearningTester.java:127](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L127) executed in 0.01 seconds: 
```java
    return Arrays.stream(input_target).map(x -> x.prettyPrint()).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [
    	[ [ -1.4, -1.544, -0.78, -0.856, 0.812, -0.42, -1.416 ], [ 0.34, 1.684, -0.704, 0.624, 0.784, 1.1, 1.692 ], [ -1.824, 1.772, -0.052, -0.608, 0.504, -0.108, 0.436 ], [ 0.956, 0.0, -0.856, -1.832, -1.436, -0.876, -0.984 ], [ -0.812, -1.252, -1.208, 1.024, 1.944, 0.748, 0.148 ], [ 1.396, -1.66, 0.916, 0.32, -1.74, -1.06, 1.076 ], [ 0.408, -1.5, 1.436, 0.668, 1.268, 0.268, 0.192 ], [ 1.38, 0.468, -0.544, -1.008, -1.02, -1.908, 0.272 ], ... ],
    	[ [ 0.52, -1.392, 0.708, 0.28, -1.208, -1.708, 1.804 ], [ -0.124, -1.508, -0.668, -0.368, -0.768, 1.132, -0.376 ], [ 0.736, 0.504, -0.248, 1.064, 1.188, 1.448, -0.036 ], [ 0.308, 0.928, 1.976, -0.728, -0.784, -1.064, -1.46 ], [ 0.892, 0.024, -0.224, 1.732, 0.28, 0.016, 1.7 ], [ 0.424, -1.272, 1.036, -0.32, -0.676, 0.46, 1.36 ], [ 0.012, 0.36, -0.804, -0.288, -0.96, -0.564, -1.42 ], [ 0.548, 1.892, -0.876, -0.564, 1.928, -1.116, -0.584 ], ... ],
    	[ [ 0.012, 0.868, 1.172, 1.24, 0.272, -1.636, 0.26 ], [ -1.132, 0.54, 0.828, 1.14, -0.54, 0.644, 1.904 ], [ 1.748, -1.764, 1.92
```
...[skipping 1539 bytes](etc/66.txt)...
```
    136, -0.976, -0.884, -0.768, 0.52, 0.948, 0.324 ], [ -1.696, -1.968, 1.468, 0.88, -1.168, 0.832, 0.864 ], ... ],
    	[ [ 1.648, 1.764, 0.388, -1.384, 1.296, 1.012, 0.704 ], [ 0.14, 1.02, 1.792, 0.228, 0.568, -0.944, -0.288 ], [ -0.056, -1.244, -0.696, -0.516, 0.156, -0.932, -1.388 ], [ -1.192, 0.876, 0.224, 0.14, 0.132, -0.624, -0.016 ], [ -1.152, -0.96, 1.952, -1.652, 1.608, 1.676, 0.084 ], [ 0.516, -0.324, 1.22, -0.708, 0.132, -1.24, 0.884 ], [ -1.168, -0.812, -1.976, 0.676, 1.184, -0.528, -0.248 ], [ 0.748, -1.468, 1.176, 0.384, -1.648, 1.216, -0.164 ], ... ],
    	[ [ 0.636, 0.272, 0.248, 1.9, 0.468, -1.548, 0.356 ], [ 0.276, -0.688, -1.968, -1.832, 0.972, 1.176, 1.58 ], [ -0.356, -1.284, 1.376, -1.348, -1.372, 0.788, 1.056 ], [ 0.412, -1.272, -1.468, 1.208, 0.388, -0.504, -0.268 ], [ -0.216, -1.752, -1.136, -1.636, 0.784, 0.068, 0.492 ], [ -1.108, 0.312, 0.984, -1.176, -0.276, 1.296, 0.76 ], [ 1.328, -1.544, -0.424, -0.216, 1.632, -0.06, 0.02 ], [ -1.0, 1.524, 1.82, -0.976, 0.22, -0.684, -1.964 ], ... ],
    	...
    ]
```



First, we use a conjugate gradient descent method, which converges the fastest for purely linear functions.

Code from [LearningTester.java:225](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L225) executed in 14.74 seconds: 
```java
    return new IterativeTrainer(trainable)
      .setLineSearchFactory(label -> new QuadraticSearch())
      .setOrientation(new GradientDescent())
      .setMonitor(monitor)
      .setTimeout(30, TimeUnit.SECONDS)
      .setMaxIterations(250)
      .setTerminateThreshold(0)
      .run();
```
Logging: 
```
    Constructing line search parameters: GD
    F(0.0) = LineSearchPoint{point=PointSample{avg=223.79596228852782}, derivative=-2.651869395323579}
    New Minimum: 223.79596228852782 > 223.79596228826136
    F(1.0E-10) = LineSearchPoint{point=PointSample{avg=223.79596228826136}, derivative=-2.6518693953216452}, delta = -2.6645352591003757E-10
    New Minimum: 223.79596228826136 > 223.79596228667197
    F(7.000000000000001E-10) = LineSearchPoint{point=PointSample{avg=223.79596228667197}, derivative=-2.6518693953100456}, delta = -1.8558523606770905E-9
    New Minimum: 223.79596228667197 > 223.79596227553242
    F(4.900000000000001E-9) = LineSearchPoint{point=PointSample{avg=223.79596227553242}, derivative=-2.6518693952288492}, delta = -1.2995400311410776E-8
    New Minimum: 223.79596227553242 > 223.7959621975679
    F(3.430000000000001E-8) = LineSearchPoint{point=PointSample{avg=223.7959621975679}, derivative=-2.6518693946604737}, delta = -9.095992936636321E-8
    New Minimum: 223.7959621975679 > 223.79596165181036
    F(2.4010000000000004E-7) = L
```
...[skipping 291251 bytes](etc/67.txt)...
```
    050529275E-9}
    F(307.1127921483299) = LineSearchPoint{point=PointSample{avg=1.1233408047986094E-5}, derivative=8.199425455548425E-9}, delta = 4.936748157440607E-7
    New Minimum: 1.0739733232242033E-5 > 1.0633958611210044E-5
    F(23.624060934486916) = LineSearchPoint{point=PointSample{avg=1.0633958611210044E-5}, derivative=-3.970337165446411E-9}, delta = -1.0577462103198911E-7
    New Minimum: 1.0633958611210044E-5 > 1.050243450902318E-5
    F(165.3684265414084) = LineSearchPoint{point=PointSample{avg=1.050243450902318E-5}, derivative=2.1145441450514516E-9}, delta = -2.3729872321885295E-7
    1.050243450902318E-5 <= 1.0739733232242033E-5
    New Minimum: 1.050243450902318E-5 > 1.0450356159746228E-5
    F(116.11114392669958) = LineSearchPoint{point=PointSample{avg=1.0450356159746228E-5}, derivative=-2.71636124290088E-22}, delta = -2.893770724958048E-7
    Left bracket at 116.11114392669958
    Converged to left
    Iteration 250 complete. Error: 1.0450356159746228E-5 Total: 239521475260315.5600; Orientation: 0.0018; Line Search: 0.0411
    
```

Returns: 

```
    1.0450356159746228E-5
```



This training run resulted in the following regressed input:

Code from [LearningTester.java:144](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L144) executed in 0.02 seconds: 
```java
    return Arrays.stream(input_gd).map(x -> x.prettyPrint()).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [
    	[ [ 0.3294683847245286, -0.5847422985237344, -0.2560207308417578, -1.109665879742548, 0.90669620757578, 0.7440279716258578, 0.3161587823349562 ], [ 1.3510386528701972, 2.1457751808856935, -1.1700531363092508, -0.40890581585686586, 0.8878286086897533, 1.3570498689214052, 0.6537029203711847 ], [ -0.072541270532123, 0.21436783924138086, 0.4005902457167695, -0.8950786326758327, 0.7343827446941908, -0.3511542237824754, -0.04385416266165623 ], [ -0.35843054898307003, -0.2263276298788939, -1.2000513957863532, -1.5901455376151186, -2.8097911030697085, 1.4344424075450088, -2.950422961772199 ], [ -0.5500445101315526, -0.3855761940842599, -0.7751650259794213, -0.05184917968249735, 1.8637539089011126, -0.10171511826319034, 0.7400088811421471 ], [ 1.0675339232878447, -1.7163988666637917, 0.3008868987601653, 1.911318293896016, -2.0184412381272705, -1.7142797633472895, -0.008297317681887543 ], [ 0.0739451804310268, -1.7527466937934713, 1.719225205940832, 0.8726151319266952, -0.36332704602249594, 0.19785737898386974, -0.1
```
...[skipping 7499 bytes](etc/68.txt)...
```
    0.2667011265280559, -0.9092432300547924, -2.1022909249936976, 1.3100469845534732, 0.6187388847843165, 2.239922360898707 ], [ 0.2593021719811797, -1.5484295236434142, 1.4502134495805084, -0.5569994513578757, -1.8927834609812129, 0.913371073775345, -0.929273487736907 ], [ 0.8669317570888804, -1.6649955607582978, -1.4434269861044307, 0.14943571941544442, -1.1903039987622892, 1.0227989511586986, -0.12636793677192448 ], [ -0.09894854593163582, -2.500395167852, -0.3497722101334146, -1.522495591439614, 0.09461139102094868, 0.6411200126712209, 1.0659385417773157 ], [ -0.8370663900728097, 0.8243504983750596, -0.7843093045070271, -1.8971743776603565, 0.1753453904841266, 0.07426436693082492, 0.034952812025546565 ], [ 1.5006475200027265, -1.2898437144822912, -0.3581742537637485, 0.028110447378484147, 0.7746882344626587, -0.9335442556713458, -0.5907035420566974 ], [ 0.2619347133917023, 2.2589731903707175, 0.7785091213364186, -2.3543584865551392, 0.8712871467709412, -1.7041075164285353, -0.2031522031769302 ], ... ],
    	...
    ]
```



Next, we run the same optimization using L-BFGS, which is nearly ideal for purely second-order or quadratic functions.

Code from [LearningTester.java:249](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L249) executed in 6.72 seconds: 
```java
    return new IterativeTrainer(trainable)
      .setLineSearchFactory(label -> new ArmijoWolfeSearch())
      .setOrientation(new LBFGS())
      .setMonitor(monitor)
      .setTimeout(30, TimeUnit.SECONDS)
      .setMaxIterations(250)
      .setTerminateThreshold(0)
      .run();
```
Logging: 
```
    LBFGS Accumulation History: 1 points
    Constructing line search parameters: GD
    th(0)=223.79596228852782;dx=-2.651869395323579
    New Minimum: 223.79596228852782 > 218.12754964506883
    WOLFE (weak): th(2.154434690031884)=218.12754964506883; dx=-2.6102187707107696 delta=5.668412643458993
    New Minimum: 218.12754964506883 > 212.54887055214257
    WOLFE (weak): th(4.308869380063768)=212.54887055214257; dx=-2.568568146097961 delta=11.247091736385244
    New Minimum: 212.54887055214257 > 191.1314896857056
    WOLFE (weak): th(12.926608140191302)=191.1314896857056; dx=-2.401965647646725 delta=32.66447260282223
    New Minimum: 191.1314896857056 > 112.52051879114062
    END: th(51.70643256076521)=112.52051879114062; dx=-1.652254404616164 delta=111.2754434973872
    Iteration 1 complete. Error: 112.52051879114062 Total: 239521563707063.5000; Orientation: 0.0035; Line Search: 0.0408
    LBFGS Accumulation History: 1 points
    th(0)=112.52051879114062;dx=-1.0712822477886832
    New Minimum: 112.52051879114062 > 35.118428046122695
    END: th(111.398132
```
...[skipping 92456 bytes](etc/69.txt)...
```
    ; dx=-2.0755308708655347E-10 delta=2.3498586908467827E-7
    New Minimum: 1.7666710867050064E-6 > 1.1544225187418601E-6
    END: th(4379.797361733723)=1.1544225187418601E-6; dx=-1.6521823383492714E-10 delta=8.472344370478245E-7
    Iteration 165 complete. Error: 1.1544225187418601E-6 Total: 239528178672754.8800; Orientation: 0.0031; Line Search: 0.0427
    LBFGS Accumulation History: 1 points
    th(0)=1.1544225187418601E-6;dx=-1.2433312007833286E-10
    New Minimum: 1.1544225187418601E-6 > 4.5186488815337545E-7
    END: th(9435.987371429255)=4.5186488815337545E-7; dx=-2.4577132327412584E-11 delta=7.025576305884847E-7
    Iteration 166 complete. Error: 4.5186488815337545E-7 Total: 239528203583570.8000; Orientation: 0.0031; Line Search: 0.0171
    LBFGS Accumulation History: 1 points
    th(0)=4.5186488815337545E-7;dx=-3.774582540239342E-9
    MAX ALPHA: th(0)=4.5186488815337545E-7;th'(0)=-3.774582540239342E-9;
    Iteration 167 failed, aborting. Error: 4.5186488815337545E-7 Total: 239528226589875.7800; Orientation: 0.0032; Line Search: 0.0149
    
```

Returns: 

```
    4.5186488815337545E-7
```



Training Converged

Code from [LearningTester.java:95](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L95) executed in 0.00 seconds: 
```java
    return TestUtil.compare(runs);
```

Returns: 

![Result](etc/test.47.png)



Code from [LearningTester.java:98](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L98) executed in 0.00 seconds: 
```java
    return TestUtil.compareTime(runs);
```

Returns: 

![Result](etc/test.48.png)



### Model Learning
In this test, attempt to train a network to emulate a randomized network given an example input/output. The target state is:

Code from [LearningTester.java:176](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L176) executed in 0.00 seconds: 
```java
    return network_target.state().stream().map(Arrays::toString).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [1.816, 0.204, -1.06, 1.78, 1.704, -1.208, 0.992, 0.744, 0.316, 1.344, 0.564, 0.56, -0.756, -1.784, -0.764, 0.848, -1.856, -0.104, -1.604, -0.24, 1.892, -0.812, 1.956, 0.76, 0.188, -0.892, -1.576, -1.564, 0.9, -0.164, 1.64, 1.408, -1.544, -0.064, -1.876, 1.936, 0.676, -1.904, -1.536, -1.18, -0.676, -0.132, -0.252, 0.096, -0.624, -1.136, 1.096, 1.968, -0.604, 0.96, 1.496, -0.54, 0.228, -0.78, -1.268, -0.06, -1.524, 1.864, 1.012, -1.248, 1.756, 0.74, 1.944, 1.78, -0.556, -0.98, -1.532, -0.488, -0.66, 0.06, 1.28, 0.404, 1.016, -1.98, 0.404, 0.664, 1.904, -1.044, -0.848, -1.576, 0.84, -0.932, -0.576, 0.976, -0.752, 1.324, 0.088, 1.576, -1.004, 1.608, 1.632, -0.416, 1.364, -1.82, -1.02, -0.088, 1.956, -0.584, 1.176, -0.732, -0.604, -0.536, 1.252, -1.54, 1.852, -0.512, 1.96, 1.564, -0.084, -0.364, 1.024, 0.252, 0.964, -1.568, 1.076, -0.172, -0.132, -2.0, 1.808, -0.136, -1.068, 0.032, 1.7, 1.72, 0.628, 1.136, -1.816, -1.18, -0.248, -0.38, 1.152, -1.572, 0.872, 1.36, 1.484, 0.728, 1.12, 1.256, -0.188, -1.128, 1.6, -1
```
...[skipping 238 bytes](etc/70.txt)...
```
     -1.832, 1.24, -0.304, -0.652, 0.916, -0.328, -1.116, -0.924, 1.272, 0.636, -1.236, -1.54, 0.884, 0.572, 1.976, -1.408, 1.576, 1.384, 0.672, 1.784, -1.636, -0.944, 0.74, -1.7, 1.712, 0.212, -1.864, -0.46, -0.268, -0.732, -0.284, 1.788, -1.356, 0.036, -0.196, -1.716, 1.672, -0.752, -1.736, 1.724, -0.144, 1.384, 1.736, -1.064, 1.692, -1.648, -0.244, -0.416, 1.908, 0.564, 0.124, 1.072, 0.452, -0.612, -0.172, -1.412, 1.584, -0.444, -0.404, -0.084, 1.424, 1.484, -0.96, -0.992, 0.78, -0.692, -0.176, 0.764, 0.192, 0.304, -1.692, 1.208, -0.16, 1.416, 0.56, 1.696, 0.812, -0.916, -1.272, -1.784, 1.104, 0.588, 0.928, -0.516, 1.588, 1.24, -1.568, -0.112, -0.148, 1.948, 0.876, -1.064, 1.38, -0.264, 0.828, -0.884, 0.328, 0.44, 0.98, 0.152, -0.904, 0.928, 0.896, 0.884, -1.3, -1.968, 0.54, 1.064, 1.192, 1.452, 0.876, 1.196, 1.024, -1.256, 1.784, 1.68, 1.676, -1.608, 0.828, 0.68, -0.692, 0.992, 0.72, 1.056, -0.392, 0.98, 1.012, -1.56, 1.504, -1.632, 0.7, 1.676, 0.756, -1.284, -1.724, -1.792, -0.14, 1.824, 0.384, 0.388, 1.412]
```



First, we use a conjugate gradient descent method, which converges the fastest for purely linear functions.

Code from [LearningTester.java:225](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L225) executed in 2.11 seconds: 
```java
    return new IterativeTrainer(trainable)
      .setLineSearchFactory(label -> new QuadraticSearch())
      .setOrientation(new GradientDescent())
      .setMonitor(monitor)
      .setTimeout(30, TimeUnit.SECONDS)
      .setMaxIterations(250)
      .setTerminateThreshold(0)
      .run();
```
Logging: 
```
    Constructing line search parameters: GD
    F(0.0) = LineSearchPoint{point=PointSample{avg=209.92709600175485}, derivative=-223.4432514785273}
    New Minimum: 209.92709600175485 > 209.92709597941
    F(1.0E-10) = LineSearchPoint{point=PointSample{avg=209.92709597941}, derivative=-223.4432514665545}, delta = -2.2344835315379896E-8
    New Minimum: 209.92709597941 > 209.92709584534614
    F(7.000000000000001E-10) = LineSearchPoint{point=PointSample{avg=209.92709584534614}, derivative=-223.44325139471812}, delta = -1.5640870287825237E-7
    New Minimum: 209.92709584534614 > 209.92709490688168
    F(4.900000000000001E-9) = LineSearchPoint{point=PointSample{avg=209.92709490688168}, derivative=-223.44325089186313}, delta = -1.094873169904531E-6
    New Minimum: 209.92709490688168 > 209.92708833764928
    F(3.430000000000001E-8) = LineSearchPoint{point=PointSample{avg=209.92708833764928}, derivative=-223.44324737187856}, delta = -7.66410556707342E-6
    New Minimum: 209.92708833764928 > 209.92704235303245
    F(2.4010000000000004E-7) = LineSearchP
```
...[skipping 32112 bytes](etc/71.txt)...
```
    
    Converged to right
    Iteration 25 complete. Error: 1.364482846999469E-33 Total: 239530420054578.6200; Orientation: 0.0001; Line Search: 0.0519
    Zero gradient: 7.767464488192921E-18
    F(0.0) = LineSearchPoint{point=PointSample{avg=1.364482846999469E-33}, derivative=-6.033350457533812E-35}
    F(2.418188020810695) = LineSearchPoint{point=PointSample{avg=1.364482846999469E-33}, derivative=-6.033350457533812E-35}, delta = 0.0
    F(16.927316145674865) = LineSearchPoint{point=PointSample{avg=7.151523306870058E-32}, derivative=4.5965586438888815E-34}, delta = 7.015075022170111E-32
    F(1.3021012419749896) = LineSearchPoint{point=PointSample{avg=1.364482846999469E-33}, derivative=-6.033350457533812E-35}, delta = 0.0
    New Minimum: 1.364482846999469E-33 > 0.0
    F(9.114708693824927) = LineSearchPoint{point=PointSample{avg=0.0}, derivative=0.0}, delta = -1.364482846999469E-33
    0.0 <= 1.364482846999469E-33
    Converged to right
    Iteration 26 complete. Error: 0.0 Total: 239530470239557.6000; Orientation: 0.0001; Line Search: 0.0402
    
```

Returns: 

```
    0.0
```



Training Converged

Next, we run the same optimization using L-BFGS, which is nearly ideal for purely second-order or quadratic functions.

Code from [LearningTester.java:249](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L249) executed in 0.99 seconds: 
```java
    return new IterativeTrainer(trainable)
      .setLineSearchFactory(label -> new ArmijoWolfeSearch())
      .setOrientation(new LBFGS())
      .setMonitor(monitor)
      .setTimeout(30, TimeUnit.SECONDS)
      .setMaxIterations(250)
      .setTerminateThreshold(0)
      .run();
```
Logging: 
```
    LBFGS Accumulation History: 1 points
    Constructing line search parameters: GD
    th(0)=216.15078094533496;dx=-229.85526986529334
    New Minimum: 216.15078094533496 > 6.451593715229602
    WOLF (strong): th(2.154434690031884)=6.451593715229602; dx=35.18778869230636 delta=209.69918723010537
    END: th(1.077217345015942)=39.92394236670203; dx=-97.33374058649345 delta=176.22683857863294
    Iteration 1 complete. Error: 6.451593715229602 Total: 239530530931120.5300; Orientation: 0.0001; Line Search: 0.0300
    LBFGS Accumulation History: 1 points
    th(0)=39.92394236670203;dx=-41.70995136156809
    New Minimum: 39.92394236670203 > 2.1903236924037315
    WOLF (strong): th(2.3207944168063896)=2.1903236924037315; dx=9.192104540630893 delta=37.7336186742983
    END: th(1.1603972084031948)=6.290482136829105; dx=-16.258923410468594 delta=33.633460229872924
    Iteration 2 complete. Error: 2.1903236924037315 Total: 239530566859770.4700; Orientation: 0.0001; Line Search: 0.0273
    LBFGS Accumulation History: 1 points
    th(0)=6.290482136829105;dx=-6.4409
```
...[skipping 12124 bytes](etc/72.txt)...
```
    9E-33 delta=1.289097326744286E-32
    New Minimum: 1.7010552825926713E-32 > 1.5431844939353163E-32
    END: th(1.2682895248506607)=1.5431844939353163E-32; dx=-2.8621274002968812E-34 delta=1.0174388991870399E-31
    Iteration 25 complete. Error: 1.5431844939353163E-32 Total: 239531413869976.6600; Orientation: 0.0002; Line Search: 0.0510
    LBFGS Accumulation History: 1 points
    th(0)=1.5431844939353163E-32;dx=-2.2011010095823283E-34
    New Minimum: 1.5431844939353163E-32 > 1.364482846999469E-33
    END: th(2.7324469493423185)=1.364482846999469E-33; dx=-5.339318390084312E-35 delta=1.4067362092353694E-32
    Iteration 26 complete. Error: 1.364482846999469E-33 Total: 239531437845493.6000; Orientation: 0.0001; Line Search: 0.0171
    LBFGS Accumulation History: 1 points
    th(0)=1.364482846999469E-33;dx=-6.033350457533812E-35
    New Minimum: 1.364482846999469E-33 > 0.0
    END: th(5.886878496334885)=0.0; dx=0.0 delta=1.364482846999469E-33
    Iteration 27 complete. Error: 0.0 Total: 239531464751442.5600; Orientation: 0.0001; Line Search: 0.0167
    
```

Returns: 

```
    0.0
```



Training Converged

Code from [LearningTester.java:95](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L95) executed in 0.00 seconds: 
```java
    return TestUtil.compare(runs);
```

Returns: 

![Result](etc/test.49.png)



Code from [LearningTester.java:98](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L98) executed in 0.00 seconds: 
```java
    return TestUtil.compareTime(runs);
```

Returns: 

![Result](etc/test.50.png)



### Performance
Now we execute larger-scale runs to benchmark performance:

Code from [PerformanceTester.java:66](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/PerformanceTester.java#L66) executed in 1.58 seconds: 
```java
    test(component, inputPrototype);
```
Logging: 
```
    100 batches
    Input Dimensions:
    	[100, 100, 7]
    Performance:
    	Evaluation performance: 0.071064s +- 0.015854s [0.053239s - 0.091374s]
    	Learning performance: 0.167798s +- 0.008603s [0.150948s - 0.175298s]
    
```

