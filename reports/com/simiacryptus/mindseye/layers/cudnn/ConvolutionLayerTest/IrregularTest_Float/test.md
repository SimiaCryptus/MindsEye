# ConvolutionLayer
## IrregularTest_Float
### Batch Execution
Code from [BatchingTester.java:66](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/BatchingTester.java#L66) executed in 0.02 seconds: 
```java
    return test(reference, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=5.6547e-17 +- 2.9383e-16 [0.0000e+00 - 3.5527e-15] (3000#), relativeTol=1.9003e-17 +- 2.8281e-16 [0.0000e+00 - 5.8433e-15] (3000#)}
```



Code from [SingleDerivativeTester.java:77](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/SingleDerivativeTester.java#L77) executed in 0.95 seconds: 
```java
    return test(component, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ 1.852, 1.12, 1.908, -1.476, 1.52, 0.752, -0.824 ], [ -1.664, -1.296, -0.324, 1.152, -1.032, 0.984, -0.304 ], [ 0.064, -0.512, -1.1, -0.3, -0.576, -0.808, 0.492 ], [ 1.104, 0.148, -1.484, -0.148, -0.852, -1.04, 0.672 ], [ -0.648, 1.696, -1.4, 1.764, 1.104, -0.136, -0.176 ] ],
    	[ [ -0.624, -1.392, -1.708, 0.18, -1.708, 1.168, 0.532 ], [ 1.224, 1.04, -0.888, -0.108, -0.516, 0.632, -0.48 ], [ -1.936, -1.236, -0.428, -0.196, -1.148, -1.996, 1.384 ], [ 1.868, -0.676, 0.732, 0.224, -1.956, 1.528, 0.772 ], [ -1.052, 1.932, 0.4, -1.224, 0.232, -0.164, 1.604 ] ],
    	[ [ 1.472, 0.716, 1.968, -1.34, -1.116, 0.348, 1.532 ], [ 0.14, 1.644, -1.672, 0.556, 0.08, -1.104, 0.452 ], [ -0.048, 0.084, 0.272, -1.796, 0.048, 0.308, 0.132 ], [ 1.48, -0.456, -1.124, 0.732, 1.38, -0.316, 1.7 ], [ -1.088, -0.68, -1.824, 1.088, -1.328, -1.456, -1.792 ] ],
    	[ [ 1.18, -1.272, -1.192, 0.444, 0.54, 0.288, -0.392 ], [ 1.012, -1.888, 0.732, -1.46, 1.124, -1.764, 0.808 ], [ -1.54, -0.056, -1.412, -1.076, -1.58, -1.1, -0.268 ], [ -0
```
...[skipping 10889 bytes](etc/73.txt)...
```
     -4.205968906489943E-12, ... ], [ -2.255751141433393E-12, -9.06175134929299E-12, 6.801226248853709E-12, -3.815836535636663E-12, 5.6710192097853E-12, 5.441203043687892E-12, -1.36528566230254E-11, 1.805333660342967E-12, ... ], [ 0.0, -1.1137535338434645E-11, 8.701817044709514E-12, -2.0805579481475434E-12, -1.2697620732637915E-11, 0.0, -1.2322365350314612E-11, -1.36528566230254E-11, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 8.701817044709514E-12, -1.0962342145148796E-11, -1.2697620732637915E-11, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 6.626033055567859E-12, -9.06175134929299E-12, 6.801226248853709E-12, ... ], ... ]
    Error Statistics: {meanExponent=-11.529690313274278, negative=2830, min=-1.1604051053382136E-12, max=-1.1604051053382136E-12, mean=3.445341102362033E-14, count=39375.0, positive=3085, stdDev=2.7829083880387512E-12, zeros=33460}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 9.9305e-13 +- 3.0277e-12 [0.0000e+00 - 3.4527e-11] (61250#)
    relativeTol: 7.9878e-12 +- 4.3213e-11 [6.2513e-17 - 1.4022e-09] (11830#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=9.9305e-13 +- 3.0277e-12 [0.0000e+00 - 3.4527e-11] (61250#), relativeTol=7.9878e-12 +- 4.3213e-11 [6.2513e-17 - 1.4022e-09] (11830#)}
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
      "id": "499d5ea0-bbf2-40f7-84d7-602c6f80363f",
      "isFrozen": false,
      "name": "ConvolutionLayer/499d5ea0-bbf2-40f7-84d7-602c6f80363f",
      "filter": [
        [
          [
            -1.336,
            -0.648,
            -0.448
          ],
          [
            1.56,
            -1.608,
            0.736
          ],
          [
            0.384,
            1.4,
            -1.588
          ]
        ],
        [
          [
            -1.552,
            1.94,
            -0.192
          ],
          [
            1.448,
            -0.1,
            -0.956
          ],
          [
            -0.392,
            0.004,
            -1.884
          ]
        ],
        [
          [
            -1.284,
            -1.452,
            0.448
          ],
          [
            1.156,
            -0.368,
            -1.384
          ],
          [
            -0.236,
            1.084,
            -0.268
          ]
        ],
        [
          [
            0.304,
            -1.4,
            -0.628
          ],
          [
            -1.148,
            -1.592,
            1.736
          ],
          [
            1.688,
            1.288,
            -0.52
          ]
        ],
        [
     
```
...[skipping 5164 bytes](etc/74.txt)...
```
       0.928,
            -0.704
          ],
          [
            -1.408,
            -1.428,
            -0.436
          ],
          [
            0.668,
            -1.62,
            -0.828
          ]
        ],
        [
          [
            1.976,
            -0.16,
            -0.352
          ],
          [
            1.388,
            -0.308,
            0.428
          ],
          [
            -0.168,
            0.084,
            -1.972
          ]
        ],
        [
          [
            -1.776,
            0.184,
            0.728
          ],
          [
            1.692,
            -0.188,
            -0.844
          ],
          [
            -1.96,
            1.464,
            1.608
          ]
        ],
        [
          [
            -0.384,
            0.008,
            0.616
          ],
          [
            0.46,
            -0.348,
            0.188
          ],
          [
            1.368,
            -1.896,
            0.732
          ]
        ],
        [
          [
            -0.804,
            -0.628,
            -0.58
          ],
          [
            0.96,
            -0.964,
            -1.564
          ],
          [
            -1.148,
            -0.924,
            -0.804
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
    	[ [ 0.324, -0.676, 0.644, -0.032, 0.648, -1.256, 0.196 ], [ 1.596, 1.532, -0.236, 0.508, -1.512, 1.46, -0.516 ], [ -0.34, -0.172, -0.964, 0.932, 1.872, -1.704, 0.848 ], [ -0.072, -0.436, 1.4, 1.932, 1.828, 1.836, 1.072 ], [ 1.6, -0.688, -0.836, -0.636, 1.988, 0.236, 0.36 ] ],
    	[ [ -1.736, 1.06, 0.404, 1.1, 0.188, 0.804, -1.512 ], [ -1.212, -0.12, 1.228, -1.3, 0.476, 1.152, 0.148 ], [ 1.552, -0.764, 1.944, 0.168, -0.444, 0.812, 0.216 ], [ -1.188, -1.252, 0.692, -0.592, -1.608, 1.876, 1.696 ], [ -0.696, 1.576, -0.868, 0.54, -0.468, 0.504, -0.296 ] ],
    	[ [ 1.292, 1.996, -1.608, -0.192, 0.692, -1.26, -1.16 ], [ 1.6, 1.928, -0.78, -1.292, 1.336, -0.46, 0.416 ], [ -0.016, 0.384, 0.364, 0.508, 1.716, -0.272, 1.148 ], [ -1.068, 0.56, 1.244, 0.688, 1.332, -0.772, -0.564 ], [ 0.784, -1.74, 0.492, -0.776, -0.26, -0.184, 1.456 ] ],
    	[ [ 0.14, -0.76, 0.044, -0.756, 0.488, -1.912, -1.808 ], [ 1.08, 1.444, -1.544, -0.064, 1.812, -1.36, -1.996 ], [ -1.256, -1.456, -0.94, 0.112, 1.464, 0.912, -0.192 ], [ -0.76, -1.
```
...[skipping 270 bytes](etc/75.txt)...
```
    8, -1.816, 1.524, 0.256, -1.264, -0.316, -0.12 ], [ 0.808, -0.384, -1.952, 1.972, 0.348, 0.98, 1.112 ] ]
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
      "id": "fb3746d9-6f7e-4fda-93fb-798ca8296692",
      "isFrozen": false,
      "name": "ConvolutionLayer/fb3746d9-6f7e-4fda-93fb-798ca8296692",
      "filter": [
        [
          [
            -1.336,
            -0.648,
            -0.448
          ],
          [
            1.56,
            -1.608,
            0.736
          ],
          [
            0.384,
            1.4,
            -1.588
          ]
        ],
        [
          [
            -1.552,
            1.94,
            -0.192
          ],
          [
            1.448,
            -0.1,
            -0.956
          ],
          [
            -0.392,
            0.004,
            -1.884
          ]
        ],
        [
          [
            -1.284,
            -1.452,
            0.448
          ],
          [
            1.156,
            -0.368,
            -1.384
          ],
          [
            -0.236,
            1.084,
            -0.268
          ]
        ],
        [
          [
            0.304,
            -1.4,
            -0.628
          ],
          [
            -1.148,
            -1.592,
            1.736
          ],
          [
            1.688,
            1.288,
            -0.52
          ]
        ],
        [
       
```
...[skipping 5135 bytes](etc/76.txt)...
```
          [
            0.52,
            0.928,
            -0.704
          ],
          [
            -1.408,
            -1.428,
            -0.436
          ],
          [
            0.668,
            -1.62,
            -0.828
          ]
        ],
        [
          [
            1.976,
            -0.16,
            -0.352
          ],
          [
            1.388,
            -0.308,
            0.428
          ],
          [
            -0.168,
            0.084,
            -1.972
          ]
        ],
        [
          [
            -1.776,
            0.184,
            0.728
          ],
          [
            1.692,
            -0.188,
            -0.844
          ],
          [
            -1.96,
            1.464,
            1.608
          ]
        ],
        [
          [
            -0.384,
            0.008,
            0.616
          ],
          [
            0.46,
            -0.348,
            0.188
          ],
          [
            1.368,
            -1.896,
            0.732
          ]
        ],
        [
          [
            -0.804,
            -0.628,
            -0.58
          ],
          [
            0.96,
            -0.964,
            -1.564
          ],
          [
            -1.148,
            -0.924,
            -0.804
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
    	[ [ -0.188, 1.868, 0.72, 0.708, -1.216, -1.972, 0.52 ], [ 0.84, 1.44, 1.136, 1.468, -0.04, -0.828, 0.476 ], [ -0.164, -0.612, -1.336, 0.224, 1.376, -0.884, -1.868 ], [ 0.02, 0.484, -1.872, -1.944, 1.464, -1.476, -1.564 ], [ 0.4, 0.052, -1.304, -1.704, 0.048, 1.944, 1.176 ] ],
    	[ [ 0.028, 0.088, -1.516, -1.476, -1.432, 0.156, 1.576 ], [ 0.896, 1.688, -0.56, 1.444, 1.172, -1.128, -1.728 ], [ -0.14, -0.076, 0.972, -1.876, 0.284, 0.328, -1.724 ], [ 1.824, 1.316, 0.06, 0.668, -1.04, -1.476, -0.444 ], [ 0.356, 0.308, -0.588, 1.524, -1.468, 1.78, 1.232 ] ],
    	[ [ -1.46, -0.388, -0.224, -0.752, 0.392, 0.592, -1.168 ], [ 0.32, 0.24, 0.824, -1.528, 1.288, 1.212, 1.076 ], [ -0.604, 0.308, 0.532, 1.968, -0.272, 0.452, -0.336 ], [ -0.328, 0.344, 0.888, -0.584, -1.688, -1.076, -0.944 ], [ -0.58, -1.876, 0.196, -1.232, 0.152, 0.016, -1.908 ] ],
    	[ [ 1.372, -0.056, -0.564, -0.512, -1.728, -1.244, 1.632 ], [ 0.432, -1.192, 0.044, -0.26, -1.12, 1.94, 0.888 ], [ 0.792, 1.904, 1.4, -1.216, 1.52, -
```
...[skipping 4647 bytes](etc/77.txt)...
```
    9999999998, -11.240000000000002, -4.112, -8.195999999999996 ], [ -6.568, 2.5199999999999987, -6.828, -1.8079999999999994, -11.239999999999998, -4.111999999999999, -8.195999999999998 ], [ -6.568, 2.5199999999999987, -6.828, -1.8079999999999994, -11.240000000000002, -4.112, -8.195999999999998 ], [ -10.856, -1.5000000000000004, -6.796, 1.56, -7.659999999999999, -5.56, -2.7999999999999994 ] ],
    	[ [ 7.8039999999999985, 9.579999999999998, 4.764, -6.54, -7.708000000000001, -1.272, -4.275999999999998 ], [ -0.07600000000000184, 5.223999999999998, -1.4040000000000004, -5.936, -13.588, -1.1719999999999997, -4.411999999999999 ], [ -0.07600000000000229, 5.223999999999999, -1.4040000000000004, -5.936, -13.588000000000001, -1.1719999999999997, -4.411999999999999 ], [ -0.07600000000000273, 5.223999999999999, -1.4040000000000006, -5.936, -13.588000000000001, -1.1719999999999997, -4.411999999999999 ], [ -8.236, 1.8279999999999996, -1.4840000000000004, -2.6600000000000006, -11.196, -1.1399999999999997, -0.2799999999999998 ] ]
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
    	[ [ 0.876, -0.992, -0.052, 1.552, -0.56, -1.632, -1.484 ], [ 1.696, -1.44, 1.724, 1.896, -0.444, 0.94, 0.808 ], [ -1.296, -1.116, 1.568, 1.352, 1.004, 0.652, -1.088 ], [ -0.82, 1.184, -0.792, -1.616, -0.012, 1.088, -0.64 ], [ 0.328, -0.94, 1.184, -0.924, 0.16, 1.452, 1.856 ], [ 1.96, 1.376, 1.328, 0.648, 0.56, -0.344, 0.616 ], [ -0.372, 1.188, -1.448, 1.8, 0.536, 1.228, -0.2 ], [ -1.528, -1.976, -0.908, 1.2, 1.9, 1.988, 0.224 ], ... ],
    	[ [ 1.324, -1.32, -0.536, 1.46, -0.352, 0.836, -0.244 ], [ -1.844, 1.232, 0.368, -1.644, 0.672, 0.644, 0.516 ], [ -0.492, -0.42, 0.796, -1.412, -0.828, -0.912, 0.416 ], [ -1.548, -0.136, 1.996, 1.0, 1.336, -0.74, -0.464 ], [ 1.832, -1.172, 0.264, 1.08, 1.388, -0.304, 0.896 ], [ -1.368, 1.676, -1.976, 1.712, -1.656, 1.884, -0.8 ], [ 0.492, -0.708, -1.924, -1.796, -0.488, -0.804, 1.596 ], [ -0.116, -1.288, 1.58, -0.044, -0.844, -0.304, -0.424 ], ... ],
    	[ [ 0.316, 1.212, 0.28, -1.284, 0.736, -1.872, 1.468 ], [ -0.56, -1.064, 0.352, 0.456, -0.224, -0.912, 1.156 ], [ 0.844, -0.
```
...[skipping 1553 bytes](etc/78.txt)...
```
     ], [ 0.728, -1.212, -1.332, -1.428, -0.636, -1.18, 0.8 ], [ -0.388, -1.5, -1.248, -1.18, 1.236, 0.32, -1.824 ], ... ],
    	[ [ -0.856, -0.372, -1.92, 0.912, -1.352, -0.74, -1.444 ], [ -1.68, -1.868, 0.66, -0.528, -1.432, -1.44, 1.556 ], [ -0.832, -1.588, -1.404, -0.712, 1.172, -1.4, 1.148 ], [ 1.376, -0.344, 1.516, 0.496, -1.932, -0.784, 1.388 ], [ 0.292, -1.748, 1.624, -1.692, -1.004, 0.948, -1.636 ], [ 0.732, -1.42, -0.768, 1.328, -0.228, -1.308, 0.448 ], [ 0.1, -0.856, 0.312, 0.34, 0.856, -0.244, 0.04 ], [ 1.78, 1.176, -1.3, 0.04, -1.196, -1.088, 0.956 ], ... ],
    	[ [ -1.36, -0.644, -0.556, -1.068, -0.104, 0.848, 1.416 ], [ 1.868, -0.652, -0.296, 1.084, 1.936, 0.104, -1.1 ], [ -0.784, -0.944, 1.996, -0.828, 1.048, 0.604, 0.592 ], [ 0.876, -0.284, -0.424, -1.032, 0.064, 0.612, 0.568 ], [ 1.072, -0.136, -0.308, 1.62, -2.0, -1.596, 1.288 ], [ 0.912, -1.932, 1.24, 1.5, 0.332, -0.3, 0.796 ], [ 1.184, -0.168, -0.6, 0.124, -1.168, -0.968, 0.092 ], [ 0.076, 0.452, 0.156, 0.568, -0.516, -0.932, -1.888 ], ... ],
    	...
    ]
```



First, we use a conjugate gradient descent method, which converges the fastest for purely linear functions.

Code from [LearningTester.java:225](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L225) executed in 14.53 seconds: 
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
    F(0.0) = LineSearchPoint{point=PointSample{avg=219.97240225206505}, derivative=-2.597802595571837}
    New Minimum: 219.97240225206505 > 219.9724022518049
    F(1.0E-10) = LineSearchPoint{point=PointSample{avg=219.9724022518049}, derivative=-2.597802595569887}, delta = -2.601439064164879E-10
    New Minimum: 219.9724022518049 > 219.97240225024493
    F(7.000000000000001E-10) = LineSearchPoint{point=PointSample{avg=219.97240225024493}, derivative=-2.597802595558185}, delta = -1.8201262719230726E-9
    New Minimum: 219.97240225024493 > 219.9724022393362
    F(4.900000000000001E-9) = LineSearchPoint{point=PointSample{avg=219.9724022393362}, derivative=-2.5978025954762733}, delta = -1.2728861520372448E-8
    New Minimum: 219.9724022393362 > 219.97240216295623
    F(3.430000000000001E-8) = LineSearchPoint{point=PointSample{avg=219.97240216295623}, derivative=-2.597802594902891}, delta = -8.9108823431161E-8
    New Minimum: 219.97240216295623 > 219.97240162833222
    F(2.4010000000000004E-7) = LineSearch
```
...[skipping 287449 bytes](etc/79.txt)...
```
    4}, derivative=-1.5547574821806792E-8}, delta = -1.963958566227297E-6
    F(659.5807257002338) = LineSearchPoint{point=PointSample{avg=1.8781138274730005E-4}, derivative=4.799878080037202E-8}, delta = 7.20926571485598E-6
    F(50.73697890001798) = LineSearchPoint{point=PointSample{avg=1.7942059510815466E-4}, derivative=-2.0435756023513026E-8}, delta = -1.1815219242894057E-6
    New Minimum: 1.7863815846621677E-4 > 1.7840774645142896E-4
    F(355.1588523001259) = LineSearchPoint{point=PointSample{avg=1.7840774645142896E-4}, derivative=1.3781512388429119E-8}, delta = -2.1943705810151064E-6
    1.7840774645142896E-4 <= 1.8060211703244407E-4
    New Minimum: 1.7840774645142896E-4 > 1.7756286790643878E-4
    F(232.5484273806396) = LineSearchPoint{point=PointSample{avg=1.7756286790643878E-4}, derivative=2.0515270686826332E-22}, delta = -3.039249126005289E-6
    Right bracket at 232.5484273806396
    Converged to right
    Iteration 250 complete. Error: 1.7756286790643878E-4 Total: 239548715413003.3400; Orientation: 0.0019; Line Search: 0.0486
    
```

Returns: 

```
    1.7756286790643878E-4
```



This training run resulted in the following regressed input:

Code from [LearningTester.java:144](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L144) executed in 0.02 seconds: 
```java
    return Arrays.stream(input_gd).map(x -> x.prettyPrint()).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [
    	[ [ 0.29230933387979957, -0.9980916686047948, -0.37179694096923527, 1.5358643965320815, 0.25621257019936694, -2.6096558396853915, -1.1412818692153675 ], [ 0.5119557412594854, -1.0948618529001435, 1.5308790466167688, 2.599723149296946, 0.14831002914930508, 1.6715525892840994, 0.8780637644324851 ], [ 0.7642728283872058, -0.14977675899407747, 0.44346193854765487, 0.8368361414201881, 0.924835343263738, -0.28395209312306097, -1.9539236262201127 ], [ 0.35824745201541314, 1.5598265896150638, -1.202499366267062, -2.545543911671109, -0.46737269298777573, 0.2508292906546989, -1.6922781616763503 ], [ 0.14600877272517107, -1.1057029990289053, 1.8721992773158047, 0.8153881137024457, -0.7470271875503914, 1.185457133107792, 0.9828714836033174 ], [ 0.4613778496270205, 2.6210453716312303, 1.2023287038176582, 0.16190609948600115, 1.8236343523375793, -0.4524451141355901, 1.1553926446627265 ], [ 0.26532313146929126, 0.34050889967475706, -1.2647424205901787, 1.9619414860906637, -0.01053986180777157, 1.6026001601389204, -0.0241
```
...[skipping 7522 bytes](etc/80.txt)...
```
    3179, -0.11813104203052352, -0.14913075402542683, 1.7933524319072298, 1.1993247078889173, 0.6980669127923396, -1.0976707802066716 ], [ -0.9712993181360999, -1.0177063062283722, 2.467010217421952, -2.2787568755860774, 0.41419167815248364, -1.3148853508720126, 0.1310625928397958 ], [ 0.5452770929814401, -1.547282247438953, -0.8394500694660026, -0.7921103497137868, 0.3196148052399717, -0.3483373771213436, -0.7739823135664469 ], [ 0.47724410895619496, 0.40534313690905033, -0.5295337894332094, 2.8995127666809504, -1.587850165342646, -0.8675293376869003, 1.0007610721600637 ], [ 1.231982776459985, -0.36025520591386795, 0.379703130789351, 1.4480584466570017, -0.454057574913518, 0.6712103455025373, 1.409011183314305 ], [ 1.7510347095682526, -0.6354191547484762, -0.833994852556816, 0.8424691393793073, -1.1065328317681369, -0.7665744968919675, 0.8225827633440252 ], [ 0.042190982409908845, 0.21446060502757, 0.969992499033135, 0.447058389647556, 0.6751502526539048, 0.06787783502353315, -0.8288627006170349 ], ... ],
    	...
    ]
```



Next, we run the same optimization using L-BFGS, which is nearly ideal for purely second-order or quadratic functions.

Code from [LearningTester.java:249](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L249) executed in 10.64 seconds: 
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
    th(0)=219.97240225206505;dx=-2.597802595571837
    New Minimum: 219.97240225206505 > 214.42086821047664
    WOLFE (weak): th(2.154434690031884)=214.42086821047664; dx=-2.55578508779842 delta=5.551534041588411
    New Minimum: 214.42086821047664 > 208.95985814522155
    WOLFE (weak): th(4.308869380063768)=208.95985814522155; dx=-2.5137675800250023 delta=11.012544106843507
    New Minimum: 208.95985814522155 > 188.0210576475764
    WOLFE (weak): th(12.926608140191302)=188.0210576475764; dx=-2.3456975489313323 delta=31.95134460448864
    New Minimum: 188.0210576475764 > 111.72020272263994
    END: th(51.70643256076521)=111.72020272263994; dx=-1.5893824090098179 delta=108.25219952942511
    Iteration 1 complete. Error: 111.72020272263994 Total: 239548811701786.2500; Orientation: 0.0039; Line Search: 0.0450
    LBFGS Accumulation History: 1 points
    th(0)=111.72020272263994;dx=-1.0347606852109366
    New Minimum: 111.72020272263994 > 35.97984863701783
    END: th(111.39813
```
...[skipping 143822 bytes](etc/81.txt)...
```
    99.7500; Orientation: 0.0032; Line Search: 0.0175
    LBFGS Accumulation History: 1 points
    th(0)=1.3637426809215114E-7;dx=-1.8305697249370298E-11
    Armijo: th(595.4908993801)=1.4940753250524825E-7; dx=6.207887472920761E-11 delta=-1.3033264413097108E-8
    Armijo: th(297.74544969005)=1.3690736516574113E-7; dx=2.1886588739910857E-11 delta=-5.330970735899876E-10
    New Minimum: 1.3637426809215114E-7 > 1.352222926426976E-7
    END: th(99.24848323001667)=1.352222926426976E-7; dx=-4.908268586281085E-12 delta=1.1519754494535469E-9
    Iteration 249 complete. Error: 1.352222926426976E-7 Total: 239559346261674.7200; Orientation: 0.0032; Line Search: 0.0666
    LBFGS Accumulation History: 1 points
    th(0)=1.352222926426976E-7;dx=-1.1016473333730706E-11
    New Minimum: 1.352222926426976E-7 > 1.3352329385311936E-7
    END: th(213.8243752037956)=1.3352329385311936E-7; dx=-4.875061838104899E-12 delta=1.6989987895782394E-9
    Iteration 250 complete. Error: 1.3352329385311936E-7 Total: 239559383107100.6600; Orientation: 0.0039; Line Search: 0.0239
    
```

Returns: 

```
    1.3352329385311936E-7
```



Training Converged

Code from [LearningTester.java:95](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L95) executed in 0.00 seconds: 
```java
    return TestUtil.compare(runs);
```

Returns: 

![Result](etc/test.51.png)



Code from [LearningTester.java:98](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L98) executed in 0.00 seconds: 
```java
    return TestUtil.compareTime(runs);
```

Returns: 

![Result](etc/test.52.png)



### Model Learning
In this test, attempt to train a network to emulate a randomized network given an example input/output. The target state is:

Code from [LearningTester.java:176](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L176) executed in 0.00 seconds: 
```java
    return network_target.state().stream().map(Arrays::toString).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [-0.264, -0.636, -1.404, -1.056, 1.432, -0.844, -0.1, 0.164, 0.46, -1.512, 1.884, 1.536, 1.688, 1.036, -0.16, -1.38, 1.616, -0.236, 1.608, 1.28, 1.084, 1.472, -0.192, -1.28, -1.652, 1.596, 0.248, -1.504, 1.888, -1.904, -0.104, -1.476, -1.016, -1.92, -1.396, -1.192, -1.14, -1.984, 1.788, 1.264, -1.184, 0.668, -1.964, -1.564, -1.416, -0.34, -1.592, -0.46, -0.36, 1.112, -0.824, 0.776, 0.48, -1.776, 0.192, -1.564, -0.872, 0.124, 0.676, -0.628, -1.08, 1.528, 0.384, -1.452, -0.54, -1.164, 1.56, 0.52, 1.396, -0.156, -0.78, -0.28, -1.8, 1.7, -0.648, -0.084, -0.052, -1.784, -0.34, 1.032, 1.38, -1.876, -0.9, 0.728, 0.184, 0.28, 0.916, 1.524, -0.188, 0.748, 0.604, -1.4, -0.448, 0.18, -1.848, -1.484, -0.14, 1.044, -0.236, -1.824, -1.0, -1.092, -1.944, -1.408, -1.884, -0.292, 1.824, -0.688, 0.748, 0.904, 1.296, 1.836, 0.616, 1.624, 0.872, -1.568, 1.74, 0.76, -1.904, 0.3, -0.372, -0.428, 1.868, -0.576, -0.344, -1.756, -1.4, -1.616, 0.128, -0.348, -1.828, 1.36, 0.732, -1.38, 1.332, 1.204, -0.7, 0.472, -1.148, 0.604, 1.632, 
```
...[skipping 257 bytes](etc/82.txt)...
```
    8, -1.728, 0.268, -1.432, 1.692, -0.6, -0.924, 0.096, 1.372, 1.544, -1.372, 1.2, -1.164, -1.588, -1.952, 0.624, -0.752, -0.348, 1.304, 0.388, 1.94, -0.52, -0.144, 1.4, 1.392, -0.452, -0.352, 1.388, 1.704, -0.624, 0.056, 1.696, 0.44, -1.388, 0.188, 1.704, -0.24, -0.516, 1.06, -0.436, 0.376, 0.928, 1.436, -1.532, -0.384, 0.504, 1.544, 0.736, -1.148, 0.48, -0.704, 1.872, -1.62, -0.168, 0.004, 1.836, 0.304, -0.752, 1.748, 1.976, -0.084, -0.252, -0.512, 0.048, -1.02, 1.056, -1.608, 1.528, -1.804, 0.304, 0.456, 0.188, 0.016, -1.604, 0.52, 0.464, 0.208, 1.464, -0.58, 1.012, -0.164, -0.964, -1.072, -1.792, -0.392, -0.232, -1.392, -0.928, -1.06, -0.804, 0.448, -0.384, 1.792, -1.896, 1.156, -1.516, -1.384, 1.736, -0.392, -0.28, 0.908, -0.828, 1.288, -0.108, -1.972, -1.376, -1.888, 1.904, 1.592, -1.292, -1.228, 1.368, 0.888, -0.476, 0.428, -1.124, 0.14, -1.284, -1.336, -0.784, 1.948, -1.036, 0.38, -0.268, -0.308, -0.652, 0.728, -1.776, -0.704, -0.888, -1.288, -0.66, 1.448, -0.868, -1.132, -1.552, -0.888, -0.628, -1.716]
```



First, we use a conjugate gradient descent method, which converges the fastest for purely linear functions.

Code from [LearningTester.java:225](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L225) executed in 1.87 seconds: 
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
    F(0.0) = LineSearchPoint{point=PointSample{avg=219.36267631193775}, derivative=-229.26631406745804}
    New Minimum: 219.36267631193775 > 219.36267628901373
    F(1.0E-10) = LineSearchPoint{point=PointSample{avg=219.36267628901373}, derivative=-229.26631405540013}, delta = -2.292401291015267E-8
    New Minimum: 219.36267628901373 > 219.36267615145317
    F(7.000000000000001E-10) = LineSearchPoint{point=PointSample{avg=219.36267615145317}, derivative=-229.26631398305267}, delta = -1.6048457496253832E-7
    New Minimum: 219.36267615145317 > 219.3626751885341
    F(4.900000000000001E-9) = LineSearchPoint{point=PointSample{avg=219.3626751885341}, derivative=-229.26631347662007}, delta = -1.1234036492169253E-6
    New Minimum: 219.3626751885341 > 219.36266844810433
    F(3.430000000000001E-8) = LineSearchPoint{point=PointSample{avg=219.36266844810433}, derivative=-229.26630993159162}, delta = -7.863833417331989E-6
    New Minimum: 219.36266844810433 > 219.36262126510113
    F(2.4010000000000004E-7) = Li
```
...[skipping 29578 bytes](etc/83.txt)...
```
    =7.990174893757324E-33}, derivative=8.438455407565847E-36}, delta = 7.449805173680931E-33
    F(1.2523690925383115) = LineSearchPoint{point=PointSample{avg=1.6566079009641249E-34}, derivative=-1.444977553018327E-36}, delta = -3.7470892997998056E-34
    F(8.766583647768181) = LineSearchPoint{point=PointSample{avg=2.9375207958167428E-33}, derivative=3.6144968570795995E-36}, delta = 2.3971510757403498E-33
    F(0.6743525882898601) = LineSearchPoint{point=PointSample{avg=5.4036972007639305E-34}, derivative=-4.0443506327981553E-36}, delta = 0.0
    F(4.720468118029021) = LineSearchPoint{point=PointSample{avg=1.7709927322211715E-33}, derivative=1.566297569726647E-36}, delta = 1.2306230121447786E-33
    Loops = 12
    New Minimum: 1.6566079009641249E-34 > 0.0
    F(3.402677824580607) = LineSearchPoint{point=PointSample{avg=0.0}, derivative=0.0}, delta = -5.4036972007639305E-34
    Right bracket at 3.402677824580607
    Converged to right
    Iteration 23 complete. Error: 0.0 Total: 239561398496216.7500; Orientation: 0.0001; Line Search: 0.1351
    
```

Returns: 

```
    0.0
```



Training Converged

Next, we run the same optimization using L-BFGS, which is nearly ideal for purely second-order or quadratic functions.

Code from [LearningTester.java:249](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L249) executed in 0.96 seconds: 
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
    th(0)=220.884982899344;dx=-230.95880754979717
    New Minimum: 220.884982899344 > 5.551435709252432
    WOLF (strong): th(2.154434690031884)=5.551435709252432; dx=31.06084992184489 delta=215.33354719009156
    END: th(1.077217345015942)=42.65517936340119; dx=-99.94897881397615 delta=178.22980353594284
    Iteration 1 complete. Error: 5.551435709252432 Total: 239561461188327.6000; Orientation: 0.0001; Line Search: 0.0258
    LBFGS Accumulation History: 1 points
    th(0)=42.65517936340119;dx=-43.78263473054004
    New Minimum: 42.65517936340119 > 1.987508636443264
    WOLF (strong): th(2.3207944168063896)=1.987508636443264; dx=8.736298499759082 delta=40.66767072695792
    END: th(1.1603972084031948)=7.085638122734028; dx=-17.523168115390487 delta=35.569541240667164
    Iteration 2 complete. Error: 1.987508636443264 Total: 239561499040582.5300; Orientation: 0.0001; Line Search: 0.0271
    LBFGS Accumulation History: 1 points
    th(0)=7.085638122734028;dx=-7.1258094844
```
...[skipping 12401 bytes](etc/84.txt)...
```
    212006739)=2.3469267908835008E-32; dx=-5.632102248983061E-34 delta=2.8895107519398708E-31
    Iteration 26 complete. Error: 2.3469267908835008E-32 Total: 239562301049850.7200; Orientation: 0.0001; Line Search: 0.0231
    LBFGS Accumulation History: 1 points
    th(0)=2.3469267908835008E-32;dx=-9.269026621169244E-34
    Armijo: th(4.415158872251164)=2.3773309454966717E-32; dx=1.269065285332311E-34 delta=-3.0404154613170903E-34
    New Minimum: 2.3469267908835008E-32 > 1.6566079009641249E-34
    END: th(2.207579436125582)=1.6566079009641249E-34; dx=-1.444977553018327E-36 delta=2.3303607118738597E-32
    Iteration 27 complete. Error: 1.6566079009641249E-34 Total: 239562333575573.7000; Orientation: 0.0001; Line Search: 0.0229
    LBFGS Accumulation History: 1 points
    th(0)=1.6566079009641249E-34;dx=-1.2072216010661587E-36
    New Minimum: 1.6566079009641249E-34 > 0.0
    END: th(4.756085718189979)=0.0; dx=0.0 delta=1.6566079009641249E-34
    Iteration 28 complete. Error: 0.0 Total: 239562359763093.6600; Orientation: 0.0001; Line Search: 0.0164
    
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

![Result](etc/test.53.png)



Code from [LearningTester.java:98](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L98) executed in 0.00 seconds: 
```java
    return TestUtil.compareTime(runs);
```

Returns: 

![Result](etc/test.54.png)



### Performance
Now we execute larger-scale runs to benchmark performance:

Code from [PerformanceTester.java:66](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/PerformanceTester.java#L66) executed in 1.56 seconds: 
```java
    test(component, inputPrototype);
```
Logging: 
```
    100 batches
    Input Dimensions:
    	[100, 100, 7]
    Performance:
    	Evaluation performance: 0.081407s +- 0.026268s [0.056769s - 0.129297s]
    	Learning performance: 0.164594s +- 0.023535s [0.137887s - 0.194706s]
    
```

