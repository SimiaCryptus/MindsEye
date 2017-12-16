# ConvolutionLayer
## AsymmetricTest
### Batch Execution
Code from [BatchingTester.java:66](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/BatchingTester.java#L66) executed in 0.05 seconds: 
```java
    return test(reference, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=5.1613e-17 +- 2.8687e-16 [0.0000e+00 - 1.7764e-15] (2250#), relativeTol=4.2117e-18 +- 2.3474e-17 [0.0000e+00 - 2.1769e-16] (2250#)}
```



Code from [SingleDerivativeTester.java:77](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/SingleDerivativeTester.java#L77) executed in 1.38 seconds: 
```java
    return test(component, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ -1.492, 0.588, -1.676 ], [ -0.76, 0.084, -0.756 ], [ -1.336, 0.76, 1.184 ], [ -1.516, -0.224, 0.72 ], [ -0.456, 0.516, -0.28 ] ],
    	[ [ -1.12, 1.24, 0.02 ], [ 0.844, -1.772, 0.488 ], [ 0.132, -0.872, 0.78 ], [ 0.22, 1.228, 1.476 ], [ -0.984, -1.072, -0.076 ] ],
    	[ [ 0.508, 1.788, -0.268 ], [ -1.928, -0.764, -0.06 ], [ 1.3, 0.576, 0.004 ], [ 1.096, 1.356, -0.864 ], [ -1.352, 1.372, 0.704 ] ],
    	[ [ -0.192, -1.024, 0.572 ], [ 0.932, -1.328, -0.012 ], [ 0.624, -0.636, -1.488 ], [ -1.24, -0.404, -0.464 ], [ -0.04, 1.512, -0.584 ] ],
    	[ [ -1.456, 0.308, -0.028 ], [ 1.196, -0.38, -1.248 ], [ 1.128, -1.764, 0.576 ], [ -1.172, 1.472, 0.948 ], [ -1.148, 0.788, -0.704 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.24300828903018962, negative=40, min=-0.704, max=-0.704, mean=-0.07866666666666668, count=75.0, positive=35, stdDev=0.9917292820567963, zeros=0}
    Output: [
    	[ [ -1.7959039999999995, -5.122304, 6.577184, -1.4508, -2.284720000000001, 5.109888 ], [ 3.7232160000000003, -3.783183999999999, 0.40009600000000
```
...[skipping 9929 bytes](etc/41.txt)...
```
    2.3753221611855224E-12, ... ], [ -7.149836278586008E-13, 3.3208991112587682E-12, -5.376143974444858E-12, 3.36070060669158E-12, -6.341149827449044E-12, 3.5016434196677437E-13, -4.5959902550407605E-12, 2.290612144406623E-12, ... ], [ 0.0, -7.149836278586008E-13, 3.3208991112587682E-12, 3.5056402225563943E-12, -5.521083590309672E-12, 0.0, 3.5016434196677437E-13, 4.285793941960492E-12, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, -1.119992987241858E-12, 3.5056402225563943E-12, 3.36070060669158E-12, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, -7.149836278586008E-13, 3.3208991112587682E-12, 3.5056402225563943E-12, ... ], ... ]
    Error Statistics: {meanExponent=-11.839960699300331, negative=1431, min=-4.460820601792648E-12, max=-4.460820601792648E-12, mean=-3.165756294512602E-14, count=24300.0, positive=1611, stdDev=1.2624602802640553E-12, zeros=21258}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 4.4157e-13 +- 1.4236e-12 [0.0000e+00 - 1.6633e-11] (35550#)
    relativeTol: 7.5351e-12 +- 4.6843e-11 [3.3077e-15 - 1.4022e-09] (6084#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=4.4157e-13 +- 1.4236e-12 [0.0000e+00 - 1.6633e-11] (35550#), relativeTol=7.5351e-12 +- 4.6843e-11 [3.3077e-15 - 1.4022e-09] (6084#)}
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
      "id": "cc896657-b2a8-4ba3-ba45-836a88a1fb3e",
      "isFrozen": false,
      "name": "ConvolutionLayer/cc896657-b2a8-4ba3-ba45-836a88a1fb3e",
      "filter": [
        [
          [
            0.64,
            1.58,
            1.564
          ],
          [
            0.696,
            -1.996,
            -1.916
          ],
          [
            0.696,
            -0.672,
            -0.976
          ]
        ],
        [
          [
            -1.012,
            1.924,
            -0.232
          ],
          [
            2.0,
            -1.34,
            -1.684
          ],
          [
            1.652,
            -0.796,
            0.676
          ]
        ],
        [
          [
            -0.072,
            -1.42,
            -0.668
          ],
          [
            -0.92,
            0.288,
            1.768
          ],
          [
            -0.192,
            0.192,
            0.768
          ]
        ],
        [
          [
            -0.128,
            -0.576,
            -0.168
          ],
          [
            0.632,
            1.58,
            -0.768
          ],
          [
            1.104,
            -0.204,
            1.168
          ]
        ],
        [
       
```
...[skipping 1793 bytes](etc/42.txt)...
```
          -1.176,
            1.656
          ],
          [
            -1.264,
            1.06,
            0.124
          ],
          [
            -0.844,
            -1.268,
            0.616
          ]
        ],
        [
          [
            -1.388,
            -0.224,
            -0.756
          ],
          [
            -1.88,
            -1.58,
            -0.696
          ],
          [
            -1.484,
            0.772,
            1.708
          ]
        ],
        [
          [
            0.548,
            -1.148,
            1.176
          ],
          [
            -0.576,
            0.424,
            -1.092
          ],
          [
            -0.152,
            -1.244,
            1.308
          ]
        ],
        [
          [
            1.924,
            1.376,
            -0.84
          ],
          [
            1.02,
            -0.416,
            -0.952
          ],
          [
            -0.112,
            1.704,
            -1.68
          ]
        ],
        [
          [
            -0.82,
            -0.544,
            0.656
          ],
          [
            1.544,
            -0.72,
            0.524
          ],
          [
            -0.956,
            1.276,
            1.192
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
    	[ [ 1.6, -1.272, 1.132 ], [ -1.776, -1.768, -1.52 ], [ 0.176, -1.748, 1.908 ], [ 1.024, -1.044, 1.628 ], [ 0.672, 1.788, 1.28 ] ],
    	[ [ 1.064, -1.132, 0.34 ], [ -1.972, -0.348, -0.128 ], [ 0.164, 0.26, 0.32 ], [ 0.82, -0.512, 0.34 ], [ 0.72, 1.352, 0.072 ] ],
    	[ [ -1.372, -0.548, 1.244 ], [ 1.552, -0.656, -0.004 ], [ 1.616, 1.276, -1.832 ], [ 1.54, 0.076, -1.372 ], [ 1.724, -0.092, 1.62 ] ],
    	[ [ -1.688, 0.036, -1.552 ], [ -0.8, 1.596, -1.6 ], [ 1.2, -0.332, 0.984 ], [ -1.604, -1.152, -0.376 ], [ 1.272, 1.872, 0.684 ] ],
    	[ [ -0.472, -0.12, -1.832 ], [ -1.744, 0.944, 1.268 ], [ 1.536, -0.764, 0.5 ], [ -0.172, -1.524, 0.82 ], [ -0.78, -0.796, -0.264 ] ]
    ]
    Error: [
    	[ [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ] ],
    	[ [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ] ],
    	[ [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ] ],
    	[ [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ] ],
    	[ [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ] ]
    ]
    Accuracy:
    absoluteTol: 0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (150#)
    relativeTol: 0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (150#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (150#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (150#)}
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
      "id": "b020b15b-ed2d-42ec-98bd-e27ac8626f61",
      "isFrozen": false,
      "name": "ConvolutionLayer/b020b15b-ed2d-42ec-98bd-e27ac8626f61",
      "filter": [
        [
          [
            0.64,
            1.58,
            1.564
          ],
          [
            0.696,
            -1.996,
            -1.916
          ],
          [
            0.696,
            -0.672,
            -0.976
          ]
        ],
        [
          [
            -1.012,
            1.924,
            -0.232
          ],
          [
            2.0,
            -1.34,
            -1.684
          ],
          [
            1.652,
            -0.796,
            0.676
          ]
        ],
        [
          [
            -0.072,
            -1.42,
            -0.668
          ],
          [
            -0.92,
            0.288,
            1.768
          ],
          [
            -0.192,
            0.192,
            0.768
          ]
        ],
        [
          [
            -0.128,
            -0.576,
            -0.168
          ],
          [
            0.632,
            1.58,
            -0.768
          ],
          [
            1.104,
            -0.204,
            1.168
          ]
        ],
        [
         
```
...[skipping 1764 bytes](etc/43.txt)...
```
    
          [
            -1.396,
            -1.176,
            1.656
          ],
          [
            -1.264,
            1.06,
            0.124
          ],
          [
            -0.844,
            -1.268,
            0.616
          ]
        ],
        [
          [
            -1.388,
            -0.224,
            -0.756
          ],
          [
            -1.88,
            -1.58,
            -0.696
          ],
          [
            -1.484,
            0.772,
            1.708
          ]
        ],
        [
          [
            0.548,
            -1.148,
            1.176
          ],
          [
            -0.576,
            0.424,
            -1.092
          ],
          [
            -0.152,
            -1.244,
            1.308
          ]
        ],
        [
          [
            1.924,
            1.376,
            -0.84
          ],
          [
            1.02,
            -0.416,
            -0.952
          ],
          [
            -0.112,
            1.704,
            -1.68
          ]
        ],
        [
          [
            -0.82,
            -0.544,
            0.656
          ],
          [
            1.544,
            -0.72,
            0.524
          ],
          [
            -0.956,
            1.276,
            1.192
          ]
        ]
      ],
      "strideX": 1,
      "strideY": 1
    }
```



### Example Input/Output Pair
Code from [ReferenceIO.java:68](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/ReferenceIO.java#L68) executed in 0.01 seconds: 
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
    	[ [ 1.528, -0.76, 1.92 ], [ 0.492, 1.912, 0.22 ], [ -1.436, 1.592, -1.156 ], [ 0.052, 1.484, -1.924 ], [ 0.352, 0.292, 0.772 ] ],
    	[ [ 0.628, 0.356, -1.348 ], [ 0.252, 0.8, 1.96 ], [ 0.532, -1.2, 0.844 ], [ 1.848, -1.968, -1.524 ], [ 0.228, 0.452, 1.524 ] ],
    	[ [ -1.5, -1.292, 1.568 ], [ 1.808, 1.904, -0.892 ], [ 1.592, 1.376, -0.304 ], [ 0.012, 1.412, 1.356 ], [ -0.452, 1.2, 1.976 ] ],
    	[ [ 1.644, 0.688, -0.18 ], [ -1.344, 1.668, 0.352 ], [ 1.66, 0.612, 0.52 ], [ 0.704, 1.352, -0.704 ], [ -0.132, -1.6, 0.548 ] ],
    	[ [ 1.032, 0.808, 0.204 ], [ -0.892, 1.86, -1.788 ], [ 0.46, 0.856, -1.012 ], [ -1.396, 1.616, -1.212 ], [ 1.86, 1.872, -0.9 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ -3.2564800000000003, -3.1628319999999994, -2.0845439999999997, 1.5794560000000002, 5.212512, -8.856992000000002 ], [ 0.11620799999999953, -9.695136000000002, 0.9368000000000004, -0.5834560000000004, 8.731408, 6.443664 ], [ 6.300032, -0.9257599999999998, 1.2484640000000007, 1.1335679999999997, 1.81756
```
...[skipping 2794 bytes](etc/44.txt)...
```
    , -2.2 ] ],
    	[ [ 7.572000000000001, 10.908000000000001, -4.9719999999999995 ], [ 11.064, 4.612, -6.924000000000001 ], [ 11.064, 4.612000000000001, -6.924000000000003 ], [ 11.064, 4.612000000000001, -6.9239999999999995 ], [ 5.0120000000000005, -2.564, -4.352 ] ],
    	[ [ 7.572000000000001, 10.908, -4.9719999999999995 ], [ 11.064, 4.611999999999999, -6.9239999999999995 ], [ 11.064, 4.612000000000001, -6.924000000000001 ], [ 11.064000000000002, 4.612000000000001, -6.924 ], [ 5.0120000000000005, -2.564, -4.352 ] ],
    	[ [ 7.572000000000001, 10.908000000000001, -4.9719999999999995 ], [ 11.064, 4.612, -6.924000000000001 ], [ 11.064000000000002, 4.612000000000001, -6.924000000000001 ], [ 11.064000000000002, 4.612, -6.924 ], [ 5.0120000000000005, -2.564, -4.352 ] ],
    	[ [ 7.364, 7.104, -3.4080000000000004 ], [ 9.684000000000001, 1.6839999999999997, -6.1 ], [ 9.684000000000001, 1.6839999999999997, -6.100000000000001 ], [ 9.684, 1.6839999999999997, -6.100000000000001 ], [ 5.444, -4.5200000000000005, -2.0280000000000005 ] ]
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
    	[ [ -1.952, 1.856, -0.088 ], [ -1.72, -1.424, 0.58 ], [ -0.828, 1.992, 1.908 ], [ -0.832, -0.952, -0.248 ], [ 0.612, -1.08, 1.932 ], [ -0.12, 0.596, -0.036 ], [ -0.312, -0.804, 0.492 ], [ -1.284, 0.972, -1.648 ], ... ],
    	[ [ -1.216, -0.84, 1.428 ], [ -1.676, -1.992, -1.4 ], [ 0.88, -0.724, 1.436 ], [ 1.156, 0.216, 1.628 ], [ -0.904, -1.22, 0.756 ], [ 0.036, 1.3, -1.556 ], [ -0.924, 0.896, -0.948 ], [ 0.484, -0.356, -0.02 ], ... ],
    	[ [ 0.848, -1.384, -0.924 ], [ 0.828, -1.456, -0.08 ], [ -1.916, -1.04, -0.048 ], [ -1.66, 1.964, -1.956 ], [ -0.732, -0.932, -0.744 ], [ 0.252, -1.308, 1.912 ], [ -1.112, -0.148, -1.964 ], [ -1.324, 1.84, 0.38 ], ... ],
    	[ [ 0.248, 1.952, 0.2 ], [ 0.684, -0.268, -0.788 ], [ -1.716, 1.624, 1.632 ], [ -0.96, 1.376, 0.788 ], [ -1.424, -0.796, 0.0 ], [ 1.688, 0.312, -0.648 ], [ 0.472, 1.492, 0.056 ], [ 1.448, 1.708, -0.196 ], ... ],
    	[ [ -1.552, 1.008, -1.056 ], [ 1.772, -0.804, -1.988 ], [ 1.748, 0.904, 0.952 ], [ -1.828, 0.056, 0.224 ], [ -1.264, -0.708, -1.616 ], [ -1.976, -1.528, 1.136 ], [ 1.22, 0.568, -1.472 ], [ 1.844, 0.512, -1.928 ], ... ],
    	[ [ -1.18, -0.744, -0.348 ], [ -1.888, -0.972, 0.912 ], [ -1.656, 0.5, -1.68 ], [ -0.748, 0.256, -0.724 ], [ 1.536, -1.86, -1.056 ], [ 0.992, -1.296, -0.7 ], [ -0.824, 1.948, -0.236 ], [ -0.796, 0.48, -0.696 ], ... ],
    	[ [ 1.324, -0.68, 0.556 ], [ 1.036, -0.72, 0.252 ], [ 1.724, -0.164, -0.256 ], [ 1.976, -1.952, -0.108 ], [ -0.552, -1.524, 0.428 ], [ -1.628, 1.516, -1.764 ], [ 0.9, -1.124, -0.44 ], [ -0.736, 0.34, -0.796 ], ... ],
    	[ [ -1.86, 1.9, 1.612 ], [ -0.172, 1.912, 1.688 ], [ -0.28, -1.34, -1.1 ], [ 1.288, 0.412, -0.444 ], [ -0.328, -0.34, 0.284 ], [ 0.124, -0.272, 1.196 ], [ -1.68, -0.54, -1.02 ], [ 1.344, -0.728, -1.236 ], ... ],
    	...
    ]
```



First, we use a conjugate gradient descent method, which converges the fastest for purely linear functions.

Code from [LearningTester.java:225](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L225) executed in 12.28 seconds: 
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
    F(0.0) = LineSearchPoint{point=PointSample{avg=88.64086978407222}, derivative=-0.625510826637745}
    New Minimum: 88.64086978407222 > 88.64086978400937
    F(1.0E-10) = LineSearchPoint{point=PointSample{avg=88.64086978400937}, derivative=-0.6255108266374805}, delta = -6.285461040533846E-11
    New Minimum: 88.64086978400937 > 88.64086978363493
    F(7.000000000000001E-10) = LineSearchPoint{point=PointSample{avg=88.64086978363493}, derivative=-0.6255108266358929}, delta = -4.3729642129619606E-10
    New Minimum: 88.64086978363493 > 88.64086978100644
    F(4.900000000000001E-9) = LineSearchPoint{point=PointSample{avg=88.64086978100644}, derivative=-0.6255108266247792}, delta = -3.0657787419841043E-9
    New Minimum: 88.64086978100644 > 88.64086976261723
    F(3.430000000000001E-8) = LineSearchPoint{point=PointSample{avg=88.64086976261723}, derivative=-0.6255108265469836}, delta = -2.1454994225678092E-8
    New Minimum: 88.64086976261723 > 88.64086963388864
    F(2.4010000000000004E-7) = LineSearchPo
```
...[skipping 286452 bytes](etc/45.txt)...
```
    earchPoint{point=PointSample{avg=8.399657211275518E-11}, derivative=-1.8677093375489983E-14}, delta = -3.993254177671528E-12
    F(1102.5383039300452) = LineSearchPoint{point=PointSample{avg=1.0420045739258802E-10}, derivative=6.143515340668926E-14}, delta = 1.6210631102161314E-11
    F(84.81063876384964) = LineSearchPoint{point=PointSample{avg=8.557829054751457E-11}, derivative=-2.483957389698353E-14}, delta = -2.411535742912139E-12
    F(593.6744713469475) = LineSearchPoint{point=PointSample{avg=8.391385187235013E-11}, derivative=1.829778975464875E-14}, delta = -4.075974418076575E-12
    8.391385187235013E-11 <= 8.798982629042671E-11
    New Minimum: 8.22476146966278E-11 > 8.193908762590693E-11
    F(377.8271725293803) = LineSearchPoint{point=PointSample{avg=8.193908762590693E-11}, derivative=7.96117286361483E-26}, delta = -6.0507386645197785E-12
    Right bracket at 377.8271725293803
    Converged to right
    Iteration 250 complete. Error: 8.193908762590693E-11 Total: 239478259974004.8000; Orientation: 0.0008; Line Search: 0.0546
    
```

Returns: 

```
    8.193908762590693E-11
```



Training Converged

Next, we run the same optimization using L-BFGS, which is nearly ideal for purely second-order or quadratic functions.

Code from [LearningTester.java:249](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L249) executed in 3.56 seconds: 
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
    th(0)=88.64086978407222;dx=-0.625510826637745
    New Minimum: 88.64086978407222 > 87.29938862610288
    WOLFE (weak): th(2.154434690031884)=87.29938862610288; dx=-0.6198099660286127 delta=1.3414811579693406
    New Minimum: 87.29938862610288 > 85.97018959999204
    WOLFE (weak): th(4.308869380063768)=85.97018959999204; dx=-0.6141091054194804 delta=2.670680184080183
    New Minimum: 85.97018959999204 > 80.77621481414761
    WOLFE (weak): th(12.926608140191302)=80.77621481414761; dx=-0.5913056629829511 delta=7.864654969924615
    New Minimum: 80.77621481414761 > 59.835190385991226
    END: th(51.70643256076521)=59.835190385991226; dx=-0.48869017201856907 delta=28.805679398080997
    Iteration 1 complete. Error: 59.835190385991226 Total: 239478318322474.7200; Orientation: 0.0022; Line Search: 0.0380
    LBFGS Accumulation History: 1 points
    th(0)=59.835190385991226;dx=-0.38432846144218547
    New Minimum: 59.835190385991226 > 26.61639365095356
    END: th(111.398132006
```
...[skipping 55994 bytes](etc/46.txt)...
```
    E-8; dx=-7.1410447448707245E-12 delta=5.465348314732292E-9
    New Minimum: 2.8669891587992438E-8 > 1.9194593511399653E-8
    END: th(2194.125121399711)=1.9194593511399653E-8; dx=-5.814412513483523E-12 delta=1.4940646391325077E-8
    Iteration 114 complete. Error: 1.9194593511399653E-8 Total: 239481785637159.2500; Orientation: 0.0015; Line Search: 0.0273
    LBFGS Accumulation History: 1 points
    th(0)=1.9194593511399653E-8;dx=-4.3804308645141644E-12
    New Minimum: 1.9194593511399653E-8 > 6.505417437246489E-9
    END: th(4727.099275813956)=6.505417437246489E-9; dx=-9.882636916070913E-13 delta=1.2689176074153164E-8
    Iteration 115 complete. Error: 6.505417437246489E-9 Total: 239481806613849.2200; Orientation: 0.0015; Line Search: 0.0140
    LBFGS Accumulation History: 1 points
    th(0)=6.505417437246489E-9;dx=-2.7953128697263652E-11
    MAX ALPHA: th(0)=6.505417437246489E-9;th'(0)=-2.7953128697263652E-11;
    Iteration 116 failed, aborting. Error: 6.505417437246489E-9 Total: 239481826037976.2000; Orientation: 0.0015; Line Search: 0.0130
    
```

Returns: 

```
    6.505417437246489E-9
```



Training Converged

Code from [LearningTester.java:95](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L95) executed in 0.00 seconds: 
```java
    return TestUtil.compare(runs);
```

Returns: 

![Result](etc/test.35.png)



Code from [LearningTester.java:98](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L98) executed in 0.00 seconds: 
```java
    return TestUtil.compareTime(runs);
```

Returns: 

![Result](etc/test.36.png)



### Model Learning
In this test, attempt to train a network to emulate a randomized network given an example input/output. The target state is:

Code from [LearningTester.java:176](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L176) executed in 0.00 seconds: 
```java
    return network_target.state().stream().map(Arrays::toString).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [-1.44, 0.656, 0.26, 1.652, 0.664, -1.196, 1.052, 1.972, 0.284, -1.684, 1.924, 0.244, -1.012, -1.644, -1.272, 1.628, -1.484, -0.752, 1.564, 1.58, -0.972, -1.032, 0.688, -0.696, -1.268, 0.656, -0.072, -0.128, 0.54, 0.676, -1.148, -0.112, -1.396, -1.176, -0.956, -0.504, 1.196, -0.444, 1.848, -0.56, 0.124, -0.82, -1.58, 0.772, 0.288, -0.412, 1.656, 1.428, 1.456, -1.02, 0.484, -1.34, 1.376, -0.78, -0.532, -1.68, 0.548, 1.192, -0.072, 1.408, 1.012, -1.092, 1.168, 1.02, 1.404, -0.192, 1.612, -0.992, 0.544, 0.02, -0.152, 0.696, 0.228, 0.1, 0.616, 2.0, 1.66, -1.028, 0.304, -1.264, -0.232, -0.672, -1.88, -0.768, 0.04, -0.952, -0.544, -0.808, 0.864, 1.544, 0.952, 1.068, -0.976, -0.756, 1.58, 1.768, 1.06, 1.176, 1.104, 0.424, 0.576, -1.244, -0.168, 1.336, 1.344, 0.376, -0.416, -0.84, -0.844, 1.968, 0.648, -0.668, -0.724, -0.576, -0.92, -0.576, 1.72, 1.22, -0.484, -1.388, 1.276, -0.76, -1.812, 1.924, -0.204, -1.152, 0.524, -0.224, 0.14, -0.796, -1.316, 0.192, 0.224, -1.184, -0.64, 0.696, 1.972, -0.732, 0.772, 1.672, -1.452, 0.768, -0.72, 0.96, 0.632, 1.708, -1.996, -1.468, -1.812, -1.028, -0.5, 0.64, 1.308, -1.42, 1.704, -1.916, 0.108, -0.976, -1.68, -0.868, 1.644, 1.916]
```



First, we use a conjugate gradient descent method, which converges the fastest for purely linear functions.

Code from [LearningTester.java:225](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L225) executed in 1.97 seconds: 
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
    F(0.0) = LineSearchPoint{point=PointSample{avg=63.891356400011865}, derivative=-56.62538278981789}
    New Minimum: 63.891356400011865 > 63.8913563943496
    F(1.0E-10) = LineSearchPoint{point=PointSample{avg=63.8913563943496}, derivative=-56.625382787301014}, delta = -5.662265323280735E-9
    New Minimum: 63.8913563943496 > 63.89135636037443
    F(7.000000000000001E-10) = LineSearchPoint{point=PointSample{avg=63.89135636037443}, derivative=-56.62538277219986}, delta = -3.9637434667838534E-8
    New Minimum: 63.89135636037443 > 63.89135612254827
    F(4.900000000000001E-9) = LineSearchPoint{point=PointSample{avg=63.89135612254827}, derivative=-56.6253826664916}, delta = -2.7746359876346105E-7
    New Minimum: 63.89135612254827 > 63.89135445776124
    F(3.430000000000001E-8) = LineSearchPoint{point=PointSample{avg=63.89135445776124}, derivative=-56.625381926534075}, delta = -1.942250626996156E-6
    New Minimum: 63.89135445776124 > 63.891342804258315
    F(2.4010000000000004E-7) = LineSearchPoint{po
```
...[skipping 17087 bytes](etc/47.txt)...
```
    ch: 0.1528
    Zero gradient: 6.85930497310722E-16
    F(0.0) = LineSearchPoint{point=PointSample{avg=1.6640448356860234E-30}, derivative=-4.705006471409344E-31}
    New Minimum: 1.6640448356860234E-30 > 5.116728503237047E-32
    F(2.2605249520865267) = LineSearchPoint{point=PointSample{avg=5.116728503237047E-32}, derivative=1.923990595419065E-33}, delta = -1.612877550653653E-30
    5.116728503237047E-32 <= 1.6640448356860234E-30
    Converged to right
    Iteration 16 complete. Error: 5.116728503237047E-32 Total: 239483882535510.1600; Orientation: 0.0001; Line Search: 0.0286
    Zero gradient: 3.9547405639087584E-17
    F(0.0) = LineSearchPoint{point=PointSample{avg=5.116728503237047E-32}, derivative=-1.5639972927825364E-33}
    New Minimum: 5.116728503237047E-32 > 0.0
    F(2.2605249520865267) = LineSearchPoint{point=PointSample{avg=0.0}, derivative=0.0}, delta = -5.116728503237047E-32
    0.0 <= 5.116728503237047E-32
    Converged to right
    Iteration 17 complete. Error: 0.0 Total: 239483927286236.1200; Orientation: 0.0001; Line Search: 0.0294
    
```

Returns: 

```
    0.0
```



Training Converged

Next, we run the same optimization using L-BFGS, which is nearly ideal for purely second-order or quadratic functions.

Code from [LearningTester.java:249](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L249) executed in 1.45 seconds: 
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
    th(0)=98.62315220403448;dx=-87.5552893311908
    New Minimum: 98.62315220403448 > 0.45074143844641684
    END: th(2.154434690031884)=0.45074143844641684; dx=-3.5799037844892108 delta=98.17241076558807
    Iteration 1 complete. Error: 0.45074143844641684 Total: 239483998616612.0600; Orientation: 0.0001; Line Search: 0.0314
    LBFGS Accumulation History: 1 points
    th(0)=0.45074143844641684;dx=-0.37836241854115027
    New Minimum: 0.45074143844641684 > 0.4085774535818904
    WOLF (strong): th(4.641588833612779)=0.4085774535818904; dx=0.36019450820873034 delta=0.04216398486452644
    New Minimum: 0.4085774535818904 > 0.0011497479904498881
    END: th(2.3207944168063896)=0.0011497479904498881; dx=-0.00908395516621018 delta=0.449591690455967
    Iteration 2 complete. Error: 0.0011497479904498881 Total: 239484057270010.0000; Orientation: 0.0001; Line Search: 0.0442
    LBFGS Accumulation History: 1 points
    th(0)=0.0011497479904498881;dx=-9.896867287621503E-4
    Armijo:
```
...[skipping 9933 bytes](etc/48.txt)...
```
    ientation: 0.0001; Line Search: 0.0330
    LBFGS Accumulation History: 1 points
    th(0)=4.868419639460498E-32;dx=-2.20652359682739E-33
    New Minimum: 4.868419639460498E-32 > 4.3998318963180095E-32
    WOLF (strong): th(4.058526479522111)=4.3998318963180095E-32; dx=1.9675059158203643E-33 delta=4.685877431424883E-33
    New Minimum: 4.3998318963180095E-32 > 1.0961879662133644E-33
    END: th(2.0292632397610557)=1.0961879662133644E-33; dx=-1.6632510758777396E-35 delta=4.7588008428391614E-32
    Iteration 22 complete. Error: 1.0961879662133644E-33 Total: 239485321467773.7500; Orientation: 0.0001; Line Search: 0.0500
    LBFGS Accumulation History: 1 points
    th(0)=1.0961879662133644E-33;dx=-1.004523247786683E-35
    Armijo: th(4.371915118947706)=1.6952292161155703E-33; dx=8.453806006922945E-36 delta=-5.990412499022059E-34
    New Minimum: 1.0961879662133644E-33 > 0.0
    END: th(2.185957559473853)=0.0; dx=0.0 delta=1.0961879662133644E-33
    Iteration 23 complete. Error: 0.0 Total: 239485381461994.7000; Orientation: 0.0001; Line Search: 0.0455
    
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

![Result](etc/test.37.png)



Code from [LearningTester.java:98](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L98) executed in 0.00 seconds: 
```java
    return TestUtil.compareTime(runs);
```

Returns: 

![Result](etc/test.38.png)



### Performance
Now we execute larger-scale runs to benchmark performance:

Code from [PerformanceTester.java:66](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/PerformanceTester.java#L66) executed in 1.79 seconds: 
```java
    test(component, inputPrototype);
```
Logging: 
```
    100 batches
    Input Dimensions:
    	[100, 100, 3]
    Performance:
    	Evaluation performance: 0.073454s +- 0.014514s [0.053916s - 0.092639s]
    	Learning performance: 0.216994s +- 0.026557s [0.188578s - 0.250676s]
    
```

