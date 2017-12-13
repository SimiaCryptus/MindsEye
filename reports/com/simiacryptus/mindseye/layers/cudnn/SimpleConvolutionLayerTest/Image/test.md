# SimpleConvolutionLayer
## Image
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
      "class": "com.simiacryptus.mindseye.layers.cudnn.SimpleConvolutionLayer",
      "id": "7481dea2-9bfe-48c2-a112-3575488ffc0f",
      "isFrozen": false,
      "name": "SimpleConvolutionLayer/7481dea2-9bfe-48c2-a112-3575488ffc0f",
      "filter": [
        [
          [
            -0.892,
            0.484,
            -0.296
          ],
          [
            0.764,
            0.948,
            1.688
          ],
          [
            -0.528,
            0.02,
            1.72
          ]
        ],
        [
          [
            1.604,
            -1.288,
            -0.564
          ],
          [
            -1.472,
            0.584,
            -1.508
          ],
          [
            -1.58,
            -1.184,
            1.392
          ]
        ],
        [
          [
            1.716,
            -1.904,
            1.352
          ],
          [
            -0.836,
            -1.756,
            -1.688
          ],
          [
            1.468,
            -0.856,
            0.492
          ]
        ],
        [
          [
            -0.716,
            1.716,
            1.688
          ],
          [
            -1.504,
            0.372,
            -0.328
          ],
          [
            -0.58,
            -1.94,
            -1.68
          ]
       
```
...[skipping 21 bytes](etc/45.txt)...
```
         -1.54,
            -0.48,
            -1.172
          ],
          [
            0.916,
            -1.928,
            -0.028
          ],
          [
            0.976,
            1.224,
            0.568
          ]
        ],
        [
          [
            -0.968,
            1.868,
            0.98
          ],
          [
            0.92,
            1.696,
            -1.496
          ],
          [
            -0.22,
            1.864,
            -0.048
          ]
        ],
        [
          [
            1.56,
            -0.468,
            -1.584
          ],
          [
            -0.456,
            0.328,
            0.008
          ],
          [
            1.428,
            -1.032,
            -0.916
          ]
        ],
        [
          [
            -0.636,
            1.864,
            0.212
          ],
          [
            -0.52,
            1.804,
            0.356
          ],
          [
            -0.596,
            0.868,
            1.816
          ]
        ],
        [
          [
            0.648,
            0.636,
            0.384
          ],
          [
            -0.52,
            1.916,
            0.972
          ],
          [
            1.084,
            1.036,
            -1.276
          ]
        ]
      ],
      "strideX": 1,
      "strideY": 1,
      "simple": false
    }
```



### Example Input/Output Pair
Code from [StandardLayerTests.java:153](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L153) executed in 0.00 seconds: 
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
    	[ [ -0.908, 0.388, -0.676 ], [ 0.848, 1.42, -0.78 ], [ -1.024, 0.668, 0.136 ] ],
    	[ [ -0.1, 1.872, -0.652 ], [ 0.4, 1.32, 0.896 ], [ 1.26, -1.852, 0.872 ] ],
    	[ [ -0.56, -0.584, 1.34 ], [ 1.464, -1.48, -1.88 ], [ -1.04, 1.472, 0.756 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 1.6305600000000002, -4.837024, 0.637136 ], [ -7.357744, -0.026415999999999187, 4.138256 ], [ 0.16222400000000012, -4.7291680000000005, 1.3699039999999998 ] ],
    	[ [ -11.746672000000002, 1.8662720000000004, 4.939392000000001 ], [ 11.254287999999999, -2.7585759999999997, 5.775391999999998 ], [ -8.441952, 9.289216000000001, 3.8137119999999993 ] ],
    	[ [ 1.4266719999999997, 3.863071999999999, -3.287136 ], [ 0.6550719999999999, 6.905920000000002, 2.992304000000001 ], [ 8.351103999999998, -11.729904000000001, 0.10547199999999983 ] ]
    ]
    --------------------
    Derivative: 
    [
    	[ [ -0.8120000000000003, 3.964, 0.8559999999999997 ], [ 0.7279999999999998, 2.536, 4.172 ], [ 4.555999999999999, -2.1479999999999997, 2.9599999999999995 ] ],
    	[ [ -1.6880000000000004, 1.6879999999999993, 2.752 ], [ -0.19600000000000062, -0.3120000000000003, 7.464 ], [ 3.3120000000000003, -3.7960000000000003, 3.92 ] ],
    	[ [ -2.18, -0.908, 5.795999999999999 ], [ -0.496, -1.384, 7.792 ], [ 2.136, -1.0919999999999996, 3.416 ] ]
    ]
```



[GPU Log](etc/cuda.log)

### Reference Implementation
Code from [StandardLayerTests.java:93](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L93) executed in 0.01 seconds: 
```java
    System.out.println(new GsonBuilder().setPrettyPrinting().create().toJson(referenceLayer.getJson()));
    getEquivalencyTester().test(referenceLayer, layer, inputPrototype);
```
Logging: 
```
    {
      "class": "com.simiacryptus.mindseye.layers.aparapi.ConvolutionLayer",
      "id": "ff89a7b6-f078-4bbf-839a-57e21178d35e",
      "isFrozen": false,
      "name": "ConvolutionLayer/ff89a7b6-f078-4bbf-839a-57e21178d35e",
      "filter": [
        [
          [
            -0.892,
            0.484,
            -0.296
          ],
          [
            0.764,
            0.948,
            1.688
          ],
          [
            -0.528,
            0.02,
            1.72
          ]
        ],
        [
          [
            -0.716,
            1.716,
            1.688
          ],
          [
            -1.504,
            0.372,
            -0.328
          ],
          [
            -0.58,
            -1.94,
            -1.68
          ]
        ],
        [
          [
            1.56,
            -0.468,
            -1.584
          ],
          [
            -0.456,
            0.328,
            0.008
          ],
          [
            1.428,
            -1.032,
            -0.916
          ]
        ],
        [
          [
            1.604,
            -1.288,
            -0.564
          ],
          [
            -1.472,
            0.584,
            -1.508
          ],
          [
            -1.58,
            -1.184,
            1.392
          ]
        ],
        [
     
```
...[skipping 629 bytes](etc/46.txt)...
```
    68,
            0.98
          ],
          [
            0.92,
            1.696,
            -1.496
          ],
          [
            -0.22,
            1.864,
            -0.048
          ]
        ],
        [
          [
            0.648,
            0.636,
            0.384
          ],
          [
            -0.52,
            1.916,
            0.972
          ],
          [
            1.084,
            1.036,
            -1.276
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
    	[ [ -0.444, 1.28, -1.172 ], [ 1.14, -0.692, -1.2 ], [ 1.252, -0.248, -0.532 ] ],
    	[ [ 0.816, 1.54, 1.96 ], [ -1.664, -0.72, -0.484 ], [ 1.896, 1.508, 1.796 ] ],
    	[ [ 0.988, -1.788, 1.792 ], [ 1.204, -1.544, -0.144 ], [ -1.928, -0.688, -1.016 ] ]
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
Code from [StandardLayerTests.java:102](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L102) executed in 0.01 seconds: 
```java
    return getBatchingTester().test(layer, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=1.2336e-18 +- 2.0232e-17 [0.0000e+00 - 3.3307e-16] (540#), relativeTol=1.9769e-18 +- 3.2423e-17 [0.0000e+00 - 5.3376e-16] (540#)}
```



### Differential Validation
Code from [StandardLayerTests.java:110](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L110) executed in 0.12 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ -1.536, 0.224, 0.484 ], [ -0.32, 1.416, 1.18 ], [ -1.536, 1.788, -1.1 ] ],
    	[ [ -1.632, 1.068, 1.988 ], [ -0.356, 0.344, -1.792 ], [ 1.12, 1.908, -0.592 ] ],
    	[ [ -0.88, -0.444, 1.688 ], [ -0.068, 1.78, -0.92 ], [ -1.392, -1.932, 1.652 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.039758710739839914, negative=14, min=1.652, max=1.652, mean=0.07925925925925921, count=27.0, positive=13, stdDev=1.2972586649915845, zeros=0}
    Output: [
    	[ [ -13.08728, 7.514608000000001, 1.5870080000000002 ], [ 0.6472159999999991, -4.654160000000002, 10.596832000000001 ], [ -5.617552000000001, -0.8756800000000006, -0.45500800000000013 ] ],
    	[ [ -4.56672, -4.020048000000002, 4.586671999999999 ], [ -6.858544000000001, 2.4273439999999997, 10.180655999999999 ], [ -0.8236320000000008, 0.7127999999999993, 1.023616 ] ],
    	[ [ -14.864063999999999, -4.52704, 7.157408 ], [ -0.322064, 6.032496, -0.522896 ], [ -7.672719999999999, 7.948959999999998, 3.2356 ] ]
    ]
    Outputs Statistics: {meanExponent=0.4802000003946145, negative=14, min=
```
...[skipping 6677 bytes](etc/47.txt)...
```
     ], [ 9.122036459530136E-12, -2.5204283105040304E-12, 1.3403722576299515E-12, -3.2002178684820137E-13, 1.4187873098592263E-11, 4.705957845629882E-12, 2.402522625288839E-13, 5.560885085742484E-12, ... ], [ 0.0, 2.402522625288839E-13, -1.1402212507505283E-11, 0.0, 8.561762410153051E-12, 5.306088901591011E-12, 0.0, 2.402522625288839E-13, ... ], [ 0.0, 0.0, 0.0, 1.9204637879965958E-12, 1.3403722576299515E-12, 0.0, 5.306088901591011E-12, 4.705957845629882E-12, ... ], [ 0.0, 0.0, 0.0, 2.402522625288839E-13, 6.361355886497222E-12, 1.3403722576299515E-12, -3.2002178684820137E-13, 5.306088901591011E-12, ... ], ... ]
    Error Statistics: {meanExponent=-11.741829123090339, negative=163, min=1.7605916724505732E-12, max=1.7605916724505732E-12, mean=2.1367001040516906E-13, count=2187.0, positive=278, stdDev=2.125619893058174E-12, zeros=1746}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 1.0889e-12 +- 2.4363e-12 [0.0000e+00 - 1.6624e-11] (2916#)
    relativeTol: 5.5987e-12 +- 2.2901e-11 [2.5689e-15 - 2.9193e-10] (882#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=1.0889e-12 +- 2.4363e-12 [0.0000e+00 - 1.6624e-11] (2916#), relativeTol=5.5987e-12 +- 2.2901e-11 [2.5689e-15 - 2.9193e-10] (882#)}
```



### Performance
Code from [StandardLayerTests.java:120](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L120) executed in 0.01 seconds: 
```java
    getPerformanceTester().test(layer, permPrototype);
```
Logging: 
```
    Evaluation performance: 0.000555s +- 0.000119s [0.000416s - 0.000713s]
    Learning performance: 0.000516s +- 0.000033s [0.000474s - 0.000571s]
    
```

