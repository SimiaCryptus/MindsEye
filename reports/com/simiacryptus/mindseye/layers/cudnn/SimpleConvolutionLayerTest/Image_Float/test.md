# SimpleConvolutionLayer
## Image_Float
### Batch Execution
Code from [BatchingTester.java:66](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/BatchingTester.java#L66) executed in 0.01 seconds: 
```java
    return test(reference, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=1.6448e-18 +- 1.9040e-17 [0.0000e+00 - 2.2204e-16] (540#), relativeTol=2.0767e-18 +- 2.4040e-17 [0.0000e+00 - 2.8036e-16] (540#)}
```



Code from [SingleDerivativeTester.java:77](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/SingleDerivativeTester.java#L77) executed in 0.12 seconds: 
```java
    return test(component, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ -1.708, -1.596, 0.348 ], [ -0.232, -0.316, 0.584 ], [ 0.604, 1.976, 1.832 ] ],
    	[ [ -1.7, -0.696, -0.816 ], [ -0.688, 0.592, -1.028 ], [ -0.136, -0.328, -1.516 ] ],
    	[ [ -1.116, 0.18, -0.204 ], [ -1.572, -0.836, -1.468 ], [ -1.3, 1.796, 1.856 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.11407480773136823, negative=18, min=1.856, max=1.856, mean=-0.2773333333333334, count=27.0, positive=9, stdDev=1.1382148466071935, zeros=0}
    Output: [
    	[ [ -1.3675680000000001, -4.758464000000001, -2.6246560000000008 ], [ -0.11502400000000057, 3.4482720000000002, 3.959232 ], [ -1.852592, 2.4438239999999998, -1.9352639999999999 ] ],
    	[ [ -2.794224, -9.704688, 3.976752 ], [ -4.733359999999999, -1.0170239999999993, 2.175055999999999 ], [ -2.3555039999999994, -4.554672, 1.848799999999999 ] ],
    	[ [ -1.0225919999999997, -5.950127999999999, -6.861952 ], [ 2.85536, -1.53168, 5.460735999999999 ], [ 3.6820480000000004, 0.8264160000000002, -1.5844480000000003 ] ]
    ]
    Outputs Statistics: {meanExponent=0.38530277640015204, nega
```
...[skipping 6749 bytes](etc/157.txt)...
```
    232792060245629E-13, -3.4861002973229915E-14, -1.1679546219056647E-13, -9.96425164601078E-15, -2.440270208126094E-13, 9.325873406851315E-14, 4.8960835385969403E-14, -3.58046925441613E-13, ... ], [ 0.0, 1.7918999617450027E-13, -3.4861002973229915E-14, 0.0, -4.540534614960734E-13, 2.000621890374532E-13, 0.0, 4.8960835385969403E-14, ... ], [ 0.0, 0.0, 0.0, -3.4861002973229915E-14, 3.2729374765949615E-13, 0.0, 6.441513988875158E-13, -3.5083047578154947E-13, ... ], [ 0.0, 0.0, 0.0, 1.7918999617450027E-13, -4.789502128232925E-13, 3.2729374765949615E-13, 4.3412495820405184E-13, -2.440270208126094E-13, ... ], ... ]
    Error Statistics: {meanExponent=-12.728802346366251, negative=229, min=-2.886579864025407E-14, max=-2.886579864025407E-14, mean=1.2963516354979566E-15, count=2187.0, positive=212, stdDev=1.587606291508803E-13, zeros=1746}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 7.9446e-14 +- 1.6973e-13 [0.0000e+00 - 1.4744e-12] (2916#)
    relativeTol: 2.2652e-13 +- 3.7469e-13 [4.5501e-16 - 5.1260e-12] (882#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=7.9446e-14 +- 1.6973e-13 [0.0000e+00 - 1.4744e-12] (2916#), relativeTol=2.2652e-13 +- 3.7469e-13 [4.5501e-16 - 5.1260e-12] (882#)}
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
      "id": "d2c73d23-9162-4177-852e-3bd7c5c61be7",
      "isFrozen": false,
      "name": "ConvolutionLayer/d2c73d23-9162-4177-852e-3bd7c5c61be7",
      "filter": [
        [
          [
            -0.204,
            0.64,
            1.22
          ],
          [
            1.676,
            -0.768,
            1.056
          ],
          [
            0.704,
            -0.96,
            -0.384
          ]
        ],
        [
          [
            -0.968,
            0.9,
            -1.9
          ],
          [
            1.048,
            1.056,
            1.384
          ],
          [
            -0.816,
            1.484,
            0.212
          ]
        ],
        [
          [
            -0.524,
            1.78,
            0.672
          ],
          [
            0.44,
            1.36,
            -1.508
          ],
          [
            1.156,
            -1.248,
            -1.868
          ]
        ],
        [
          [
            -1.124,
            -0.668,
            1.564
          ],
          [
            0.944,
            -0.892,
            -1.98
          ],
          [
            0.544,
            1.296,
            -1.292
          ]
        ],
        [
          [
     
```
...[skipping 10 bytes](etc/158.txt)...
```
    72,
            1.832,
            1.82
          ],
          [
            -0.44,
            0.448,
            0.788
          ],
          [
            -1.356,
            -1.192,
            -1.864
          ]
        ],
        [
          [
            -1.124,
            0.784,
            -1.508
          ],
          [
            -1.456,
            0.928,
            0.524
          ],
          [
            -0.532,
            0.52,
            -1.648
          ]
        ],
        [
          [
            0.244,
            1.52,
            -0.556
          ],
          [
            1.496,
            1.292,
            -1.204
          ],
          [
            -1.164,
            -0.06,
            0.444
          ]
        ],
        [
          [
            1.176,
            0.768,
            0.096
          ],
          [
            0.924,
            1.872,
            -0.268
          ],
          [
            1.408,
            0.936,
            -1.868
          ]
        ],
        [
          [
            -2.0,
            0.808,
            1.236
          ],
          [
            0.032,
            -1.788,
            0.992
          ],
          [
            0.28,
            -0.704,
            -0.424
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
    	[ [ -0.72, -1.132, 1.164 ], [ 0.348, -1.996, 1.048 ], [ -1.644, -0.276, -1.204 ] ],
    	[ [ 1.652, -0.396, -1.848 ], [ -1.184, 1.212, 0.312 ], [ -1.76, 1.412, -1.964 ] ],
    	[ [ 1.34, -0.884, 1.6 ], [ -1.988, -1.232, -1.8 ], [ 1.392, 1.496, -0.044 ] ]
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

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (27#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (27#)}
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
      "class": "com.simiacryptus.mindseye.layers.cudnn.SimpleConvolutionLayer",
      "id": "dae90c54-33f7-4bf7-ae0c-bcb25b7db77a",
      "isFrozen": false,
      "name": "SimpleConvolutionLayer/dae90c54-33f7-4bf7-ae0c-bcb25b7db77a",
      "filter": [
        [
          [
            -0.204,
            0.64,
            1.22
          ],
          [
            1.676,
            -0.768,
            1.056
          ],
          [
            0.704,
            -0.96,
            -0.384
          ]
        ],
        [
          [
            -1.124,
            -0.668,
            1.564
          ],
          [
            0.944,
            -0.892,
            -1.98
          ],
          [
            0.544,
            1.296,
            -1.292
          ]
        ],
        [
          [
            0.244,
            1.52,
            -0.556
          ],
          [
            1.496,
            1.292,
            -1.204
          ],
          [
            -1.164,
            -0.06,
            0.444
          ]
        ],
        [
          [
            -0.968,
            0.9,
            -1.9
          ],
          [
            1.048,
            1.056,
            1.384
          ],
          [
            -0.816,
            1.484,
            0.212
          ]
        ],
        
```
...[skipping 12 bytes](etc/159.txt)...
```
          0.372,
            1.832,
            1.82
          ],
          [
            -0.44,
            0.448,
            0.788
          ],
          [
            -1.356,
            -1.192,
            -1.864
          ]
        ],
        [
          [
            1.176,
            0.768,
            0.096
          ],
          [
            0.924,
            1.872,
            -0.268
          ],
          [
            1.408,
            0.936,
            -1.868
          ]
        ],
        [
          [
            -0.524,
            1.78,
            0.672
          ],
          [
            0.44,
            1.36,
            -1.508
          ],
          [
            1.156,
            -1.248,
            -1.868
          ]
        ],
        [
          [
            -1.124,
            0.784,
            -1.508
          ],
          [
            -1.456,
            0.928,
            0.524
          ],
          [
            -0.532,
            0.52,
            -1.648
          ]
        ],
        [
          [
            -2.0,
            0.808,
            1.236
          ],
          [
            0.032,
            -1.788,
            0.992
          ],
          [
            0.28,
            -0.704,
            -0.424
          ]
        ]
      ],
      "strideX": 1,
      "strideY": 1,
      "simple": false
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
    	[ [ -1.376, -0.34, 1.604 ], [ -0.724, -0.916, -1.42 ], [ 0.88, 1.308, 1.796 ] ],
    	[ [ 0.412, 0.092, 1.972 ], [ 0.36, 0.188, -0.04 ], [ -0.924, 1.576, -1.348 ] ],
    	[ [ -1.188, 1.712, -1.568 ], [ 0.532, -1.912, -1.492 ], [ -1.188, 0.284, -1.444 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 4.855504000000001, -0.13511999999999935, -8.418576 ], [ -0.4042400000000002, 3.0677600000000003, 7.430175999999999 ], [ -1.6042560000000006, 0.05336000000000014, -1.8131200000000007 ] ],
    	[ [ -3.0882719999999995, -4.652416, 2.2920799999999995 ], [ 0.32143999999999967, -3.3412159999999997, 3.3135039999999996 ], [ -8.341391999999999, 0.8069280000000003, 7.1288 ] ],
    	[ [ -4.656624, -7.82416, 2.311168 ], [ 2.419312, -8.63328, -4.8701919999999985 ], [ -6.9778720000000005, -2.03344, 0.538592 ] ]
    ]
    --------------------
    Derivative: 
    [
    	[ [ -0.18399999999999994, -4.364, -0.7799999999999998 ], [ 3.128, -0.5399999999999991, 3.0920000000000005 ], [ 5.892000000000001, 3.6400000000000006, 4.768000000000001 ] ],
    	[ [ 4.024, -6.66, 2.196 ], [ 5.640000000000001, -4.712, 5.488000000000001 ], [ 7.360000000000001, 0.8120000000000003, 6.640000000000001 ] ],
    	[ [ 5.132, -1.188, 4.524000000000001 ], [ 6.756, -1.116, 7.04 ], [ 6.436, -0.3959999999999999, 6.344 ] ]
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
    	[ [ -1.856, 0.676, -0.34 ], [ 1.4, -0.268, -1.58 ], [ -1.496, -0.684, -0.692 ], [ 0.968, -1.82, 0.976 ], [ -0.504, -1.944, 1.096 ], [ 0.984, 0.436, -1.512 ], [ -0.168, 0.452, -1.932 ], [ -1.008, 1.412, -1.684 ], ... ],
    	[ [ 1.208, -1.148, 0.624 ], [ 0.94, -0.348, 1.748 ], [ -1.192, 1.7, -1.324 ], [ -1.704, -1.38, 1.872 ], [ 1.424, 0.644, -1.012 ], [ -0.8, -1.424, 0.176 ], [ -1.912, 1.516, -1.12 ], [ 0.784, 0.084, -0.008 ], ... ],
    	[ [ -0.46, 1.132, 0.54 ], [ -1.896, -0.108, 1.476 ], [ 0.7, 0.94, -0.376 ], [ 0.76, -1.816, -0.824 ], [ 1.604, 1.048, 1.132 ], [ 0.14, -1.264, 0.244 ], [ -1.688, -0.84, 0.152 ], [ -0.66, 1.82, 0.5 ], ... ],
    	[ [ -1.004, -0.268, 0.828 ], [ -0.364, 1.508, 0.988 ], [ 1.072, 0.444, 0.192 ], [ -0.552, 1.216, 1.54 ], [ -0.108, -0.832, 0.46 ], [ 0.628, -1.136, 1.476 ], [ 0.444, 0.108, 1.0 ], [ 0.136, 0.004, -0.476 ], ... ],
    	[ [ 0.052, -1.892, -1.48 ], [ -1.78, -1.728, -1.796 ], [ -1.488, 1.64, -0.62 ], [ 0.48, -0.452, 0.492 ], [ -1.348, 1.724, -0.188 ], [ 0.32, -1.86, -0.776 ], [ -1.44, -1.716, -1.92 ], [ -0.012, 1.064, -0.936 ], ... ],
    	[ [ -0.356, -1.864, 1.912 ], [ -0.764, -1.02, -0.444 ], [ -0.484, -1.228, 1.912 ], [ 1.36, -0.572, -1.6 ], [ 1.448, 0.388, 0.568 ], [ -1.48, -1.124, -0.42 ], [ -1.252, 1.128, 1.084 ], [ 1.664, -0.56, 1.336 ], ... ],
    	[ [ -0.108, -1.088, -1.772 ], [ -0.668, -1.932, 0.58 ], [ -1.112, -1.276, 0.148 ], [ -0.568, -0.04, -1.944 ], [ 0.9, 1.872, 0.096 ], [ -1.708, -0.804, 1.484 ], [ 1.516, -1.976, -0.988 ], [ -0.216, 1.58, -0.408 ], ... ],
    	[ [ -0.248, 0.172, 1.072 ], [ 0.84, 1.968, 0.564 ], [ 1.124, -0.704, 1.752 ], [ -1.244, -1.144, 0.924 ], [ 0.448, 1.896, -0.052 ], [ 1.26, -0.216, 0.348 ], [ 0.004, 0.996, -1.052 ], [ 1.412, 1.924, -0.5 ], ... ],
    	...
    ]
```



First, we use a conjugate gradient descent method, which converges the fastest for purely linear functions.

Code from [LearningTester.java:225](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L225) executed in 7.35 seconds: 
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
    F(0.0) = LineSearchPoint{point=PointSample{avg=91.0498178080895}, derivative=-0.8547671979904553}
    New Minimum: 91.0498178080895 > 91.04981780800438
    F(1.0E-10) = LineSearchPoint{point=PointSample{avg=91.04981780800438}, derivative=-0.8547671979899318}, delta = -8.512301974406E-11
    New Minimum: 91.04981780800438 > 91.04981780749253
    F(7.000000000000001E-10) = LineSearchPoint{point=PointSample{avg=91.04981780749253}, derivative=-0.8547671979867912}, delta = -5.969695848762058E-10
    New Minimum: 91.04981780749253 > 91.04981780390258
    F(4.900000000000001E-9) = LineSearchPoint{point=PointSample{avg=91.04981780390258}, derivative=-0.8547671979648068}, delta = -4.186915703030536E-9
    New Minimum: 91.04981780390258 > 91.04981777877185
    F(3.430000000000001E-8) = LineSearchPoint{point=PointSample{avg=91.04981777877185}, derivative=-0.8547671978109163}, delta = -2.9317646976778633E-8
    New Minimum: 91.04981777877185 > 91.04981760286071
    F(2.4010000000000004E-7) = LineSearchPoint{po
```
...[skipping 294166 bytes](etc/160.txt)...
```
    80379E-7}
    F(310.2151982431467) = LineSearchPoint{point=PointSample{avg=0.0024624596630031894}, derivative=4.6067604308184626E-7}, delta = 2.214745515709141E-5
    New Minimum: 0.002440312207846098 > 0.0024334410945831375
    F(23.86270755716513) = LineSearchPoint{point=PointSample{avg=0.0024334410945831375}, derivative=-2.579987878534307E-7}, delta = -6.871113262960516E-6
    New Minimum: 0.0024334410945831375 > 0.0024222260878141904
    F(167.03895290015592) = LineSearchPoint{point=PointSample{avg=0.0024222260878141904}, derivative=1.0133862761420816E-7}, delta = -1.8086120031907647E-5
    0.0024222260878141904 <= 0.002440312207846098
    New Minimum: 0.0024222260878141904 > 0.0024201801681768176
    F(126.66106964763806) = LineSearchPoint{point=PointSample{avg=0.0024201801681768176}, derivative=-2.7967327900619487E-22}, delta = -2.013203966928041E-5
    Left bracket at 126.66106964763806
    Converged to left
    Iteration 250 complete. Error: 0.0024201801681768176 Total: 239615990504855.0300; Orientation: 0.0008; Line Search: 0.0217
    
```

Returns: 

```
    0.0024201801681768176
```



This training run resulted in the following regressed input:

Code from [LearningTester.java:144](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L144) executed in 0.01 seconds: 
```java
    return Arrays.stream(input_gd).map(x -> x.prettyPrint()).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [
    	[ [ -1.087774740489363, 0.605413292100432, -1.085986952317327 ], [ 1.2432402647071574, 0.401086726887776, -1.4163569883768368 ], [ -1.6338617641622184, -0.9794564376487468, -0.829117771348147 ], [ 1.3021143499916892, -2.326313797728525, 1.1219622579092596 ], [ -0.48631361475778134, -1.5556190680965492, 1.0064120549382738 ], [ 1.0927556783025032, 0.70184722964664, -1.554210302675974 ], [ -0.4896590974332472, 0.4182063104298799, -1.8399036428746682 ], [ -0.9220573749614294, 1.1000420572189458, -1.8345106233153086 ], ... ],
    	[ [ 1.331375500119649, -1.0583662854562679, 0.38486240649091186 ], [ 0.9653487594583758, 0.20230656976308586, 1.7874048168023435 ], [ -0.6997608787287654, 1.3653449565122744, -0.9883759530994821 ], [ -2.2242056285576783, -1.3549636615449396, 1.6372788590358591 ], [ 1.258546770910704, 0.5722848208019263, -1.2017896965332977 ], [ -0.5163372826397296, -1.3149115596962226, 0.20737828385257515 ], [ -1.6160247925811646, 1.6302004733122557, -0.7475863862541603 ], [ 0.7793965190372241, -0.0108833
```
...[skipping 2240 bytes](etc/161.txt)...
```
    0302 ], [ -0.6773037438110735, -2.0249636564392373, 0.5416725328357471 ], [ -1.2790051189416833, -1.2345346781327295, 3.7870077686853103E-4 ], [ -0.48884103530170453, -0.05036039097339106, -1.9664467876598526 ], [ 0.8909822276966369, 2.0584840009229293, 0.1412701035763813 ], [ -1.6083735380267334, -0.930531194678172, 1.5121545675398056 ], [ 1.3144349527846135, -1.9985960547894315, -1.081175257279513 ], [ -0.15152492528277595, 1.4823200762398163, -0.3781423426819817 ], ... ],
    	[ [ -0.46162833850235985, -0.03803898114300493, 0.9042189919078105 ], [ 0.7914894311193437, 1.9426774552857038, 0.49490589337320273 ], [ 1.2993054823254926, -0.7125766294289989, 1.855958108226849 ], [ -1.2171691898421224, -1.1209438394218985, 0.88357967386419 ], [ 0.45094073594594586, 1.8729087131211601, -0.041873529720757016 ], [ 1.216119001065132, -0.11679919101462581, 0.25756591605577916 ], [ -0.05367204619880702, 1.0647009480072864, -0.9471023827977464 ], [ 1.4878012301632326, 1.7163874009453923, -0.48563683625377063 ], ... ],
    	...
    ]
```



Next, we run the same optimization using L-BFGS, which is nearly ideal for purely second-order or quadratic functions.

Code from [LearningTester.java:249](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L249) executed in 1.32 seconds: 
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
    th(0)=91.0498178080895;dx=-0.8547671979904553
    New Minimum: 91.0498178080895 > 89.220425611347
    WOLFE (weak): th(2.154434690031884)=89.220425611347; dx=-0.8434900805505956 delta=1.8293921967425035
    New Minimum: 89.220425611347 > 87.4153292276185
    WOLFE (weak): th(4.308869380063768)=87.4153292276185; dx=-0.832212963110736 delta=3.6344885804710003
    New Minimum: 87.4153292276185 > 80.43790182286638
    WOLFE (weak): th(12.926608140191302)=80.43790182286638; dx=-0.7871044933512973 delta=10.61191598522312
    New Minimum: 80.43790182286638 > 53.8500494786498
    END: th(51.70643256076521)=53.8500494786498; dx=-0.5841163794338232 delta=37.1997683294397
    Iteration 1 complete. Error: 53.8500494786498 Total: 239616047686054.0000; Orientation: 0.0015; Line Search: 0.0256
    LBFGS Accumulation History: 1 points
    th(0)=53.8500494786498;dx=-0.4124883876358266
    New Minimum: 53.8500494786498 > 21.385839089624355
    END: th(111.39813200670669)=21.3858390896243
```
...[skipping 33408 bytes](etc/162.txt)...
```
    OLFE (weak): th(196.13574989738325)=0.022849745669531303; dx=-2.0870488577979535E-6 delta=4.118825248839719E-4
    New Minimum: 0.022849745669531303 > 0.02244293840882202
    WOLFE (weak): th(392.2714997947665)=0.02244293840882202; dx=-2.0611725747787037E-6 delta=8.186897855932546E-4
    New Minimum: 0.02244293840882202 > 0.020866462007729545
    WOLFE (weak): th(1176.8144993842996)=0.020866462007729545; dx=-1.9576674427017056E-6 delta=0.00239516618668573
    New Minimum: 0.020866462007729545 > 0.014777220509372854
    END: th(4707.257997537198)=0.014777220509372854; dx=-1.4918943483552144E-6 delta=0.008484407685042421
    Iteration 65 complete. Error: 0.014777220509372854 Total: 239617316463706.7200; Orientation: 0.0016; Line Search: 0.0225
    LBFGS Accumulation History: 1 points
    th(0)=0.014777220509372854;dx=-1.2750831313892006E-6
    MAX ALPHA: th(0)=0.014777220509372854;th'(0)=-1.2750831313892006E-6;
    Iteration 66 failed, aborting. Error: 0.014777220509372854 Total: 239617327928387.7000; Orientation: 0.0014; Line Search: 0.0078
    
```

Returns: 

```
    0.014777220509372854
```



This training run resulted in the following regressed input:

Code from [LearningTester.java:154](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L154) executed in 0.01 seconds: 
```java
    return Arrays.stream(input_lbgfs).map(x -> x.prettyPrint()).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [
    	[ [ -0.7992595594387701, 0.3199254753156193, -1.4442996111056345 ], [ 1.348592048070863, 0.6085051487657648, -1.488082426835074 ], [ -1.4004303101835285, -0.8752691735004597, -0.96412142777882 ], [ 1.3409180876635112, -2.348720278922617, 1.202208890874427 ], [ -0.5790079955322541, -1.5836249054022946, 0.9489715597720575 ], [ 1.0924630459195892, 0.7527747290713223, -1.5225796846992101 ], [ -0.38483156292578025, 0.4575171299795029, -1.8284058382153177 ], [ -0.7691707943988968, 0.9447575109570805, -1.7828686609299356 ], ... ],
    	[ [ 1.309803036041833, -0.8564362418986454, 0.2052515340457793 ], [ 0.9055434374450106, 0.3696507645736873, 1.6552796061520196 ], [ -0.5381005535145025, 1.13875009251208, -1.0219443546143288 ], [ -2.3104841690636815, -1.2682320985874092, 1.6429979693708092 ], [ 1.3659604686331746, 0.5989770876205922, -1.1199602807896785 ], [ -0.5060736929608673, -1.2078297786563903, 0.22653381339028408 ], [ -1.51090954508813, 1.513297618252228, -0.7759330843594823 ], [ 0.6767155914001772, 0.09260393257
```
...[skipping 2242 bytes](etc/163.txt)...
```
    7980986 ], [ -0.7075058745069075, -2.018040871319945, 0.4907944359737154 ], [ -1.0339426652453507, -1.1068689270536807, 0.008950718404683537 ], [ -0.48031543985463704, -0.028423142151121128, -1.9623017152780688 ], [ 0.7863221439395387, 2.0417209428481446, 0.10830459220062852 ], [ -1.6133371432635406, -0.989904333789897, 1.3867733591743765 ], [ 1.2304661299438984, -1.7901904884892015, -1.1441165525860348 ], [ 0.020160324750053325, 1.2529200051033627, -0.15499156942495168 ], ... ],
    	[ [ -0.4956288883280174, -1.060981602120023E-5, 1.0323745086401845 ], [ 0.8274086310421988, 1.8452142922043975, 0.5571281915698968 ], [ 1.2163002695020388, -0.7495060105597925, 1.6191527182895509 ], [ -1.2671383914148369, -0.9935881902841851, 0.7931761219057892 ], [ 0.678181711198271, 1.7432517847270146, 0.10151516077333178 ], [ 1.112610230983036, 0.12861466746151679, 0.1637953881676036 ], [ -0.020740161783208373, 1.0071916698360743, -0.8468099479615747 ], [ 1.4033864070328568, 1.581034204627766, -0.4708998549495558 ], ... ],
    	...
    ]
```



Code from [LearningTester.java:95](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L95) executed in 0.00 seconds: 
```java
    return TestUtil.compare(runs);
```

Returns: 

![Result](etc/test.92.png)



Code from [LearningTester.java:98](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L98) executed in 0.00 seconds: 
```java
    return TestUtil.compareTime(runs);
```

Returns: 

![Result](etc/test.93.png)



### Model Learning
In this test, attempt to train a network to emulate a randomized network given an example input/output. The target state is:

Code from [LearningTester.java:176](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L176) executed in 0.00 seconds: 
```java
    return network_target.state().stream().map(Arrays::toString).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [0.244, 0.704, 1.408, -0.524, 0.9, 1.564, 1.52, 0.808, 0.212, 1.236, -0.96, 0.924, -1.192, 0.64, -0.704, -0.268, -0.384, 0.524, -0.968, 0.52, -0.892, 1.384, -1.864, -1.868, -0.424, 1.048, 0.928, 1.872, 1.056, 1.22, -1.124, -2.0, -0.06, 0.784, 1.156, 1.484, 1.056, -1.124, -1.648, 1.496, 1.676, 0.448, -1.248, -0.768, -1.164, -0.816, -1.868, 0.28, -0.668, 0.788, -1.98, -1.788, 1.78, 0.936, -1.356, -1.204, 0.096, -1.508, 1.296, -0.204, 0.444, 0.44, -0.532, -1.9, 0.768, 0.992, -1.292, 0.372, 1.292, -0.556, 0.944, 1.832, 1.176, 0.544, 0.672, 1.82, 0.032, 1.36, -1.456, -1.508, -0.44]
```



First, we use a conjugate gradient descent method, which converges the fastest for purely linear functions.

Code from [LearningTester.java:225](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L225) executed in 0.46 seconds: 
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
    F(0.0) = LineSearchPoint{point=PointSample{avg=81.68585590754859}, derivative=-143.12209348058468}
    New Minimum: 81.68585590754859 > 81.68585589323601
    F(1.0E-10) = LineSearchPoint{point=PointSample{avg=81.68585589323601}, derivative=-143.1220934680249}, delta = -1.431257601325342E-8
    New Minimum: 81.68585589323601 > 81.68585580736341
    F(7.000000000000001E-10) = LineSearchPoint{point=PointSample{avg=81.68585580736341}, derivative=-143.12209339266553}, delta = -1.0018517571097618E-7
    New Minimum: 81.68585580736341 > 81.68585520625052
    F(4.900000000000001E-9) = LineSearchPoint{point=PointSample{avg=81.68585520625052}, derivative=-143.12209286515}, delta = -7.012980631770915E-7
    New Minimum: 81.68585520625052 > 81.68585099846109
    F(3.430000000000001E-8) = LineSearchPoint{point=PointSample{avg=81.68585099846109}, derivative=-143.12208917254125}, delta = -4.90908749384289E-6
    New Minimum: 81.68585099846109 > 81.68582154393746
    F(2.4010000000000004E-7) = LineSearchPoint{poin
```
...[skipping 11871 bytes](etc/164.txt)...
```
    .4313350482999267E-32}, derivative=1.9908768072723285E-33}, delta = -1.4325311346014925E-30
    2.4313350482999267E-32 <= 1.4568444850844917E-30
    Converged to right
    Iteration 15 complete. Error: 2.4313350482999267E-32 Total: 239617895574847.1200; Orientation: 0.0000; Line Search: 0.0111
    Zero gradient: 3.651170608440084E-17
    F(0.0) = LineSearchPoint{point=PointSample{avg=2.4313350482999267E-32}, derivative=-1.3331046811936732E-33}
    New Minimum: 2.4313350482999267E-32 > 1.4934944742074886E-33
    F(1.1421248445282648) = LineSearchPoint{point=PointSample{avg=1.4934944742074886E-33}, derivative=1.5646933570203743E-34}, delta = -2.281985600879178E-32
    1.4934944742074886E-33 <= 2.4313350482999267E-32
    New Minimum: 1.4934944742074886E-33 > 0.0
    F(1.0221526150954778) = LineSearchPoint{point=PointSample{avg=0.0}, derivative=0.0}, delta = -2.4313350482999267E-32
    Right bracket at 1.0221526150954778
    Converged to right
    Iteration 16 complete. Error: 0.0 Total: 239617923084666.1000; Orientation: 0.0001; Line Search: 0.0221
    
```

Returns: 

```
    0.0
```



Training Converged

Next, we run the same optimization using L-BFGS, which is nearly ideal for purely second-order or quadratic functions.

Code from [LearningTester.java:249](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L249) executed in 0.48 seconds: 
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
    th(0)=100.93095284361057;dx=-176.25465145462366
    New Minimum: 100.93095284361057 > 79.02708852990305
    WOLF (strong): th(2.154434690031884)=79.02708852990305; dx=155.9209050523294 delta=21.903864313707516
    New Minimum: 79.02708852990305 > 0.5227029218536318
    END: th(1.077217345015942)=0.5227029218536318; dx=-10.166873201147155 delta=100.40824992175693
    Iteration 1 complete. Error: 0.5227029218536318 Total: 239617954972323.0600; Orientation: 0.0001; Line Search: 0.0160
    LBFGS Accumulation History: 1 points
    th(0)=0.5227029218536318;dx=-0.8782351328139563
    New Minimum: 0.5227029218536318 > 0.4730194241321537
    WOLF (strong): th(2.3207944168063896)=0.4730194241321537; dx=0.8354191924086477 delta=0.049683497721478065
    New Minimum: 0.4730194241321537 > 7.312492037977565E-4
    END: th(1.1603972084031948)=7.312492037977565E-4; dx=-0.02140797020265436 delta=0.521971672649834
    Iteration 2 complete. Error: 7.312492037977565E-4 Total: 23961797618
```
...[skipping 9041 bytes](etc/165.txt)...
```
    9 complete. Error: 7.288584570700309E-31 Total: 239618362024055.6600; Orientation: 0.0001; Line Search: 0.0178
    LBFGS Accumulation History: 1 points
    th(0)=7.288584570700309E-31;dx=-2.3254797372697684E-31
    Armijo: th(2.6231490713686236)=7.440924429435072E-31; dx=2.364567639865003E-31 delta=-1.523398587347631E-32
    New Minimum: 7.288584570700309E-31 > 1.013028879121316E-32
    END: th(1.3115745356843118)=1.013028879121316E-32; dx=-9.95355675686945E-34 delta=7.1872816827881775E-31
    Iteration 20 complete. Error: 1.013028879121316E-32 Total: 239618383129555.6200; Orientation: 0.0001; Line Search: 0.0159
    LBFGS Accumulation History: 1 points
    th(0)=1.013028879121316E-32;dx=-2.4465834458884057E-34
    Armijo: th(2.8257016782407423)=1.7811821855802763E-32; dx=3.984618725189144E-34 delta=-7.681533064589604E-33
    New Minimum: 1.013028879121316E-32 > 0.0
    END: th(1.4128508391203711)=0.0; dx=0.0 delta=1.013028879121316E-32
    Iteration 21 complete. Error: 0.0 Total: 239618405160379.6200; Orientation: 0.0001; Line Search: 0.0163
    
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

![Result](etc/test.94.png)



Code from [LearningTester.java:98](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L98) executed in 0.00 seconds: 
```java
    return TestUtil.compareTime(runs);
```

Returns: 

![Result](etc/test.95.png)



### Performance
Now we execute larger-scale runs to benchmark performance:

Code from [PerformanceTester.java:66](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/PerformanceTester.java#L66) executed in 0.88 seconds: 
```java
    test(component, inputPrototype);
```
Logging: 
```
    100 batches
    Input Dimensions:
    	[100, 100, 3]
    Performance:
    	Evaluation performance: 0.036773s +- 0.005126s [0.030257s - 0.043658s]
    	Learning performance: 0.106535s +- 0.020763s [0.085621s - 0.141705s]
    
```

