# SimpleConvolutionLayer
## Image
### Batch Execution
Code from [BatchingTester.java:66](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/BatchingTester.java#L66) executed in 0.02 seconds: 
```java
    return test(reference, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=1.1513e-17 +- 1.1408e-16 [0.0000e+00 - 1.7764e-15] (540#), relativeTol=6.8310e-19 +- 6.5614e-18 [0.0000e+00 - 8.7661e-17] (540#)}
```



Code from [SingleDerivativeTester.java:77](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/SingleDerivativeTester.java#L77) executed in 0.17 seconds: 
```java
    return test(component, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ 0.708, -1.272, 0.032 ], [ -1.252, -1.136, -1.876 ], [ 1.2, 1.768, -1.824 ] ],
    	[ [ 0.444, -1.324, -1.74 ], [ -1.508, -1.9, -0.156 ], [ -0.624, -0.108, -0.8 ] ],
    	[ [ 0.872, -0.708, -0.956 ], [ -0.236, -0.16, 1.256 ], [ -0.092, 0.784, 0.468 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.19108328659760632, negative=18, min=0.468, max=0.468, mean=-0.3755555555555556, count=27.0, positive=9, stdDev=1.0396941620579025, zeros=0}
    Output: [
    	[ [ -10.412608, -11.306912, -0.8441919999999997 ], [ -3.86248, -17.348128, 4.700944 ], [ 4.556192, 1.0011999999999996, -1.6935519999999995 ] ],
    	[ [ -0.5637279999999992, -5.6852, -7.174239999999999 ], [ 4.424079999999999, 2.642944, -1.4792960000000004 ], [ 6.487503999999999, 5.5964160000000005, -1.2567359999999994 ] ],
    	[ [ 3.268688, -0.8369120000000002, 0.48347199999999985 ], [ 4.752527999999998, -0.6215520000000001, -3.4131999999999993 ], [ 2.754495999999999, -1.3569599999999997, -1.2378879999999997 ] ]
    ]
    Outputs Statistics: {meanExponent=0.4234310398737198, negati
```
...[skipping 6762 bytes](etc/148.txt)...
```
    633220256E-11, -2.7756130727141226E-12, -1.570410468332284E-12, -2.695399459184955E-12, -1.1750600492632657E-12, -4.565903211073419E-12, -1.0202949596305189E-12, -1.7996715229173788E-13, ... ], [ 0.0, 3.750333377183779E-13, -2.7756130727141226E-12, 0.0, -2.695399459184955E-12, -1.1750600492632657E-12, 0.0, -1.0202949596305189E-12, ... ], [ 0.0, 0.0, 0.0, -2.7756130727141226E-12, -6.01130256683291E-12, 0.0, -1.1750600492632657E-12, 4.3158809859278335E-12, ... ], [ 0.0, 0.0, 0.0, 3.750333377183779E-13, 1.6652790257865036E-12, -6.01130256683291E-12, -2.695399459184955E-12, -1.1750600492632657E-12, ... ], ... ]
    Error Statistics: {meanExponent=-11.864764807591785, negative=236, min=-4.499178807293447E-14, max=-4.499178807293447E-14, mean=1.1954634812935992E-13, count=2187.0, positive=205, stdDev=1.754430528383549E-12, zeros=1746}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 7.9352e-13 +- 2.0425e-12 [0.0000e+00 - 2.3240e-11] (2916#)
    relativeTol: 4.2933e-12 +- 1.3976e-11 [8.4704e-15 - 2.9193e-10] (882#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=7.9352e-13 +- 2.0425e-12 [0.0000e+00 - 2.3240e-11] (2916#), relativeTol=4.2933e-12 +- 1.3976e-11 [8.4704e-15 - 2.9193e-10] (882#)}
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
      "id": "088bc00a-6163-4acb-a1d1-27fcc8221ab2",
      "isFrozen": false,
      "name": "ConvolutionLayer/088bc00a-6163-4acb-a1d1-27fcc8221ab2",
      "filter": [
        [
          [
            -1.428,
            -1.468,
            1.248
          ],
          [
            -1.048,
            0.104,
            1.684
          ],
          [
            -1.392,
            -1.284,
            1.732
          ]
        ],
        [
          [
            1.716,
            0.94,
            0.776
          ],
          [
            -0.16,
            0.828,
            1.284
          ],
          [
            -0.288,
            -0.268,
            0.928
          ]
        ],
        [
          [
            -0.748,
            -0.192,
            0.072
          ],
          [
            -1.872,
            -0.448,
            0.88
          ],
          [
            0.188,
            -0.044,
            1.528
          ]
        ],
        [
          [
            1.936,
            1.712,
            -0.16
          ],
          [
            -0.476,
            1.66,
            -1.548
          ],
          [
            1.52,
            -1.888,
            -1.92
          ]
        ],
        [
       
```
...[skipping 7 bytes](etc/149.txt)...
```
          1.444,
            -1.804,
            1.452
          ],
          [
            0.976,
            1.4,
            0.088
          ],
          [
            0.66,
            1.4,
            -0.016
          ]
        ],
        [
          [
            -0.588,
            0.492,
            0.984
          ],
          [
            0.016,
            -0.172,
            -0.388
          ],
          [
            1.132,
            -1.524,
            -0.18
          ]
        ],
        [
          [
            0.548,
            1.96,
            -1.072
          ],
          [
            1.8,
            -0.416,
            -0.02
          ],
          [
            0.948,
            0.716,
            0.1
          ]
        ],
        [
          [
            1.012,
            0.588,
            0.296
          ],
          [
            1.76,
            1.036,
            -1.804
          ],
          [
            1.548,
            -1.68,
            -0.924
          ]
        ],
        [
          [
            -1.46,
            1.248,
            1.888
          ],
          [
            -0.172,
            -0.184,
            -0.236
          ],
          [
            -0.112,
            0.992,
            1.78
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

Code from [EquivalencyTester.java:64](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/EquivalencyTester.java#L64) executed in 0.01 seconds: 
```java
    return test(subject, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ 1.692, -1.384, 1.62 ], [ -0.444, 1.288, 1.272 ], [ 1.612, 0.168, 0.264 ] ],
    	[ [ -1.54, -0.668, 0.804 ], [ -0.424, -1.648, 1.864 ], [ -0.98, -1.256, -1.708 ] ],
    	[ [ 0.352, 1.344, -1.76 ], [ -1.272, 1.464, 0.636 ], [ -1.8, 1.42, 0.376 ] ]
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
      "id": "fa6a8d00-531e-4534-ae92-cf1a97cd558d",
      "isFrozen": false,
      "name": "SimpleConvolutionLayer/fa6a8d00-531e-4534-ae92-cf1a97cd558d",
      "filter": [
        [
          [
            -1.428,
            -1.468,
            1.248
          ],
          [
            -1.048,
            0.104,
            1.684
          ],
          [
            -1.392,
            -1.284,
            1.732
          ]
        ],
        [
          [
            1.936,
            1.712,
            -0.16
          ],
          [
            -0.476,
            1.66,
            -1.548
          ],
          [
            1.52,
            -1.888,
            -1.92
          ]
        ],
        [
          [
            0.548,
            1.96,
            -1.072
          ],
          [
            1.8,
            -0.416,
            -0.02
          ],
          [
            0.948,
            0.716,
            0.1
          ]
        ],
        [
          [
            1.716,
            0.94,
            0.776
          ],
          [
            -0.16,
            0.828,
            1.284
          ],
          [
            -0.288,
            -0.268,
            0.928
          ]
        ],
        [
```
...[skipping 9 bytes](etc/150.txt)...
```
           1.444,
            -1.804,
            1.452
          ],
          [
            0.976,
            1.4,
            0.088
          ],
          [
            0.66,
            1.4,
            -0.016
          ]
        ],
        [
          [
            1.012,
            0.588,
            0.296
          ],
          [
            1.76,
            1.036,
            -1.804
          ],
          [
            1.548,
            -1.68,
            -0.924
          ]
        ],
        [
          [
            -0.748,
            -0.192,
            0.072
          ],
          [
            -1.872,
            -0.448,
            0.88
          ],
          [
            0.188,
            -0.044,
            1.528
          ]
        ],
        [
          [
            -0.588,
            0.492,
            0.984
          ],
          [
            0.016,
            -0.172,
            -0.388
          ],
          [
            1.132,
            -1.524,
            -0.18
          ]
        ],
        [
          [
            -1.46,
            1.248,
            1.888
          ],
          [
            -0.172,
            -0.184,
            -0.236
          ],
          [
            -0.112,
            0.992,
            1.78
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
    	[ [ -0.096, 0.956, -0.56 ], [ -1.232, 0.924, -0.784 ], [ 1.78, 1.212, -0.164 ] ],
    	[ [ -0.056, -0.088, -0.832 ], [ -1.156, 1.456, 0.58 ], [ -1.824, 1.572, 1.488 ] ],
    	[ [ -1.424, -1.072, 1.508 ], [ -0.312, -0.572, 1.86 ], [ 0.176, 1.856, -0.36 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 7.0541599999999995, -3.4427360000000005, -0.897376 ], [ 5.9710719999999995, 3.099296, -1.0829599999999997 ], [ 9.755087999999999, 12.578351999999999, 1.4422239999999997 ] ],
    	[ [ 8.119856000000002, -0.9224799999999996, 1.1158399999999988 ], [ 15.513551999999997, 10.135536000000002, -1.2019520000000004 ], [ -2.2979679999999996, 9.601584, -4.781568 ] ],
    	[ [ -1.669439999999999, 3.5247200000000003, 5.270352 ], [ -2.956576000000001, -6.555968, 4.473408000000001 ], [ -4.213216, -7.990832, -1.1635360000000001 ] ]
    ]
    --------------------
    Derivative: 
    [
    	[ [ 6.9239999999999995, -3.0880000000000005, -0.6399999999999999 ], [ 8.3, -0.4120000000000008, 4.268000000000001 ], [ 5.708, 3.7159999999999993, 3.2840000000000003 ] ],
    	[ [ 2.3520000000000003, 0.7400000000000002, 5.132 ], [ 3.2680000000000007, 6.207999999999998, 10.139999999999999 ], [ 2.168, 7.023999999999999, 6.772 ] ],
    	[ [ -5.684, 4.703999999999999, 6.236000000000001 ], [ -6.864, 7.895999999999999, 10.132000000000001 ], [ -3.776, 6.595999999999999, 7.720000000000001 ] ]
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
    	[ [ -0.584, -1.528, 1.652 ], [ -0.816, 0.776, 1.532 ], [ -1.548, -1.988, 1.34 ], [ 0.368, 1.496, 1.072 ], [ 0.104, -1.168, -1.036 ], [ 0.436, 1.952, -1.616 ], [ -0.132, -1.96, -0.068 ], [ -1.648, 1.68, 0.964 ], ... ],
    	[ [ 1.188, 0.888, -1.932 ], [ -1.856, 1.0, 1.396 ], [ 0.616, 1.08, -1.628 ], [ -0.712, 1.312, 1.984 ], [ -1.9, 0.276, -0.244 ], [ -1.236, -0.748, -0.692 ], [ 1.316, 1.244, 1.972 ], [ 1.58, 1.612, 0.236 ], ... ],
    	[ [ 0.836, -0.62, -1.624 ], [ 1.34, -1.992, 0.76 ], [ -1.5, 1.296, 0.58 ], [ 1.248, -0.388, -0.608 ], [ -1.896, 0.936, -1.516 ], [ -0.472, 1.212, 0.528 ], [ -1.74, -1.912, -1.928 ], [ -1.376, -0.116, 1.812 ], ... ],
    	[ [ -0.892, -1.468, -1.44 ], [ 0.844, -1.072, -1.908 ], [ -0.34, 1.804, -1.24 ], [ 0.332, -1.204, 0.036 ], [ 0.996, -1.224, -0.388 ], [ 0.216, 1.72, -0.22 ], [ 1.476, 1.372, 1.648 ], [ 1.572, 0.236, -1.92 ], ... ],
    	[ [ 0.252, -1.512, 0.332 ], [ -0.868, 1.556, 1.672 ], [ 0.632, 0.808, -0.216 ], [ 0.204, 1.596, 1.2 ], [ -0.2, -1.744, -1.696 ], [ 0.148, 1.016, -0.364 ], [ 0.628, -1.824, 1.972 ], [ 1.772, -0.068, 0.792 ], ... ],
    	[ [ -0.376, -1.404, -0.204 ], [ 1.008, -1.664, -1.956 ], [ 1.324, -1.416, 0.676 ], [ 0.684, 0.668, -1.224 ], [ -1.588, -0.236, -1.824 ], [ 1.168, 1.12, -1.564 ], [ 1.772, -1.86, -0.484 ], [ -0.444, -1.528, 0.476 ], ... ],
    	[ [ -1.832, 0.448, 1.248 ], [ -1.788, 0.404, -0.236 ], [ 1.604, -1.836, 0.136 ], [ -0.644, -1.488, 0.812 ], [ -1.924, -0.024, 0.636 ], [ -1.724, -1.948, -1.456 ], [ -0.568, -0.472, -0.02 ], [ -0.996, 1.212, 1.948 ], ... ],
    	[ [ 0.792, 1.676, 0.548 ], [ 0.104, -0.088, 1.324 ], [ 1.748, 0.132, -1.4 ], [ -0.544, 0.64, -1.032 ], [ 1.864, 0.728, 1.812 ], [ 1.788, 1.108, -1.972 ], [ 0.076, 1.0, 0.508 ], [ -1.156, -1.72, 0.432 ], ... ],
    	...
    ]
```



First, we use a conjugate gradient descent method, which converges the fastest for purely linear functions.

Code from [LearningTester.java:225](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L225) executed in 7.17 seconds: 
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
    F(0.0) = LineSearchPoint{point=PointSample{avg=94.92220973249132}, derivative=-0.9180838064175152}
    New Minimum: 94.92220973249132 > 94.92220973239881
    F(1.0E-10) = LineSearchPoint{point=PointSample{avg=94.92220973239881}, derivative=-0.9180838064169744}, delta = -9.251266419596504E-11
    New Minimum: 94.92220973239881 > 94.92220973184776
    F(7.000000000000001E-10) = LineSearchPoint{point=PointSample{avg=94.92220973184776}, derivative=-0.9180838064137296}, delta = -6.435669774873531E-10
    New Minimum: 94.92220973184776 > 94.92220972799184
    F(4.900000000000001E-9) = LineSearchPoint{point=PointSample{avg=94.92220972799184}, derivative=-0.9180838063910157}, delta = -4.499483452491404E-9
    New Minimum: 94.92220972799184 > 94.9222097009998
    F(3.430000000000001E-8) = LineSearchPoint{point=PointSample{avg=94.9222097009998}, derivative=-0.9180838062320194}, delta = -3.149152405512723E-8
    New Minimum: 94.9222097009998 > 94.92220951205817
    F(2.4010000000000004E-7) = LineSearchPoint{p
```
...[skipping 257568 bytes](etc/151.txt)...
```
    ion 249 complete. Error: 0.0033604637114207914 Total: 239603656605377.3800; Orientation: 0.0009; Line Search: 0.0198
    F(0.0) = LineSearchPoint{point=PointSample{avg=0.0033604637114207914}, derivative=-1.0156659135020908E-7}
    New Minimum: 0.0033604637114207914 > 0.003347335558972197
    F(149.68367706648667) = LineSearchPoint{point=PointSample{avg=0.003347335558972197}, derivative=-7.384535343748911E-8}, delta = -1.3128152448594384E-5
    F(1047.7857394654066) = LineSearchPoint{point=PointSample{avg=0.003355704397576421}, derivative=9.248207403882848E-8}, delta = -4.75931384437021E-6
    0.003355704397576421 <= 0.0033604637114207914
    New Minimum: 0.003347335558972197 > 0.0033326131738433394
    F(548.4192628148403) = LineSearchPoint{point=PointSample{avg=0.0033326131738433394}, derivative=2.1625259827167265E-22}, delta = -2.7850537577452008E-5
    Right bracket at 548.4192628148403
    Converged to right
    Iteration 250 complete. Error: 0.0033326131738433394 Total: 239603677019519.3400; Orientation: 0.0008; Line Search: 0.0176
    
```

Returns: 

```
    0.0033326131738433394
```



This training run resulted in the following regressed input:

Code from [LearningTester.java:144](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L144) executed in 0.01 seconds: 
```java
    return Arrays.stream(input_gd).map(x -> x.prettyPrint()).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [
    	[ [ -0.48098157573070954, -1.3881927056921992, 1.8516689060993032 ], [ -0.7576123751216332, 0.7009269713149433, 1.626388731395146 ], [ -1.6558877189474734, -1.928832573877313, 1.2424158211517418 ], [ 0.3409011975436412, 1.2205409147033714, 0.9873641956221695 ], [ 0.06039363007080456, -0.7943954569058526, -0.9372367728907955 ], [ 0.38721012331005794, 1.8237020240470756, -1.5361134450009397 ], [ -0.1306376034846644, -1.8432233922224142, -0.1553692321575172 ], [ -1.5587270278508696, 1.537241917395717, 0.9379627809800504 ], ... ],
    	[ [ 1.1074145540930418, 0.8002403812758567, -2.0604204751415356 ], [ -1.8821532301579842, 0.7702120733794956, 1.2674005422610286 ], [ 0.6717835183273947, 1.277149208529674, -1.5508350980639363 ], [ -0.7264684329843459, 1.3096491148132945, 2.016533958242991 ], [ -1.720645219547728, 0.2518130838886005, -0.2533423836029033 ], [ -1.2864636686023228, -0.7983310168369896, -0.6652329341248082 ], [ 1.1130361549208172, 1.1362715837292563, 1.839824729265202 ], [ 1.5275498812063044, 1.61491991
```
...[skipping 2236 bytes](etc/152.txt)...
```
    06174109558 ], [ -1.3637850487115988, 1.167684108265606, -0.15668608822886976 ], [ 1.544441518038757, -2.318800067640123, 0.1169195652028136 ], [ -0.8127407706036429, -1.691594175135877, 0.42918564014959343 ], [ -2.0590048633607227, -0.17015640160908926, 0.5907092508082076 ], [ -1.9328588778908586, -1.9634642664316286, -1.4885665829214447 ], [ -0.6719635584634771, -0.5018220744188695, 0.04830514884906777 ], [ -0.8282995886652529, 1.5302206879273554, 2.2309226543484697 ], ... ],
    	[ [ 0.9171523326972483, 1.7437489335998062, 0.12515532345498845 ], [ 0.19860225782858193, -0.5450821043604459, 1.0614574858966066 ], [ 1.6261625734392746, 0.6576033205816061, -1.0751245046294502 ], [ -0.5337668682959485, 0.5621096093076029, -0.8894922177222726 ], [ 2.1950314394374795, 0.9567340417858567, 1.776891257333305 ], [ 1.8187780338223773, 1.0899919757720107, -1.9060820120750683 ], [ -0.007166497966423311, 0.9192341795551027, 0.31286243283806925 ], [ -1.1288280926036647, -2.0865493569038653, 0.18251832942102708 ], ... ],
    	...
    ]
```



Next, we run the same optimization using L-BFGS, which is nearly ideal for purely second-order or quadratic functions.

Code from [LearningTester.java:249](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L249) executed in 1.16 seconds: 
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
    th(0)=94.92220973249132;dx=-0.9180838064175152
    New Minimum: 94.92220973249132 > 92.95680908711168
    WOLFE (weak): th(2.154434690031884)=92.95680908711168; dx=-0.9064325314151565 delta=1.9654006453796455
    New Minimum: 92.95680908711168 > 91.01651035278203
    WOLFE (weak): th(4.308869380063768)=91.01651035278203; dx=-0.8947812564127976 delta=3.90569937970929
    New Minimum: 91.01651035278203 > 83.50633452594293
    WOLFE (weak): th(12.926608140191302)=83.50633452594293; dx=-0.8481761564033625 delta=11.415875206548392
    New Minimum: 83.50633452594293 > 54.680721692709085
    END: th(51.70643256076521)=54.680721692709085; dx=-0.6384532063609045 delta=40.24148803978224
    Iteration 1 complete. Error: 54.680721692709085 Total: 239603746877982.3000; Orientation: 0.0017; Line Search: 0.0259
    LBFGS Accumulation History: 1 points
    th(0)=54.680721692709085;dx=-0.45257468971472603
    New Minimum: 54.680721692709085 > 19.319858268821882
    END: th(111.398132006
```
...[skipping 29228 bytes](etc/153.txt)...
```
    7
    WOLFE (weak): th(259.6218860754985)=0.022743058998293007; dx=-1.5099846758139003E-6 delta=3.9467025416310003E-4
    New Minimum: 0.022743058998293007 > 0.022353679113496294
    WOLFE (weak): th(519.243772150997)=0.022353679113496294; dx=-1.4896074670731013E-6 delta=7.840501389598138E-4
    New Minimum: 0.022353679113496294 > 0.020849063267971615
    WOLFE (weak): th(1557.731316452991)=0.020849063267971615; dx=-1.408098632109905E-6 delta=0.0022886659844844927
    New Minimum: 0.020849063267971615 > 0.01512578509762579
    END: th(6230.925265811964)=0.01512578509762579; dx=-1.0413088747755223E-6 delta=0.008011944154830317
    Iteration 60 complete. Error: 0.01512578509762579 Total: 239604843734441.2000; Orientation: 0.0014; Line Search: 0.0217
    LBFGS Accumulation History: 1 points
    th(0)=0.01512578509762579;dx=-1.8316957201380663E-6
    MAX ALPHA: th(0)=0.01512578509762579;th'(0)=-1.8316957201380663E-6;
    Iteration 61 failed, aborting. Error: 0.01512578509762579 Total: 239604854756265.1600; Orientation: 0.0015; Line Search: 0.0072
    
```

Returns: 

```
    0.01512578509762579
```



This training run resulted in the following regressed input:

Code from [LearningTester.java:154](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L154) executed in 0.01 seconds: 
```java
    return Arrays.stream(input_lbgfs).map(x -> x.prettyPrint()).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [
    	[ [ -0.41641335559882775, -1.3791418198261658, 1.9303399820191314 ], [ -0.7914093581043733, 0.7144974918405705, 1.6464072755387866 ], [ -1.6992853663216858, -1.9197198264576487, 1.209455946438401 ], [ 0.3966765433888811, 1.0974188941959233, 0.991043646615669 ], [ -0.03392901627799598, -0.6129917940475542, -0.8558970013286477 ], [ 0.2786575842614225, 1.8457886052090258, -1.5391002462575536 ], [ -0.09092894436955648, -1.8893685303744079, -0.21649041872433916 ], [ -1.5091743418593582, 1.6315342821542984, 0.983629356869199 ], ... ],
    	[ [ 1.0691656042574713, 0.7270428905517096, -2.0540326968122016 ], [ -1.8979997736844085, 0.7915849053390764, 1.2371394360020962 ], [ 0.7368207985922661, 1.3464626666857988, -1.5313014904269162 ], [ -0.7450895803639891, 1.206136219240731, 1.9799726268634559 ], [ -1.5789983706764512, 0.2820534213904687, -0.2651639216728462 ], [ -1.2258521091563153, -0.8031408334908834, -0.6185066963077994 ], [ 0.8733888913263128, 0.9713532610783346, 1.7117285809132212 ], [ 1.4625613205500616, 1.688
```
...[skipping 2223 bytes](etc/154.txt)...
```
    19813406689612 ], [ -1.178548614818009, 1.0057562655827037, -0.2604134345322299 ], [ 1.4793414317152909, -2.2984909829646085, 0.03627211750370922 ], [ -0.8085622416409218, -1.7042655971356908, 0.3840159427213382 ], [ -2.007816291815354, -0.23324385847635576, 0.5656280034086499 ], [ -2.0414325493175913, -1.979323949336448, -1.5768641581881568 ], [ -0.6341464263558104, -0.5252814888318363, -0.048260111096320495 ], [ -0.7320143885803535, 1.3585704725026801, 2.1614669914039024 ], ... ],
    	[ [ 0.7861015299470409, 1.4311631977844366, -0.05773664885428689 ], [ 0.08185058581999777, -0.5225107149340059, 1.0315875782912098 ], [ 1.7178350097925754, 1.0565811746176255, -0.742474903944385 ], [ -0.46163442791817033, 0.6851252357400216, -0.6137871953010479 ], [ 2.188064563426406, 0.8434788396353982, 1.5981085197894935 ], [ 1.8824165067079093, 1.157468516640957, -2.023011689182545 ], [ 0.02799061069126928, 0.8671995609646734, 0.3893282486772497 ], [ -1.1937983003768609, -2.1152002635243843, 0.2156967319969788 ], ... ],
    	...
    ]
```



Code from [LearningTester.java:95](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L95) executed in 0.00 seconds: 
```java
    return TestUtil.compare(runs);
```

Returns: 

![Result](etc/test.88.png)



Code from [LearningTester.java:98](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L98) executed in 0.00 seconds: 
```java
    return TestUtil.compareTime(runs);
```

Returns: 

![Result](etc/test.89.png)



### Model Learning
In this test, attempt to train a network to emulate a randomized network given an example input/output. The target state is:

Code from [LearningTester.java:176](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L176) executed in 0.00 seconds: 
```java
    return network_target.state().stream().map(Arrays::toString).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [1.036, 0.984, 0.104, -1.804, -1.392, 1.96, -1.888, 1.8, -1.46, -1.468, -0.16, 0.948, 1.716, -1.872, 0.88, -1.92, 1.732, 1.888, -1.524, -0.416, 0.188, 1.4, 0.828, 0.716, 1.012, -0.184, -0.172, -0.02, -1.804, 0.588, 0.776, -0.236, -0.588, 1.132, 1.684, 1.712, -0.172, 1.528, -1.072, -1.048, 1.248, 0.072, 0.928, -0.192, -0.18, 1.66, 1.444, -0.044, 1.78, -0.748, -0.476, 1.452, 0.548, 1.248, -0.448, 0.492, -0.016, 1.4, -0.268, -1.428, 1.548, -0.16, 0.296, 0.016, 0.1, 1.76, 0.66, -1.284, 0.088, 1.936, 0.992, 0.94, -0.112, 1.284, -0.388, -0.288, -1.68, -0.924, -1.548, 1.52, 0.976]
```



First, we use a conjugate gradient descent method, which converges the fastest for purely linear functions.

Code from [LearningTester.java:225](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L225) executed in 1.32 seconds: 
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
    F(0.0) = LineSearchPoint{point=PointSample{avg=101.60692973957917}, derivative=-177.32512899494552}
    New Minimum: 101.60692973957917 > 101.60692972184734
    F(1.0E-10) = LineSearchPoint{point=PointSample{avg=101.60692972184734}, derivative=-177.32512897941953}, delta = -1.773183555542346E-8
    New Minimum: 101.60692972184734 > 101.60692961545188
    F(7.000000000000001E-10) = LineSearchPoint{point=PointSample{avg=101.60692961545188}, derivative=-177.32512888626195}, delta = -1.2412729688549007E-7
    New Minimum: 101.60692961545188 > 101.60692887068612
    F(4.900000000000001E-9) = LineSearchPoint{point=PointSample{avg=101.60692887068612}, derivative=-177.32512823416099}, delta = -8.688930535072359E-7
    New Minimum: 101.60692887068612 > 101.6069236573277
    F(3.430000000000001E-8) = LineSearchPoint{point=PointSample{avg=101.6069236573277}, derivative=-177.3251236694539}, delta = -6.082251474026634E-6
    New Minimum: 101.6069236573277 > 101.60688716381969
    F(2.4010000000000004E-7) = Line
```
...[skipping 20979 bytes](etc/155.txt)...
```
    924E-33
    Left bracket at 0.9203367457644934
    F(0.9395873838704323) = LineSearchPoint{point=PointSample{avg=3.537137256795672E-33}, derivative=-1.1937364852172646E-35}, delta = -8.3105674634924E-33
    Left bracket at 0.9395873838704323
    F(0.9518781500450517) = LineSearchPoint{point=PointSample{avg=3.537137256795672E-33}, derivative=-1.1937364852172646E-35}, delta = -8.3105674634924E-33
    Left bracket at 0.9518781500450517
    Converged to left
    Iteration 18 complete. Error: 2.393288944225205E-33 Total: 239606282449836.8400; Orientation: 0.0000; Line Search: 0.1181
    Zero gradient: 7.188802026067926E-18
    F(0.0) = LineSearchPoint{point=PointSample{avg=3.537137256795672E-33}, derivative=-5.167887456999832E-35}
    New Minimum: 3.537137256795672E-33 > 0.0
    F(0.9518781500450517) = LineSearchPoint{point=PointSample{avg=0.0}, derivative=0.0}, delta = -3.537137256795672E-33
    0.0 <= 3.537137256795672E-33
    Converged to right
    Iteration 19 complete. Error: 0.0 Total: 239606307402259.7200; Orientation: 0.0001; Line Search: 0.0167
    
```

Returns: 

```
    0.0
```



Training Converged

Next, we run the same optimization using L-BFGS, which is nearly ideal for purely second-order or quadratic functions.

Code from [LearningTester.java:249](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L249) executed in 1.16 seconds: 
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
    th(0)=92.0394270429573;dx=-161.97318465546664
    New Minimum: 92.0394270429573 > 74.90447232934507
    WOLF (strong): th(2.154434690031884)=74.90447232934507; dx=146.06650176283483 delta=17.13495471361223
    New Minimum: 74.90447232934507 > 0.5155263953845486
    END: th(1.077217345015942)=0.5155263953845486; dx=-7.953341446315966 delta=91.52390064757275
    Iteration 1 complete. Error: 0.5155263953845486 Total: 239606352749445.7000; Orientation: 0.0001; Line Search: 0.0250
    LBFGS Accumulation History: 1 points
    th(0)=0.5155263953845486;dx=-0.8635983389353326
    New Minimum: 0.5155263953845486 > 0.4629656728018127
    WOLF (strong): th(2.3207944168063896)=0.4629656728018127; dx=0.8183028813091753 delta=0.052560722582735864
    New Minimum: 0.4629656728018127 > 0.0013276638977655728
    END: th(1.1603972084031948)=0.0013276638977655728; dx=-0.022647728813078364 delta=0.514198731486783
    Iteration 2 complete. Error: 0.0013276638977655728 Total: 2396063859394
```
...[skipping 14041 bytes](etc/156.txt)...
```
    8)=1.009084574595211E-33; dx=-9.084314949012347E-36 delta=0.0
    Armijo: th(1.9458997532639197E-8)=1.009084574595211E-33; dx=-9.084314949012347E-36 delta=0.0
    Armijo: th(1.3826129825822586E-8)=1.009084574595211E-33; dx=-9.084314949012347E-36 delta=0.0
    Armijo: th(1.100969597241428E-8)=1.009084574595211E-33; dx=-9.084314949012347E-36 delta=0.0
    Armijo: th(9.601479045710128E-9)=1.009084574595211E-33; dx=-9.084314949012347E-36 delta=0.0
    WOLFE (weak): th(8.897370582358053E-9)=1.009084574595211E-33; dx=-9.084314949012347E-36 delta=0.0
    WOLFE (weak): th(9.24942481403409E-9)=1.009084574595211E-33; dx=-9.084314949012347E-36 delta=0.0
    Armijo: th(9.42545192987211E-9)=1.009084574595211E-33; dx=-9.084314949012347E-36 delta=0.0
    WOLFE (weak): th(9.3374383719531E-9)=1.009084574595211E-33; dx=-9.084314949012347E-36 delta=0.0
    mu /= nu: th(0)=1.009084574595211E-33;th'(0)=-9.084314949012347E-36;
    Iteration 27 failed, aborting. Error: 1.009084574595211E-33 Total: 239607465925192.7800; Orientation: 0.0001; Line Search: 0.2228
    
```

Returns: 

```
    1.009084574595211E-33
```



Training Converged

Code from [LearningTester.java:95](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L95) executed in 0.00 seconds: 
```java
    return TestUtil.compare(runs);
```

Returns: 

![Result](etc/test.90.png)



Code from [LearningTester.java:98](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L98) executed in 0.00 seconds: 
```java
    return TestUtil.compareTime(runs);
```

Returns: 

![Result](etc/test.91.png)



### Performance
Now we execute larger-scale runs to benchmark performance:

Code from [PerformanceTester.java:66](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/PerformanceTester.java#L66) executed in 0.89 seconds: 
```java
    test(component, inputPrototype);
```
Logging: 
```
    100 batches
    Input Dimensions:
    	[100, 100, 3]
    Performance:
    	Evaluation performance: 0.031172s +- 0.007379s [0.022107s - 0.042079s]
    	Learning performance: 0.109014s +- 0.004116s [0.103274s - 0.114092s]
    
```

