# ConvolutionLayer
## Float
### Batch Execution
Code from [BatchingTester.java:66](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/BatchingTester.java#L66) executed in 0.02 seconds: 
```java
    return test(reference, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (1000#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (1000#)}
```



Code from [SingleDerivativeTester.java:77](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/SingleDerivativeTester.java#L77) executed in 0.18 seconds: 
```java
    return test(component, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ -1.048, -1.704 ], [ 1.888, -1.568 ], [ -1.26, -0.152 ], [ -1.496, 0.26 ], [ -0.904, -0.496 ] ],
    	[ [ 1.044, 0.792 ], [ -1.832, 0.708 ], [ -1.672, 1.712 ], [ -0.296, -1.608 ], [ -0.248, -0.644 ] ],
    	[ [ 0.492, -0.804 ], [ 0.092, 1.064 ], [ -0.848, -0.776 ], [ 1.656, -1.408 ], [ -1.36, -0.444 ] ],
    	[ [ -0.788, 1.3 ], [ 0.08, 0.24 ], [ 0.648, 1.16 ], [ -0.712, -0.68 ], [ 0.7, -0.732 ] ],
    	[ [ 1.888, 0.692 ], [ 1.3, 1.924 ], [ -0.016, -1.664 ], [ 1.028, 1.5 ], [ 0.724, -0.524 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.131182664993659, negative=27, min=-0.524, max=-0.524, mean=-0.055840000000000015, count=50.0, positive=23, stdDev=1.1123561904354198, zeros=0}
    Output: [
    	[ [ 0.053215999999999986, -0.233344 ], [ -1.17184, -0.340224 ], [ 0.504016, 0.030511999999999997 ], [ 0.700608, 0.10846399999999999 ], [ 0.27183999999999997, -0.041568 ] ],
    	[ [ -0.263088, 0.08395200000000001 ], [ 0.948352, 0.19672 ], [ 1.1128, 0.354336 ], [ -0.24636800000000006, -0.250688 ], [ -0.04326400000000001, -0.0947040000000
```
...[skipping 4755 bytes](etc/55.txt)...
```
    39999999999412, mean=-0.027920000000002186, count=200.0, positive=46, stdDev=0.7870499816401733, zeros=100}
    Gradient Error: [ [ 3.9523939676655573E-13, 4.4853010194856324E-14, 2.7000623958883807E-13, 6.552536291337674E-13, -1.220357148667972E-12, -1.220357148667972E-12, 6.104006189389111E-13, 1.4752088439706768E-13, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 1.9695356456850277E-13, 1.4876988529977098E-14, 1.9517720772910252E-13, 7.4495964952348E-13, -8.504308368628699E-14, 1.5405454689698672E-12, 3.750333377183779E-13, 6.483702463810914E-14, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ] ]
    Error Statistics: {meanExponent=-12.983486337286056, negative=44, min=5.88418203051333E-14, max=5.88418203051333E-14, mean=-2.1910251390977464E-15, count=200.0, positive=56, stdDev=2.7996015797575073E-13, zeros=100}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 1.2447e-14 +- 8.7173e-14 [0.0000e+00 - 1.8405e-12] (2700#)
    relativeTol: 3.5650e-13 +- 1.1411e-12 [6.2513e-17 - 1.4378e-11] (200#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=1.2447e-14 +- 8.7173e-14 [0.0000e+00 - 1.8405e-12] (2700#), relativeTol=3.5650e-13 +- 1.1411e-12 [6.2513e-17 - 1.4378e-11] (200#)}
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
      "id": "536597d1-d663-4c59-93c3-a103ce339ad0",
      "isFrozen": false,
      "name": "ConvolutionLayer/536597d1-d663-4c59-93c3-a103ce339ad0",
      "filter": [
        [
          [
            -0.428
          ]
        ],
        [
          [
            -0.044
          ]
        ],
        [
          [
            0.232
          ]
        ],
        [
          [
            0.164
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
    	[ [ -1.056, 1.764 ], [ 0.92, 1.524 ], [ -0.088, -1.296 ], [ 1.316, 0.964 ], [ -0.444, -0.008 ] ],
    	[ [ -1.76, 1.924 ], [ 1.052, 0.244 ], [ -1.552, 1.244 ], [ -0.52, -0.06 ], [ 0.104, 0.52 ] ],
    	[ [ 0.2, -0.084 ], [ 1.876, -1.76 ], [ -0.748, -0.052 ], [ -0.972, 0.976 ], [ 0.2, -1.964 ] ],
    	[ [ 1.648, -0.892 ], [ -1.944, 1.536 ], [ -1.504, 1.024 ], [ 0.42, -1.912 ], [ -0.328, -0.256 ] ],
    	[ [ -0.8, -1.708 ], [ -0.696, -0.376 ], [ -0.752, -0.756 ], [ 0.732, 0.4 ], [ -0.884, 1.3 ] ]
    ]
    Error: [
    	[ [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ] ],
    	[ [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ] ],
    	[ [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ] ],
    	[ [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ] ],
    	[ [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ] ]
    ]
    Accuracy:
    absoluteTol: 0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (50#)
    relativeTol: 0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (50#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (50#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (50#)}
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
      "id": "185b53a6-e0f3-45ea-9b3e-ea434545324b",
      "isFrozen": false,
      "name": "ConvolutionLayer/185b53a6-e0f3-45ea-9b3e-ea434545324b",
      "filter": [
        [
          [
            -0.428
          ]
        ],
        [
          [
            -0.044
          ]
        ],
        [
          [
            0.232
          ]
        ],
        [
          [
            0.164
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
    	[ [ -0.124, -1.072 ], [ 1.456, 0.14 ], [ 0.864, 0.784 ], [ -1.64, -0.116 ], [ -1.544, 1.6 ] ],
    	[ [ 0.004, -1.848 ], [ 1.148, 0.148 ], [ -1.648, 0.812 ], [ -1.22, 0.604 ], [ 1.308, -1.26 ] ],
    	[ [ 1.68, -1.988 ], [ 1.884, -0.956 ], [ 0.508, 0.508 ], [ -0.424, -1.9 ], [ 1.912, -0.472 ] ],
    	[ [ -0.324, -1.592 ], [ -0.048, 1.784 ], [ -0.436, 1.448 ], [ 0.956, -0.896 ], [ -1.708, 1.356 ] ],
    	[ [ -1.328, 1.248 ], [ 0.488, -1.036 ], [ 0.624, 0.688 ], [ 0.592, -0.456 ], [ 1.764, 1.98 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ -0.19563200000000003, -0.17035200000000003 ], [ -0.590688, -0.041103999999999995 ], [ -0.187904, 0.09056000000000002 ], [ 0.6750079999999999, 0.05313599999999999 ], [ 1.032032, 0.330336 ] ],
    	[ [ -0.43044800000000005, -0.303248 ], [ -0.45700799999999997, -0.026239999999999996 ], [ 0.893728, 0.20568 ], [ 0.662288, 0.152736 ], [ -0.852144, -0.264192 ] ],
    	[ [ -1.180256, -0.39995200000000003 ], [ -1.028144, -0.23968 ], [ -0.099568, 0.06096000000000001 ], [ -0.259328, -0.292944 ], [ -0.92784, -0.16153599999999999 ] ],
    	[ [ -0.23067200000000004, -0.24683200000000002 ], [ 0.43443200000000004, 0.294688 ], [ 0.522544, 0.256656 ], [ -0.61704, -0.189008 ], [ 1.045616, 0.297536 ] ],
    	[ [ 0.85792, 0.263104 ], [ -0.449216, -0.19137600000000002 ], [ -0.10745599999999998, 0.085376 ], [ -0.359168, -0.100832 ], [ -0.29563199999999995, 0.24710400000000002 ] ]
    ]
    --------------------
    Derivative: 
    [
    	[ [ -0.472, 0.396 ], [ -0.472, 0.396 ], [ -0.472, 0.396 ], [ -0.472, 0.396 ], [ -0.472, 0.396 ] ],
    	[ [ -0.472, 0.396 ], [ -0.472, 0.396 ], [ -0.472, 0.396 ], [ -0.472, 0.396 ], [ -0.472, 0.396 ] ],
    	[ [ -0.472, 0.396 ], [ -0.472, 0.396 ], [ -0.472, 0.396 ], [ -0.472, 0.396 ], [ -0.472, 0.396 ] ],
    	[ [ -0.472, 0.396 ], [ -0.472, 0.396 ], [ -0.472, 0.396 ], [ -0.472, 0.396 ], [ -0.472, 0.396 ] ],
    	[ [ -0.472, 0.396 ], [ -0.472, 0.396 ], [ -0.472, 0.396 ], [ -0.472, 0.396 ], [ -0.472, 0.396 ] ]
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
    	[ [ -0.416, -1.5 ], [ -1.844, 1.6 ], [ 0.76, 1.916 ], [ 1.352, 0.336 ], [ -1.112, -1.352 ], [ -1.28, -1.376 ], [ -0.544, 0.952 ], [ -1.976, 1.944 ], ... ],
    	[ [ -0.74, -0.916 ], [ 1.312, 0.024 ], [ -0.16, -0.532 ], [ 1.808, -1.148 ], [ -1.268, 1.44 ], [ -0.604, 1.8 ], [ -0.764, 0.4 ], [ -0.624, 0.704 ], ... ],
    	[ [ 0.188, 1.008 ], [ -0.452, 1.016 ], [ -1.412, -1.14 ], [ 1.668, -1.692 ], [ 1.4, 1.736 ], [ -1.36, 0.332 ], [ 1.312, -1.976 ], [ -0.324, -1.484 ], ... ],
    	[ [ 1.572, 1.704 ], [ -0.356, -0.148 ], [ -0.996, -1.168 ], [ 0.06, 1.308 ], [ -1.86, 1.408 ], [ -0.856, -1.508 ], [ -1.564, 0.352 ], [ -1.64, 0.476 ], ... ],
    	[ [ -1.452, -1.12 ], [ -1.864, 0.936 ], [ -0.66, 1.872 ], [ 0.792, -1.364 ], [ -0.948, 1.672 ], [ -0.892, -0.632 ], [ -0.224, -0.44 ], [ -1.752, -0.492 ], ... ],
    	[ [ -1.744, -1.604 ], [ 1.044, -1.06 ], [ 0.492, -0.244 ], [ -0.24, 0.192 ], [ 0.288, 1.712 ], [ 0.736, -1.916 ], [ -0.44, -1.144 ], [ 0.548, -1.516 ], ... ],
    	[ [ -0.368, 0.196 ], [ 1.86, 0.524 ], [ 0.456, 0.128 ], [ -1.564, 0.232 ], [ -0.504, 1.732 ], [ -1.936, 1.132 ], [ -0.94, 0.984 ], [ 0.176, -1.488 ], ... ],
    	[ [ -1.312, -1.516 ], [ -0.492, -0.22 ], [ -1.788, -0.788 ], [ -0.128, -0.376 ], [ -0.68, -0.172 ], [ -0.764, -0.6 ], [ -0.924, 1.28 ], [ -1.308, -0.48 ], ... ],
    	...
    ]
```



First, we use a conjugate gradient descent method, which converges the fastest for purely linear functions.

Code from [LearningTester.java:225](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L225) executed in 2.94 seconds: 
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
    F(0.0) = LineSearchPoint{point=PointSample{avg=0.34970361972377706}, derivative=-1.409401981639867E-5}
    New Minimum: 0.34970361972377706 > 0.3497036197237757
    F(1.0E-10) = LineSearchPoint{point=PointSample{avg=0.3497036197237757}, derivative=-1.409401981639864E-5}, delta = -1.3322676295501878E-15
    New Minimum: 0.3497036197237757 > 0.3497036197237677
    F(7.000000000000001E-10) = LineSearchPoint{point=PointSample{avg=0.3497036197237677}, derivative=-1.409401981639845E-5}, delta = -9.381384558082573E-15
    New Minimum: 0.3497036197237677 > 0.3497036197237082
    F(4.900000000000001E-9) = LineSearchPoint{point=PointSample{avg=0.3497036197237082}, derivative=-1.4094019816397125E-5}, delta = -6.88338275267597E-14
    New Minimum: 0.3497036197237082 > 0.34970361972329095
    F(3.430000000000001E-8) = LineSearchPoint{point=PointSample{avg=0.34970361972329095}, derivative=-1.4094019816387854E-5}, delta = -4.861111513321248E-13
    New Minimum: 0.34970361972329095 > 0.349703619720392
    F(2.4010
```
...[skipping 127909 bytes](etc/56.txt)...
```
    ght bracket at 0.01081350562578125
    F(0.005406752812890625) = LineSearchPoint{point=PointSample{avg=6.087937107647931E-34}, derivative=-9.596978448341558E-39}, delta = 0.0
    Right bracket at 0.005406752812890625
    F(0.0027033764064453127) = LineSearchPoint{point=PointSample{avg=6.087937107647931E-34}, derivative=-9.596978448341558E-39}, delta = 0.0
    Right bracket at 0.0027033764064453127
    F(0.0013516882032226563) = LineSearchPoint{point=PointSample{avg=6.087937107647931E-34}, derivative=-9.596978448341558E-39}, delta = 0.0
    Right bracket at 0.0013516882032226563
    F(6.758441016113282E-4) = LineSearchPoint{point=PointSample{avg=6.087937107647931E-34}, derivative=-9.596978448341558E-39}, delta = 0.0
    Right bracket at 6.758441016113282E-4
    F(3.379220508056641E-4) = LineSearchPoint{point=PointSample{avg=6.087937107647931E-34}, derivative=-9.596978448341558E-39}, delta = 0.0
    Loops = 12
    Iteration 66 failed, aborting. Error: 6.087937107647931E-34 Total: 239494881078119.2800; Orientation: 0.0007; Line Search: 0.1540
    
```

Returns: 

```
    6.087937107647931E-34
```



Training Converged

Next, we run the same optimization using L-BFGS, which is nearly ideal for purely second-order or quadratic functions.

Code from [LearningTester.java:249](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L249) executed in 8.82 seconds: 
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
    th(0)=0.34970361972377706;dx=-1.409401981639867E-5
    New Minimum: 0.34970361972377706 > 0.3496732558104814
    WOLFE (weak): th(2.154434690031884)=0.3496732558104814; dx=-1.4093340362427521E-5 delta=3.0363913295639033E-5
    New Minimum: 0.3496732558104814 > 0.349642893361025
    WOLFE (weak): th(4.308869380063768)=0.349642893361025; dx=-1.409266090845637E-5 delta=6.072636275206156E-5
    New Minimum: 0.349642893361025 > 0.3495214582015945
    WOLFE (weak): th(12.926608140191302)=0.3495214582015945; dx=-1.4089943092571768E-5 delta=1.8216152218253345E-4
    New Minimum: 0.3495214582015945 > 0.34897528982432136
    WOLFE (weak): th(51.70643256076521)=0.34897528982432136; dx=-1.4077712921091059E-5 delta=7.283298994557041E-4
    New Minimum: 0.34897528982432136 > 0.34607040194032285
    WOLFE (weak): th(258.53216280382605)=0.34607040194032285; dx=-1.4012485339860614E-5 delta=0.0036332177834542123
    New Minimum: 0.34607040194032285 > 0.3282205022914988
    WOLFE (weak
```
...[skipping 322580 bytes](etc/57.txt)...
```
    71341 > 0.002816426271728846
    WOLFE (weak): th(4.308869380063768)=0.002816426271728846; dx=-2.0876271216501788E-8 delta=8.99538441074231E-8
    New Minimum: 0.002816426271728846 > 0.0028162463683503646
    WOLFE (weak): th(12.926608140191302)=0.0028162463683503646; dx=-2.08756044254505E-8 delta=2.698572225887097E-7
    New Minimum: 0.0028162463683503646 > 0.0028154368742566178
    WOLFE (weak): th(51.70643256076521)=0.0028154368742566178; dx=-2.0872603865719705E-8 delta=1.079351316335582E-6
    New Minimum: 0.0028154368742566178 > 0.002811121537634566
    WOLFE (weak): th(258.53216280382605)=0.002811121537634566; dx=-2.0856600880488797E-8 delta=5.394687938387391E-6
    New Minimum: 0.002811121537634566 > 0.0027842256720624872
    WOLFE (weak): th(1551.1929768229563)=0.0027842256720624872; dx=-2.0756582222795623E-8 delta=3.229055351046612E-5
    MAX ALPHA: th(0)=0.0028165162255729534;th'(0)=-2.087660461202743E-8;
    Iteration 250 complete. Error: 0.0027842256720624872 Total: 239503705663937.3400; Orientation: 0.0012; Line Search: 0.0356
    
```

Returns: 

```
    0.0027842256720624872
```



This training run resulted in the following regressed input:

Code from [LearningTester.java:154](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L154) executed in 0.01 seconds: 
```java
    return Arrays.stream(input_lbgfs).map(x -> x.prettyPrint()).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [
    	[ [ -0.0296152462095026, -1.2901019492098387 ], [ -1.2134807878875413, 1.9421070453409066 ], [ 0.3830174075423758, 1.7110935737320416 ], [ 0.8891887482468287, 0.0846485181165114 ], [ -0.8269318243075867, -1.1973746939963072 ], [ -1.1722853080456908, -1.3173819133627405 ], [ -0.7950393334933297, 0.8154362462204814 ], [ -2.298585941748816, 1.7683800039609119 ], ... ],
    	[ [ -0.7504851048840496, -0.9215935250880151 ], [ 1.3794148528080599, 0.060797366901393944 ], [ -0.15803413165492422, -0.530944925773448 ], [ 1.8153620407036994, -1.1435764260818333 ], [ -1.0751271911885922, 1.544519364255419 ], [ -0.9610053001261615, 1.6059828914064675 ], [ -0.5888371305605952, 0.495282080718215 ], [ -0.9864519214281432, 0.5070061490370559 ], ... ],
    	[ [ 0.11170637416519837, 0.9666792576303166 ], [ -0.7175538041120958, 0.8716486825967946 ], [ -0.60390199022542, -0.7009839232209892 ], [ 1.3635388851125738, -1.8569106708219254 ], [ 1.3410039124244713, 1.7038584185990786 ], [ -1.6146334186301832, 0.1935884601360307 ], [ 1.463829
```
...[skipping 937 bytes](etc/58.txt)...
```
    443181504, -0.36517300226162874 ], [ 0.023929850713133548, 0.33525676846294106 ], [ 0.2289129624543263, 1.679937646717964 ], [ 0.717617599783279, -1.9259632329960887 ], [ 0.027461137563250897, -0.8901851242666625 ], [ 0.660885123808643, -1.4542490457374009 ], ... ],
    	[ [ -0.10771901745495176, 0.33715361295536045 ], [ 1.608915902087066, 0.38742555409952906 ], [ 0.34422879943302537, 0.06720158202665623 ], [ -1.0278524870628019, 0.5232188051642258 ], [ -0.691443382470806, 1.6297667987880582 ], [ -1.8443084814772017, 1.1818442634764126 ], [ -1.1427378908954253, 0.8738736821484396 ], [ 0.29973786538581787, -1.4206080902018963 ], ... ],
    	[ [ -1.1604070868979721, -1.433555635401449 ], [ -0.4296898535478077, -0.18621885018172976 ], [ -0.8899734790285422, -0.30024723178763313 ], [ -0.02939792178207973, -0.3222034267379683 ], [ -0.6992877072204225, -0.1824217527173053 ], [ -0.6992939625483501, -0.5650812540661977 ], [ -0.7807396628285078, 1.3577284042392546 ], [ -1.317525512124448, -0.48516241976597196 ], ... ],
    	...
    ]
```



Code from [LearningTester.java:95](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L95) executed in 0.01 seconds: 
```java
    return TestUtil.compare(runs);
```

Returns: 

![Result](etc/test.43.png)



Code from [LearningTester.java:98](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L98) executed in 0.00 seconds: 
```java
    return TestUtil.compareTime(runs);
```

Returns: 

![Result](etc/test.44.png)



### Model Learning
In this test, attempt to train a network to emulate a randomized network given an example input/output. The target state is:

Code from [LearningTester.java:176](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L176) executed in 0.00 seconds: 
```java
    return network_target.state().stream().map(Arrays::toString).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [-0.428, 0.164, -0.044, 0.232]
```



First, we use a conjugate gradient descent method, which converges the fastest for purely linear functions.

Code from [LearningTester.java:225](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L225) executed in 0.35 seconds: 
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
    F(0.0) = LineSearchPoint{point=PointSample{avg=0.3602570476193828}, derivative=-0.9593227061745141}
    New Minimum: 0.3602570476193828 > 0.36025704752344895
    F(1.0E-10) = LineSearchPoint{point=PointSample{avg=0.36025704752344895}, derivative=-0.9593227060467686}, delta = -9.59338719574987E-11
    New Minimum: 0.36025704752344895 > 0.36025704694785216
    F(7.000000000000001E-10) = LineSearchPoint{point=PointSample{avg=0.36025704694785216}, derivative=-0.9593227052803528}, delta = -6.71530664408948E-10
    New Minimum: 0.36025704694785216 > 0.3602570429186958
    F(4.900000000000001E-9) = LineSearchPoint{point=PointSample{avg=0.3602570429186958}, derivative=-0.9593226999154632}, delta = -4.700687006309323E-9
    New Minimum: 0.3602570429186958 > 0.3602570147146107
    F(3.430000000000001E-8) = LineSearchPoint{point=PointSample{avg=0.3602570147146107}, derivative=-0.9593226623612259}, delta = -3.290477212924969E-8
    New Minimum: 0.3602570147146107 > 0.3602568172860373
    F(2.4010000000000004E-
```
...[skipping 6064 bytes](etc/59.txt)...
```
    0117
    Zero gradient: 2.6239576965841143E-13
    F(0.0) = LineSearchPoint{point=PointSample{avg=2.5711686227464612E-26}, derivative=-6.88515399346301E-26}
    New Minimum: 2.5711686227464612E-26 > 1.9299821116448304E-30
    F(0.7527926001292405) = LineSearchPoint{point=PointSample{avg=1.9299821116448304E-30}, derivative=5.445124897867556E-28}, delta = -2.5709756245352968E-26
    1.9299821116448304E-30 <= 2.5711686227464612E-26
    Converged to right
    Iteration 7 complete. Error: 1.9299821116448304E-30 Total: 239504170910251.8400; Orientation: 0.0001; Line Search: 0.0143
    Zero gradient: 2.273000339899278E-15
    F(0.0) = LineSearchPoint{point=PointSample{avg=1.9299821116448304E-30}, derivative=-5.166530545182232E-30}
    New Minimum: 1.9299821116448304E-30 > 0.0
    F(0.7527926001292405) = LineSearchPoint{point=PointSample{avg=0.0}, derivative=0.0}, delta = -1.9299821116448304E-30
    0.0 <= 1.9299821116448304E-30
    Converged to right
    Iteration 8 complete. Error: 0.0 Total: 239504191457478.8400; Orientation: 0.0001; Line Search: 0.0147
    
```

Returns: 

```
    0.0
```



Training Converged

Next, we run the same optimization using L-BFGS, which is nearly ideal for purely second-order or quadratic functions.

Code from [LearningTester.java:249](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L249) executed in 0.72 seconds: 
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
    th(0)=0.10094340453695991;dx=-0.26753919944544335
    Armijo: th(2.154434690031884)=0.34740285606192217; dx=0.4963318870727579 delta=-0.24645945152496226
    New Minimum: 0.10094340453695991 > 0.01845933436104537
    WOLF (strong): th(1.077217345015942)=0.01845933436104537; dx=0.1143963438136582 delta=0.08248407017591454
    END: th(0.3590724483386473)=0.02773453760312294; dx=-0.14022735169240874 delta=0.07320886693383696
    Iteration 1 complete. Error: 0.01845933436104537 Total: 239504237541334.8000; Orientation: 0.0001; Line Search: 0.0294
    LBFGS Accumulation History: 1 points
    th(0)=0.02773453760312294;dx=-0.07350121301354125
    New Minimum: 0.02773453760312294 > 1.8763098317197204E-5
    WOLF (strong): th(0.7735981389354633)=1.8763098317197204E-5; dx=0.0018470217484351926 delta=0.027715774504805744
    END: th(0.3867990694677316)=0.006590493577727947; dx=-0.035827095632552944 delta=0.021144044025394992
    Iteration 2 complete. Error: 1.876309831719720
```
...[skipping 9681 bytes](etc/60.txt)...
```
    on History: 1 points
    th(0)=1.1566314279961131E-30;dx=-3.0592992433639605E-30
    New Minimum: 1.1566314279961131E-30 > 8.479860376499643E-31
    WOLF (strong): th(1.4128508391203713)=8.479860376499643E-31; dx=2.6189006852871013E-30 delta=3.086453903461488E-31
    New Minimum: 8.479860376499643E-31 > 6.73530697361825E-33
    END: th(0.7064254195601857)=6.73530697361825E-33; dx=-2.093124585730213E-31 delta=1.1498961210224949E-30
    Iteration 21 complete. Error: 6.73530697361825E-33 Total: 239504880147085.1600; Orientation: 0.0001; Line Search: 0.0273
    LBFGS Accumulation History: 1 points
    th(0)=6.73530697361825E-33;dx=-1.4419505881270174E-32
    New Minimum: 6.73530697361825E-33 > 6.546086348011406E-33
    WOLF (strong): th(1.521947429820792)=6.546086348011406E-33; dx=1.4187506040740776E-32 delta=1.892206256068438E-34
    New Minimum: 6.546086348011406E-33 > 0.0
    END: th(0.760973714910396)=0.0; dx=0.0 delta=6.73530697361825E-33
    Iteration 22 complete. Error: 0.0 Total: 239504911213434.1200; Orientation: 0.0001; Line Search: 0.0244
    
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

![Result](etc/test.45.png)



Code from [LearningTester.java:98](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L98) executed in 0.00 seconds: 
```java
    return TestUtil.compareTime(runs);
```

Returns: 

![Result](etc/test.46.png)



### Performance
Now we execute larger-scale runs to benchmark performance:

Code from [PerformanceTester.java:66](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/PerformanceTester.java#L66) executed in 0.62 seconds: 
```java
    test(component, inputPrototype);
```
Logging: 
```
    100 batches
    Input Dimensions:
    	[100, 100, 2]
    Performance:
    	Evaluation performance: 0.016265s +- 0.001091s [0.014622s - 0.017238s]
    	Learning performance: 0.086941s +- 0.030231s [0.061946s - 0.142400s]
    
```

