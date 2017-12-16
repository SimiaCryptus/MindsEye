# SimpleConvolutionLayer
## MultiBand
### Batch Execution
Code from [BatchingTester.java:66](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/BatchingTester.java#L66) executed in 0.03 seconds: 
```java
    return test(reference, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (60#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (60#)}
```



Code from [SingleDerivativeTester.java:77](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/SingleDerivativeTester.java#L77) executed in 0.06 seconds: 
```java
    return test(component, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ -0.652, -0.184, 1.66 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.2336088310728294, negative=2, min=1.66, max=1.66, mean=0.2746666666666666, count=3.0, positive=1, stdDev=0.9980371847893355, zeros=0}
    Output: [
    	[ [ -0.543168, 1.854704, -0.5820480000000001 ] ]
    ]
    Outputs Statistics: {meanExponent=-0.07727747132126114, negative=2, min=-0.5820480000000001, max=-0.5820480000000001, mean=0.24316266666666655, count=3.0, positive=1, stdDev=1.1396423460546248, zeros=0}
    Feedback for input 0
    Inputs Values: [
    	[ [ -0.652, -0.184, 1.66 ] ]
    ]
    Value Statistics: {meanExponent=-0.2336088310728294, negative=2, min=1.66, max=1.66, mean=0.2746666666666666, count=3.0, positive=1, stdDev=0.9980371847893355, zeros=0}
    Implemented Feedback: [ [ 0.136, 1.756, 1.036 ], [ -1.716, -0.424, 1.044 ], [ -0.464, 1.76, 0.172 ] ]
    Implemented Statistics: {meanExponent=-0.17538156562247137, negative=3, min=0.172, max=0.172, mean=0.3666666666666667, count=9.0, positive=6, stdDev=1.0812634584904213, zeros=0}
    Measured Feedback: 
```
...[skipping 1656 bytes](etc/171.txt)...
```
    , -0.18399999999973993 ], [ 0.0, 0.0, 1.6599999999999948 ] ]
    Measured Statistics: {meanExponent=-0.23360883107262664, negative=6, min=1.6599999999999948, max=1.6599999999999948, mean=0.09155555555547441, count=27.0, positive=3, stdDev=0.5905852297339772, zeros=18}
    Gradient Error: [ [ 1.2512213487525514E-13, 0.0, 0.0 ], [ -8.501532811067136E-13, 0.0, 0.0 ], [ -1.1153300505384323E-12, 0.0, 0.0 ], [ 0.0, 1.2353451595004117E-12, 0.0 ], [ 0.0, -8.501532811067136E-13, 0.0 ], [ 0.0, -1.1153300505384323E-12, 0.0 ], [ 0.0, 0.0, 1.2512213487525514E-13 ], [ 0.0, 0.0, 2.600697435184429E-13 ], [ 0.0, 0.0, -5.10702591327572E-15 ] ]
    Error Statistics: {meanExponent=-12.515164913525876, negative=5, min=-5.10702591327572E-15, max=-5.10702591327572E-15, mean=-8.112646357163713E-14, count=27.0, positive=4, stdDev=4.464175912556277E-13, zeros=18}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 2.9882e-13 +- 4.5276e-13 [0.0000e+00 - 1.6451e-12] (36#)
    relativeTol: 7.2273e-13 +- 8.0708e-13 [1.5383e-15 - 2.3102e-12] (18#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=2.9882e-13 +- 4.5276e-13 [0.0000e+00 - 1.6451e-12] (36#), relativeTol=7.2273e-13 +- 8.0708e-13 [1.5383e-15 - 2.3102e-12] (18#)}
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
      "id": "406491d3-2cf0-42dd-923d-84faef913f92",
      "isFrozen": false,
      "name": "ConvolutionLayer/406491d3-2cf0-42dd-923d-84faef913f92",
      "filter": [
        [
          [
            0.136
          ]
        ],
        [
          [
            1.756
          ]
        ],
        [
          [
            1.036
          ]
        ],
        [
          [
            -1.716
          ]
        ],
        [
          [
            -0.424
          ]
        ],
        [
          [
            1.044
          ]
        ],
        [
          [
            -0.464
          ]
        ],
        [
          [
            1.76
          ]
        ],
        [
          [
            0.172
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

Code from [EquivalencyTester.java:64](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/EquivalencyTester.java#L64) executed in 0.03 seconds: 
```java
    return test(subject, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ 1.804, -1.836, -1.192 ] ]
    ]
    Error: [
    	[ [ 0.0, 0.0, 0.0 ] ]
    ]
    Accuracy:
    absoluteTol: 0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (3#)
    relativeTol: 0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (3#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (3#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (3#)}
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
      "id": "bafcbbf1-ed27-43bc-afdd-2eece07d45a9",
      "isFrozen": false,
      "name": "SimpleConvolutionLayer/bafcbbf1-ed27-43bc-afdd-2eece07d45a9",
      "filter": [
        [
          [
            0.136
          ]
        ],
        [
          [
            -1.716
          ]
        ],
        [
          [
            -0.464
          ]
        ],
        [
          [
            1.756
          ]
        ],
        [
          [
            -0.424
          ]
        ],
        [
          [
            1.76
          ]
        ],
        [
          [
            1.036
          ]
        ],
        [
          [
            1.044
          ]
        ],
        [
          [
            0.172
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
    	[ [ 0.468, 1.248, 0.732 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ -2.4175679999999997, 1.5809760000000002, 1.913664 ] ]
    ]
    --------------------
    Derivative: 
    [
    	[ [ 2.928, -1.096, 1.468 ] ]
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
    	[ [ 1.628, 0.828, -0.9 ], [ -0.584, -1.404, 0.38 ], [ -1.828, -0.86, -0.732 ], [ -0.34, 1.296, -0.236 ], [ -0.696, -1.692, -0.396 ], [ 1.924, -1.672, -0.572 ], [ 0.14, -0.62, -1.096 ], [ 0.1, 0.62, -1.368 ], ... ],
    	[ [ -0.784, 1.768, 0.512 ], [ -0.968, -1.768, 0.332 ], [ -0.088, 0.216, 1.092 ], [ -0.448, -0.668, -1.404 ], [ 1.044, -1.816, 0.256 ], [ -0.928, -0.884, 1.16 ], [ 0.416, 0.62, -0.172 ], [ 1.636, -0.9, -1.236 ], ... ],
    	[ [ -1.944, -0.356, -1.36 ], [ -0.88, -1.304, 0.12 ], [ -0.364, -1.984, 0.664 ], [ -0.176, -0.116, -0.936 ], [ 0.716, 1.62, -1.684 ], [ -1.844, 1.112, 1.736 ], [ 1.008, 0.124, -1.564 ], [ -1.616, -1.04, -1.552 ], ... ],
    	[ [ -0.116, -1.86, 0.888 ], [ -1.132, 0.376, -1.632 ], [ 0.916, -1.968, 1.492 ], [ 1.12, 1.56, -1.804 ], [ 0.108, -0.476, 0.14 ], [ 1.872, 1.032, -0.54 ], [ -0.504, -0.356, 1.268 ], [ 1.88, 0.688, 1.364 ], ... ],
    	[ [ -1.78, 0.716, 0.832 ], [ -0.088, -0.42, -0.036 ], [ -1.396, 1.736, -0.684 ], [ 1.896, -1.996, -0.136 ], [ 1.512, -0.412, 1.912 ], [ -1.108, -1.824, -1.112 ], [ 1.672, -0.336, -1.992 ], [ -0.604, 1.34, 1.936 ], ... ],
    	[ [ -1.348, -0.052, 0.86 ], [ 1.372, -1.276, -1.504 ], [ -0.048, -1.084, 1.028 ], [ -1.292, 1.328, 1.216 ], [ -1.644, -1.216, -1.696 ], [ -1.216, 0.264, -0.224 ], [ -1.024, -1.244, 0.656 ], [ -0.696, 0.34, -0.22 ], ... ],
    	[ [ -1.732, 1.468, 1.072 ], [ -1.004, 0.46, -0.912 ], [ 1.632, 1.88, 1.404 ], [ -1.876, 1.948, 0.28 ], [ -1.804, 1.412, 1.588 ], [ 1.968, 0.688, 1.908 ], [ -0.392, -0.932, -0.8 ], [ 0.584, 1.94, -1.208 ], ... ],
    	[ [ -1.112, 0.12, -0.204 ], [ -1.172, 1.964, -0.38 ], [ 0.092, -1.36, 1.92 ], [ -0.128, -1.544, 0.292 ], [ -1.792, 1.48, -1.668 ], [ -1.98, 1.604, 1.732 ], [ 1.776, 0.528, 1.504 ], [ -0.368, 0.376, -1.612 ], ... ],
    	...
    ]
```



First, we use a conjugate gradient descent method, which converges the fastest for purely linear functions.

Code from [LearningTester.java:225](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L225) executed in 3.16 seconds: 
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
    F(0.0) = LineSearchPoint{point=PointSample{avg=10.325139344050868}, derivative=-0.006769691996734621}
    New Minimum: 10.325139344050868 > 10.325139344050198
    F(1.0E-10) = LineSearchPoint{point=PointSample{avg=10.325139344050198}, derivative=-0.00676969199673437}, delta = -6.696865284538944E-13
    New Minimum: 10.325139344050198 > 10.325139344046137
    F(7.000000000000001E-10) = LineSearchPoint{point=PointSample{avg=10.325139344046137}, derivative=-0.0067696919967328685}, delta = -4.730438263322867E-12
    New Minimum: 10.325139344046137 > 10.325139344017774
    F(4.900000000000001E-9) = LineSearchPoint{point=PointSample{avg=10.325139344017774}, derivative=-0.006769691996722356}, delta = -3.3093527918026666E-11
    New Minimum: 10.325139344017774 > 10.325139343818597
    F(3.430000000000001E-8) = LineSearchPoint{point=PointSample{avg=10.325139343818597}, derivative=-0.006769691996648771}, delta = -2.3227109124945855E-10
    New Minimum: 10.325139343818597 > 10.325139342425407
    F(2.40100000
```
...[skipping 108310 bytes](etc/172.txt)...
```
    acket at 0.01081350562578125
    F(0.005406752812890625) = LineSearchPoint{point=PointSample{avg=1.0495957004720415E-32}, derivative=-4.613882190683131E-36}, delta = 0.0
    Right bracket at 0.005406752812890625
    F(0.0027033764064453127) = LineSearchPoint{point=PointSample{avg=1.0495957004720415E-32}, derivative=-4.613882190683131E-36}, delta = 0.0
    Right bracket at 0.0027033764064453127
    F(0.0013516882032226563) = LineSearchPoint{point=PointSample{avg=1.0495957004720415E-32}, derivative=-4.613882190683131E-36}, delta = 0.0
    Right bracket at 0.0013516882032226563
    F(6.758441016113282E-4) = LineSearchPoint{point=PointSample{avg=1.0495957004720415E-32}, derivative=-4.613882190683131E-36}, delta = 0.0
    Right bracket at 6.758441016113282E-4
    F(3.379220508056641E-4) = LineSearchPoint{point=PointSample{avg=1.0495957004720415E-32}, derivative=-4.613882190683131E-36}, delta = 0.0
    Loops = 12
    Iteration 90 failed, aborting. Error: 1.0495957004720415E-32 Total: 239630136364761.0300; Orientation: 0.0009; Line Search: 0.1607
    
```

Returns: 

```
    1.0495957004720415E-32
```



Training Converged

Next, we run the same optimization using L-BFGS, which is nearly ideal for purely second-order or quadratic functions.

Code from [LearningTester.java:249](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L249) executed in 0.32 seconds: 
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
    th(0)=10.325139344050868;dx=-0.006769691996734621
    New Minimum: 10.325139344050868 > 10.310560293485784
    WOLFE (weak): th(2.154434690031884)=10.310560293485784; dx=-0.006764299664749016 delta=0.014579050565084017
    New Minimum: 10.310560293485784 > 10.295992860347734
    WOLFE (weak): th(4.308869380063768)=10.295992860347734; dx=-0.006758907332763412 delta=0.02914648370313344
    New Minimum: 10.295992860347734 > 10.237839302066845
    WOLFE (weak): th(12.926608140191302)=10.237839302066845; dx=-0.006737338004820994 delta=0.08730004198402241
    New Minimum: 10.237839302066845 > 9.978448540366399
    WOLFE (weak): th(51.70643256076521)=9.978448540366399; dx=-0.0066402760290801156 delta=0.3466908036844689
    New Minimum: 9.978448540366399 > 8.658601705666966
    WOLFE (weak): th(258.53216280382605)=8.658601705666966; dx=-0.0061226121584620945 delta=1.6665376383839021
    New Minimum: 8.658601705666966 > 2.8352777651776946
    END: th(1551.1929768229563)=2.835
```
...[skipping 4798 bytes](etc/173.txt)...
```
    5E-4; dx=-3.223141617406227E-8 delta=2.6056073642482646E-4
    Iteration 11 complete. Error: 1.0592433444428589E-4 Total: 239630418608497.6200; Orientation: 0.0022; Line Search: 0.0203
    LBFGS Accumulation History: 1 points
    th(0)=1.258729381539365E-4;dx=-3.209668920645679E-8
    New Minimum: 1.258729381539365E-4 > 8.999118141572332E-5
    WOLF (strong): th(9375.000000000005)=8.999118141572332E-5; dx=2.4441914435638005E-8 delta=3.588175673821318E-5
    New Minimum: 8.999118141572332E-5 > 4.1675883641750744E-5
    END: th(4687.500000000003)=4.1675883641750744E-5; dx=-3.827387385409399E-9 delta=8.419705451218575E-5
    Iteration 12 complete. Error: 4.1675883641750744E-5 Total: 239630440352633.6000; Orientation: 0.0020; Line Search: 0.0157
    LBFGS Accumulation History: 1 points
    th(0)=4.1675883641750744E-5;dx=-1.4917483252180415E-8
    MAX ALPHA: th(0)=4.1675883641750744E-5;th'(0)=-1.4917483252180415E-8;
    Iteration 13 failed, aborting. Error: 4.1675883641750744E-5 Total: 239630455093711.5600; Orientation: 0.0015; Line Search: 0.0090
    
```

Returns: 

```
    4.1675883641750744E-5
```



This training run resulted in the following regressed input:

Code from [LearningTester.java:154](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L154) executed in 0.01 seconds: 
```java
    return Arrays.stream(input_lbgfs).map(x -> x.prettyPrint()).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [
    	[ [ 1.6197966759933111, 0.8198111380027459, -0.8949547379489937 ], [ -0.5837771925850395, -1.4041987579168653, 0.3802709812053161 ], [ -1.8283541213230707, -0.8583838092902782, -0.7336903540111172 ], [ -0.34217401117371643, 1.2955836116786081, -0.23636192126234798 ], [ -0.691550367244749, -1.686995721906947, -0.39928155799265325 ], [ 1.9257578902025512, -1.670406284377544, -0.5729251062038473 ], [ 0.14456978914238328, -0.6158124696223786, -1.0984480305428752 ], [ 0.0961233858684789, 0.6185471473225997, -1.3679571936087833 ], ... ],
    	[ [ -0.791683334071383, 1.760376753647729, 0.5166803661899964 ], [ -0.9642512159442626, -1.7660546650811622, 0.33143509911867447 ], [ -0.08840698659917758, 0.21412077400854435, 1.0936772443109788 ], [ -0.43978773442931657, -0.6583205941501029, -1.4104860886621016 ], [ 1.0451314600110684, -1.8186171300732439, 0.258933661104505 ], [ -0.9235332770320364, -0.8777560570701168, 1.1555235246272078 ], [ 0.41239814141425757, 0.6179113962940103, -0.17124458551349675 ], [ 1.63136775092084
```
...[skipping 2237 bytes](etc/174.txt)...
```
    , 1.0683582400893143 ], [ -1.006335268637363, 0.45958383956051424, -0.9124189082150872 ], [ 1.6233732771256764, 1.8701139591333373, 1.410540369518325 ], [ -1.8759271723781688, 1.9491424753887476, 0.2789188563621731 ], [ -1.796259198190376, 1.4184851594457746, 1.5844423929911036 ], [ 1.9739485539712474, 0.6910418040138075, 1.907147255308259 ], [ -0.3829037053135559, -0.9225021660773761, -0.805998991778593 ], [ 0.5703659409596918, 1.928443172300211, -1.2016037381806528 ], ... ],
    	[ [ -1.1102575391762206, 0.12137855985783937, -0.2047221040922712 ], [ -1.172080704920241, 1.9660588434247181, -0.382022928108527 ], [ 0.10286988482180763, -1.3517835054595186, 1.915866657339778 ], [ -0.12563511242257563, -1.5446913700272518, 0.29350226220353276 ], [ -1.7954825789028113, 1.4826420041563224, -1.6717854040670597 ], [ -1.9817313780894443, 1.604974382333389, 1.7304465771768838 ], [ 1.7801277300258813, 0.5299888822687214, 1.5035263097561562 ], [ -0.36629487848376396, 0.37805645191340687, -1.6133919607770655 ], ... ],
    	...
    ]
```



Code from [LearningTester.java:95](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L95) executed in 0.00 seconds: 
```java
    return TestUtil.compare(runs);
```

Returns: 

![Result](etc/test.100.png)



Code from [LearningTester.java:98](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L98) executed in 0.00 seconds: 
```java
    return TestUtil.compareTime(runs);
```

Returns: 

![Result](etc/test.101.png)



### Model Learning
In this test, attempt to train a network to emulate a randomized network given an example input/output. The target state is:

Code from [LearningTester.java:176](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L176) executed in 0.00 seconds: 
```java
    return network_target.state().stream().map(Arrays::toString).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [-1.716, 1.036, -0.464, 0.172, 1.044, 1.76, 1.756, -0.424, 0.136]
```



First, we use a conjugate gradient descent method, which converges the fastest for purely linear functions.

Code from [LearningTester.java:225](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L225) executed in 0.48 seconds: 
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
    F(0.0) = LineSearchPoint{point=PointSample{avg=9.829480406950648}, derivative=-17.39717890395771}
    New Minimum: 9.829480406950648 > 9.829480405210989
    F(1.0E-10) = LineSearchPoint{point=PointSample{avg=9.829480405210989}, derivative=-17.397178902418034}, delta = -1.7396590834550807E-9
    New Minimum: 9.829480405210989 > 9.829480394772549
    F(7.000000000000001E-10) = LineSearchPoint{point=PointSample{avg=9.829480394772549}, derivative=-17.397178893179913}, delta = -1.2178098529602721E-8
    New Minimum: 9.829480394772549 > 9.829480321704423
    F(4.900000000000001E-9) = LineSearchPoint{point=PointSample{avg=9.829480321704423}, derivative=-17.39717882851306}, delta = -8.524622430172712E-8
    New Minimum: 9.829480321704423 > 9.829479810227397
    F(3.430000000000001E-8) = LineSearchPoint{point=PointSample{avg=9.829479810227397}, derivative=-17.397178375845222}, delta = -5.967232503678588E-7
    New Minimum: 9.829479810227397 > 9.829476229888446
    F(2.4010000000000004E-7) = LineSearchPoint{
```
...[skipping 9400 bytes](etc/175.txt)...
```
    {point=PointSample{avg=2.3108032872174353E-27}, derivative=4.106406512300061E-27}, delta = -3.2478589989428124E-29
    2.3108032872174353E-27 <= 2.3432818772068634E-27
    New Minimum: 1.0384958917745315E-30 > 3.0715466153861724E-31
    F(1.1327633382696016) = LineSearchPoint{point=PointSample{avg=3.0715466153861724E-31}, derivative=-1.0597557245211895E-30}, delta = -2.342974722545325E-27
    Left bracket at 1.1327633382696016
    Converged to left
    Iteration 8 complete. Error: 3.0715466153861724E-31 Total: 239631067357813.0000; Orientation: 0.0001; Line Search: 0.0586
    Zero gradient: 6.867092033368374E-16
    F(0.0) = LineSearchPoint{point=PointSample{avg=3.0715466153861724E-31}, derivative=-4.715695299475138E-31}
    New Minimum: 3.0715466153861724E-31 > 0.0
    F(1.1327633382696016) = LineSearchPoint{point=PointSample{avg=0.0}, derivative=0.0}, delta = -3.0715466153861724E-31
    0.0 <= 3.0715466153861724E-31
    Converged to right
    Iteration 9 complete. Error: 0.0 Total: 239631087369849.9400; Orientation: 0.0001; Line Search: 0.0138
    
```

Returns: 

```
    0.0
```



Training Converged

Next, we run the same optimization using L-BFGS, which is nearly ideal for purely second-order or quadratic functions.

Code from [LearningTester.java:249](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L249) executed in 0.49 seconds: 
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
    th(0)=5.896887004695055;dx=-10.440000227699779
    New Minimum: 5.896887004695055 > 4.853127924114591
    WOLF (strong): th(2.154434690031884)=4.853127924114591; dx=9.471060129018966 delta=1.0437590805804637
    New Minimum: 4.853127924114591 > 0.012872570925616772
    END: th(1.077217345015942)=0.012872570925616772; dx=-0.48447004934040794 delta=5.884014433769438
    Iteration 1 complete. Error: 0.012872570925616772 Total: 239631124899508.9000; Orientation: 0.0001; Line Search: 0.0215
    LBFGS Accumulation History: 1 points
    th(0)=0.012872570925616772;dx=-0.02276264040430057
    Armijo: th(2.3207944168063896)=0.0142459806223148; dx=0.02394620900197745 delta=-0.0013734096966980282
    New Minimum: 0.012872570925616772 > 9.071159273220435E-6
    WOLF (strong): th(1.1603972084031948)=9.071159273220435E-6; dx=5.917842988384504E-4 delta=0.012863499766343552
    END: th(0.3867990694677316)=0.005573581089126114; dx=-0.014977832169920931 delta=0.0072989898364906585
```
...[skipping 8010 bytes](etc/176.txt)...
```
    00; Orientation: 0.0002; Line Search: 0.0198
    LBFGS Accumulation History: 1 points
    th(0)=7.809323910447542E-28;dx=-1.3730208918091055E-27
    New Minimum: 7.809323910447542E-28 > 7.610823413888604E-28
    WOLF (strong): th(2.2605613425925934)=7.610823413888604E-28; dx=1.3554532759826528E-27 delta=1.9850049655893795E-29
    New Minimum: 7.610823413888604E-28 > 6.028991381925356E-32
    END: th(1.1302806712962967)=6.028991381925356E-32; dx=-7.271899205580063E-30 delta=7.80872101130935E-28
    Iteration 18 complete. Error: 6.028991381925356E-32 Total: 239631555270738.4700; Orientation: 0.0001; Line Search: 0.0187
    LBFGS Accumulation History: 1 points
    th(0)=6.028991381925356E-32;dx=-6.274267318411568E-32
    Armijo: th(2.4351158877132666)=6.155884984133957E-32; dx=6.320556213441194E-32 delta=-1.2689360220860084E-33
    New Minimum: 6.028991381925356E-32 > 0.0
    END: th(1.2175579438566333)=0.0; dx=0.0 delta=6.028991381925356E-32
    Iteration 19 complete. Error: 0.0 Total: 239631581125973.4400; Orientation: 0.0001; Line Search: 0.0190
    
```

Returns: 

```
    0.0
```



Training Converged

Code from [LearningTester.java:95](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L95) executed in 0.01 seconds: 
```java
    return TestUtil.compare(runs);
```

Returns: 

![Result](etc/test.102.png)



Code from [LearningTester.java:98](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L98) executed in 0.00 seconds: 
```java
    return TestUtil.compareTime(runs);
```

Returns: 

![Result](etc/test.103.png)



### Performance
Now we execute larger-scale runs to benchmark performance:

Code from [PerformanceTester.java:66](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/PerformanceTester.java#L66) executed in 0.98 seconds: 
```java
    test(component, inputPrototype);
```
Logging: 
```
    100 batches
    Input Dimensions:
    	[100, 100, 3]
    Performance:
    	Evaluation performance: 0.027832s +- 0.004801s [0.022626s - 0.033460s]
    	Learning performance: 0.129856s +- 0.024667s [0.101856s - 0.157985s]
    
```

