# SimpleConvolutionLayer
## Matrix
### Batch Execution
Code from [BatchingTester.java:66](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/BatchingTester.java#L66) executed in 0.01 seconds: 
```java
    return test(reference, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=9.8686e-18 +- 6.5461e-17 [0.0000e+00 - 4.4409e-16] (180#), relativeTol=4.6202e-18 +- 3.0647e-17 [0.0000e+00 - 2.0791e-16] (180#)}
```



Code from [SingleDerivativeTester.java:77](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/SingleDerivativeTester.java#L77) executed in 0.03 seconds: 
```java
    return test(component, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ -0.132 ], [ 1.292 ], [ -0.42 ] ],
    	[ [ 0.78 ], [ -0.416 ], [ 1.864 ] ],
    	[ [ 0.868 ], [ 0.268 ], [ 0.86 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.22912527255034998, negative=3, min=0.86, max=0.86, mean=0.5515555555555556, count=9.0, positive=6, stdDev=0.7413176124099142, zeros=0}
    Output: [
    	[ [ -1.9517440000000004 ], [ -0.0010559999999996687 ], [ 2.237824 ] ],
    	[ [ 1.194992 ], [ 0.7452639999999997 ], [ -5.71344 ] ],
    	[ [ -2.664096 ], [ 3.2547200000000003 ], [ -0.638656 ] ]
    ]
    Outputs Statistics: {meanExponent=-0.09846488777413175, negative=5, min=-0.638656, max=-0.638656, mean=-0.39291022222222227, count=9.0, positive=4, stdDev=2.585280303573796, zeros=0}
    Feedback for input 0
    Inputs Values: [
    	[ [ -0.132 ], [ 1.292 ], [ -0.42 ] ],
    	[ [ 0.78 ], [ -0.416 ], [ 1.864 ] ],
    	[ [ 0.868 ], [ 0.268 ], [ 0.86 ] ]
    ]
    Value Statistics: {meanExponent=-0.22912527255034998, negative=3, min=0.86, max=0.86, mean=0.5515555555555556, count=9.0, positive=6, stdDev=0.7413176124099142, zeros=0}
    Implemented Feed
```
...[skipping 6835 bytes](etc/166.txt)...
```
    E-13, -4.1311398746302075E-12, 3.049782648645305E-13 ], [ 0.0, 6.451505996096785E-13, 2.4453772340393698E-12, 0.0, 1.6251444634463041E-12, 1.360300760921973E-12, 0.0, 3.46572770482112E-12, 3.097522238704187E-13 ], [ 0.0, 0.0, 0.0, 2.2493118478905672E-13, -5.752065490582936E-13, 0.0, 1.360300760921973E-12, -3.3957281431185038E-12, 0.0 ], [ 0.0, 0.0, 0.0, 1.4155343563970746E-14, 2.2493118478905672E-13, 5.350164755668629E-13, -5.953015858040089E-13, -3.080591337578653E-12, -6.505906924303417E-14 ], [ 0.0, 0.0, 0.0, 0.0, -4.650724250154781E-13, -1.9955148644612564E-12, 0.0, -5.953015858040089E-13, -8.6014528832834E-13 ] ]
    Error Statistics: {meanExponent=-12.084812710122417, negative=22, min=-8.6014528832834E-13, max=-8.6014528832834E-13, mean=-2.039918302863425E-13, count=81.0, positive=27, stdDev=1.4695896746274062E-12, zeros=32}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 7.9702e-13 +- 1.0962e-12 [0.0000e+00 - 4.1359e-12] (162#)
    relativeTol: 1.4614e-12 +- 2.3533e-12 [1.5553e-14 - 1.4378e-11] (98#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=7.9702e-13 +- 1.0962e-12 [0.0000e+00 - 4.1359e-12] (162#), relativeTol=1.4614e-12 +- 2.3533e-12 [1.5553e-14 - 1.4378e-11] (98#)}
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
      "id": "ef952d88-846f-41e5-924a-69424a209f19",
      "isFrozen": false,
      "name": "ConvolutionLayer/ef952d88-846f-41e5-924a-69424a209f19",
      "filter": [
        [
          [
            0.964,
            -1.072,
            1.904
          ],
          [
            -0.536,
            -1.912,
            0.096
          ],
          [
            0.336,
            1.992,
            -0.704
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
    	[ [ -1.28 ], [ 0.596 ], [ 1.748 ] ],
    	[ [ -0.624 ], [ 0.116 ], [ -1.896 ] ],
    	[ [ 0.728 ], [ 0.364 ], [ -1.74 ] ]
    ]
    Error: [
    	[ [ 0.0 ], [ 0.0 ], [ 0.0 ] ],
    	[ [ 0.0 ], [ 0.0 ], [ 0.0 ] ],
    	[ [ 0.0 ], [ 0.0 ], [ 0.0 ] ]
    ]
    Accuracy:
    absoluteTol: 0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (9#)
    relativeTol: 0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (9#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (9#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (9#)}
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
      "id": "08bb29a2-10c2-4ec6-890d-bb3a7a1d8f9b",
      "isFrozen": false,
      "name": "SimpleConvolutionLayer/08bb29a2-10c2-4ec6-890d-bb3a7a1d8f9b",
      "filter": [
        [
          [
            0.964,
            -1.072,
            1.904
          ],
          [
            -0.536,
            -1.912,
            0.096
          ],
          [
            0.336,
            1.992,
            -0.704
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
    	[ [ -0.828 ], [ -0.1 ], [ 0.552 ] ],
    	[ [ -0.716 ], [ -0.432 ], [ -0.516 ] ],
    	[ [ 1.94 ], [ -1.096 ], [ -0.972 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 1.6576639999999998 ], [ -2.556368 ], [ -1.1232 ] ],
    	[ [ -0.5341760000000003 ], [ 1.8794720000000003 ], [ 0.40217599999999987 ] ],
    	[ [ -3.4256319999999993 ], [ 6.482143999999999 ], [ -0.07017600000000032 ] ]
    ]
    --------------------
    Derivative: 
    [
    	[ [ -0.5279999999999996 ], [ 0.30400000000000005 ], [ -0.9840000000000001 ] ],
    	[ [ -0.7279999999999998 ], [ 1.0679999999999996 ], [ -0.556 ] ],
    	[ [ -0.1200000000000001 ], [ -0.22799999999999998 ], [ -2.556 ] ]
    ]
```



[GPU Log](etc/cuda.log)

### Input Learning
In this test, we use a network to learn this target input, given it's pre-evaluated output:

Code from [LearningTester.java:127](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L127) executed in 0.00 seconds: 
```java
    return Arrays.stream(input_target).map(x -> x.prettyPrint()).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [
    	[ [ -0.152 ], [ -0.556 ], [ -0.076 ], [ -0.764 ], [ 1.58 ], [ 0.724 ], [ 0.692 ], [ 1.68 ], ... ],
    	[ [ 0.02 ], [ 0.36 ], [ -0.832 ], [ 0.416 ], [ -1.404 ], [ 1.092 ], [ 1.976 ], [ -0.356 ], ... ],
    	[ [ 0.196 ], [ -0.836 ], [ 0.116 ], [ -0.036 ], [ -0.216 ], [ 1.8 ], [ 0.672 ], [ -1.464 ], ... ],
    	[ [ -1.412 ], [ -0.328 ], [ 0.756 ], [ -0.696 ], [ -1.048 ], [ 0.924 ], [ -0.412 ], [ 1.216 ], ... ],
    	[ [ 1.788 ], [ 0.548 ], [ -0.404 ], [ -0.12 ], [ 1.376 ], [ 1.504 ], [ 1.024 ], [ 1.568 ], ... ],
    	[ [ -0.008 ], [ 1.372 ], [ 1.692 ], [ 0.336 ], [ -0.572 ], [ -0.372 ], [ 1.892 ], [ -1.9 ], ... ],
    	[ [ 1.952 ], [ -1.756 ], [ -1.752 ], [ -1.048 ], [ -0.888 ], [ 0.14 ], [ 1.184 ], [ -1.872 ], ... ],
    	[ [ -0.932 ], [ -0.992 ], [ -1.144 ], [ 0.948 ], [ -1.0 ], [ -1.212 ], [ 0.76 ], [ 0.116 ], ... ],
    	...
    ]
```



First, we use a conjugate gradient descent method, which converges the fastest for purely linear functions.

Code from [LearningTester.java:225](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L225) executed in 3.83 seconds: 
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
    F(0.0) = LineSearchPoint{point=PointSample{avg=37.009161337074005}, derivative=-0.36082773420408104}
    New Minimum: 37.009161337074005 > 37.009161337038016
    F(1.0E-10) = LineSearchPoint{point=PointSample{avg=37.009161337038016}, derivative=-0.360827734203858}, delta = -3.5988989566249074E-11
    New Minimum: 37.009161337038016 > 37.00916133682167
    F(7.000000000000001E-10) = LineSearchPoint{point=PointSample{avg=37.00916133682167}, derivative=-0.3608277342025195}, delta = -2.523350417504844E-10
    New Minimum: 37.00916133682167 > 37.009161335306274
    F(4.900000000000001E-9) = LineSearchPoint{point=PointSample{avg=37.009161335306274}, derivative=-0.3608277341931503}, delta = -1.7677308505881228E-9
    New Minimum: 37.009161335306274 > 37.00916132469773
    F(3.430000000000001E-8) = LineSearchPoint{point=PointSample{avg=37.00916132469773}, derivative=-0.3608277341275655}, delta = -1.237627600403357E-8
    New Minimum: 37.00916132469773 > 37.00916125043927
    F(2.4010000000000004E-7) = Line
```
...[skipping 290893 bytes](etc/167.txt)...
```
    8511976E-4}, derivative=-1.0488930942941751E-8}, delta = -1.1268061859413033E-6
    F(613.576308741128) = LineSearchPoint{point=PointSample{avg=2.7356824736137245E-4}, derivative=1.7906241757538754E-8}, delta = 8.23659324233558E-7
    F(47.19817759547138) = LineSearchPoint{point=PointSample{avg=2.7208630007532635E-4}, derivative=-1.2673174996824836E-8}, delta = -6.582879618125502E-7
    New Minimum: 2.7098211396626895E-4 > 2.706623346042427E-4
    F(330.3872431682996) = LineSearchPoint{point=PointSample{avg=2.706623346042427E-4}, derivative=2.616533380357019E-9}, delta = -2.082253432896181E-6
    2.706623346042427E-4 <= 2.727445880371389E-4
    New Minimum: 2.706623346042427E-4 > 2.705989330591267E-4
    F(281.92499491971705) = LineSearchPoint{point=PointSample{avg=2.705989330591267E-4}, derivative=-7.486630778458458E-23}, delta = -2.1456549780122028E-6
    Left bracket at 281.92499491971705
    Converged to left
    Iteration 250 complete. Error: 2.705989330591267E-4 Total: 239623351812704.6600; Orientation: 0.0003; Line Search: 0.0152
    
```

Returns: 

```
    2.705989330591267E-4
```



This training run resulted in the following regressed input:

Code from [LearningTester.java:144](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L144) executed in 0.04 seconds: 
```java
    return Arrays.stream(input_gd).map(x -> x.prettyPrint()).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [
    	[ [ -0.14960311936117088 ], [ -0.5471866216572293 ], [ -0.0997601042534494 ], [ -0.7460521386170585 ], [ 1.5861241803677837 ], [ 0.6773870047162395 ], [ 0.7579740539116107 ], [ 1.6379193439487727 ], ... ],
    	[ [ -0.0014366539785754064 ], [ 0.3732300625304511 ], [ -0.8269008585895081 ], [ 0.3714334921085172 ], [ -1.3401386718767248 ], [ 1.0567097184116725 ], [ 1.9433928570235126 ], [ -0.2500732830174894 ], ... ],
    	[ [ 0.19619736140171054 ], [ -0.8616867357010314 ], [ 0.16017678646549413 ], [ -0.0604012904735763 ], [ -0.23426737828605373 ], [ 1.8698229626128178 ], [ 0.5907791776754577 ], [ -1.4223772771965522 ], ... ],
    	[ [ -1.385847693783549 ], [ -0.3381698464591904 ], [ 0.7537670837196926 ], [ -0.6628453000995471 ], [ -1.0891107365809303 ], [ 0.9520358880848309 ], [ -0.411634496650661 ], [ 1.189827303623695 ], ... ],
    	[ [ 1.792481922787085 ], [ 0.5544971853304848 ], [ -0.4167429671427342 ], [ -0.1047144537461819 ], [ 1.3569963146776045 ], [ 1.5191171642739674 ], [ 1.016497152419046 ], [ 1.5511057392319687 ], ... ],
    	[ [ -0.004436853601940964 ], [ 1.378924435699018 ], [ 1.6686220551813091 ], [ 0.36734257310432483 ], [ -0.5982373499208844 ], [ -0.38000931620330297 ], [ 1.9408339807113615 ], [ -1.9781428414863296 ], ... ],
    	[ [ 1.9322055498951842 ], [ -1.7323822561182414 ], [ -1.781008193273469 ], [ -1.0522842364325418 ], [ -0.8526388827514697 ], [ 0.07904578761731093 ], [ 1.2368286645859252 ], [ -1.8916513744451218 ], ... ],
    	[ [ -0.9539730663612678 ], [ -0.9952613132114196 ], [ -1.126904337659136 ], [ 0.9156568488458842 ], [ -0.9704354767141177 ], [ -1.2266857735708858 ], [ 0.7653549534570945 ], [ 0.1169012516789462 ], ... ],
    	...
    ]
```



Next, we run the same optimization using L-BFGS, which is nearly ideal for purely second-order or quadratic functions.

Code from [LearningTester.java:249](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L249) executed in 1.23 seconds: 
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
    th(0)=37.009161337074005;dx=-0.36082773420408104
    New Minimum: 37.009161337074005 > 36.23695871755596
    WOLFE (weak): th(2.154434690031884)=36.23695871755596; dx=-0.35602167700420634 delta=0.7722026195180476
    New Minimum: 36.23695871755596 > 35.475110434391844
    WOLFE (weak): th(4.308869380063768)=35.475110434391844; dx=-0.35121561980433164 delta=1.5340509026821607
    New Minimum: 35.475110434391844 > 32.531260665271205
    WOLFE (weak): th(12.926608140191302)=32.531260665271205; dx=-0.33199139100483277 delta=4.4779006718028
    New Minimum: 32.531260665271205 > 21.334095302259097
    END: th(51.70643256076521)=21.334095302259097; dx=-0.24548236140708787 delta=15.675066034814908
    Iteration 1 complete. Error: 21.334095302259097 Total: 239623423616430.6000; Orientation: 0.0005; Line Search: 0.0146
    LBFGS Accumulation History: 1 points
    th(0)=21.334095302259097;dx=-0.17188339736288022
    New Minimum: 21.334095302259097 > 7.962888919855285
    END: th(1
```
...[skipping 57127 bytes](etc/168.txt)...
```
    .15516854172063)=4.698563123302462E-4; dx=-2.238963838516635E-8 delta=4.99437828122858E-6
    New Minimum: 4.698563123302462E-4 > 4.6490274283338183E-4
    WOLFE (weak): th(444.31033708344125)=4.6490274283338183E-4; dx=-2.2205943427131318E-8 delta=9.947947778092938E-6
    New Minimum: 4.6490274283338183E-4 > 4.4549655268955523E-4
    WOLFE (weak): th(1332.9310112503238)=4.4549655268955523E-4; dx=-2.1471163594991167E-8 delta=2.9354137921919535E-5
    New Minimum: 4.4549655268955523E-4 > 3.6624883634610664E-4
    END: th(5331.724045001295)=3.6624883634610664E-4; dx=-1.8164654350360494E-8 delta=1.0860185426536813E-4
    Iteration 108 complete. Error: 3.6624883634610664E-4 Total: 239624622831142.3800; Orientation: 0.0005; Line Search: 0.0110
    LBFGS Accumulation History: 1 points
    th(0)=3.6624883634610664E-4;dx=-1.5745934559605677E-8
    MAX ALPHA: th(0)=3.6624883634610664E-4;th'(0)=-1.5745934559605677E-8;
    Iteration 109 failed, aborting. Error: 3.6624883634610664E-4 Total: 239624628922270.3800; Orientation: 0.0007; Line Search: 0.0039
    
```

Returns: 

```
    3.6624883634610664E-4
```



This training run resulted in the following regressed input:

Code from [LearningTester.java:154](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L154) executed in 0.00 seconds: 
```java
    return Arrays.stream(input_lbgfs).map(x -> x.prettyPrint()).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [
    	[ [ -0.14935377281500556 ], [ -0.5460769145663978 ], [ -0.10257021763945018 ], [ -0.7440638244838312 ], [ 1.5869144364748897 ], [ 0.6721037188141109 ], [ 0.7650964237930323 ], [ 1.6337388549866623 ], ... ],
    	[ [ -0.004009994776077245 ], [ 0.37462309557836654 ], [ -0.8261189300222889 ], [ 0.36627124838052677 ], [ -1.333178637910088 ], [ 1.0532715125628855 ], [ 1.9392324766322253 ], [ -0.2380932134139235 ], ... ],
    	[ [ 0.19644872769586488 ], [ -0.864636723861704 ], [ 0.16488797504642186 ], [ -0.06263533118787729 ], [ -0.23672566085567487 ], [ 1.877654312154274 ], [ 0.5819170729746274 ], [ -1.4179124964525431 ], ... ],
    	[ [ -1.3832074138509405 ], [ -0.3389552543208851 ], [ 0.7532872553267285 ], [ -0.6593925876411617 ], [ -1.0934057574017215 ], [ 0.9552633184012669 ], [ -0.4120769214476322 ], [ 1.186840208085674 ], ... ],
    	[ [ 1.7929485862445516 ], [ 0.5548059270752249 ], [ -0.4178701224688462 ], [ -0.10257972039930402 ], [ 1.353796883380932 ], [ 1.5214347199735212 ], [ 1.016990346469582 ], [ 1.5456259402496932 ], ... ],
    	[ [ -0.003767068294120419 ], [ 1.3802346280907376 ], [ 1.6647962607017697 ], [ 0.371699622800707 ], [ -0.6003110362551842 ], [ -0.3838949520425875 ], [ 1.9502240082137634 ], [ -1.9893976764263361 ], ... ],
    	[ [ 1.9289048638493642 ], [ -1.729280892129635 ], [ -1.7836414291238127 ], [ -1.055049838893824 ], [ -0.8457654628086292 ], [ 0.07052596362249956 ], [ 1.2427343926105168 ], [ -1.8925665707056298 ], ... ],
    	[ [ -0.9560019905261019 ], [ -0.9967870830840727 ], [ -1.1232556306746555 ], [ 0.911328122662517 ], [ -0.9669309606391568 ], [ -1.228170302327084 ], [ 0.7668457669622982 ], [ 0.11568593500205308 ], ... ],
    	...
    ]
```



Code from [LearningTester.java:95](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L95) executed in 0.00 seconds: 
```java
    return TestUtil.compare(runs);
```

Returns: 

![Result](etc/test.96.png)



Code from [LearningTester.java:98](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L98) executed in 0.00 seconds: 
```java
    return TestUtil.compareTime(runs);
```

Returns: 

![Result](etc/test.97.png)



### Model Learning
In this test, attempt to train a network to emulate a randomized network given an example input/output. The target state is:

Code from [LearningTester.java:176](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L176) executed in 0.00 seconds: 
```java
    return network_target.state().stream().map(Arrays::toString).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [-1.072, 1.992, 0.096, -0.704, -1.912, -0.536, 0.964, 1.904, 0.336]
```



First, we use a conjugate gradient descent method, which converges the fastest for purely linear functions.

Code from [LearningTester.java:225](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L225) executed in 0.79 seconds: 
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
    F(0.0) = LineSearchPoint{point=PointSample{avg=39.91132731520494}, derivative=-213.756165932497}
    New Minimum: 39.91132731520494 > 39.91132729382934
    F(1.0E-10) = LineSearchPoint{point=PointSample{avg=39.91132729382934}, derivative=-213.7561658752416}, delta = -2.1375605285811616E-8
    New Minimum: 39.91132729382934 > 39.91132716557587
    F(7.000000000000001E-10) = LineSearchPoint{point=PointSample{avg=39.91132716557587}, derivative=-213.75616553171065}, delta = -1.496290735758521E-7
    New Minimum: 39.91132716557587 > 39.9113262677999
    F(4.900000000000001E-9) = LineSearchPoint{point=PointSample{avg=39.9113262677999}, derivative=-213.75616312699364}, delta = -1.0474050426978465E-6
    New Minimum: 39.9113262677999 > 39.911319983368806
    F(3.430000000000001E-8) = LineSearchPoint{point=PointSample{avg=39.911319983368806}, derivative=-213.75614629397526}, delta = -7.331836137325354E-6
    New Minimum: 39.911319983368806 > 39.911275992366086
    F(2.4010000000000004E-7) = LineSearchPoint{
```
...[skipping 14731 bytes](etc/169.txt)...
```
    eSearchPoint{point=PointSample{avg=3.359658813452475E-30}, derivative=1.5951469035690682E-29}, delta = -1.65164172104922E-31
    3.359658813452475E-30 <= 3.524822985557397E-30
    New Minimum: 3.463293503168248E-32 > 8.965897225902563E-33
    F(0.382583794733383) = LineSearchPoint{point=PointSample{avg=8.965897225902563E-33}, derivative=3.388449264780605E-32}, delta = -3.515857088331495E-30
    Right bracket at 0.382583794733383
    Converged to right
    Iteration 12 complete. Error: 8.965897225902563E-33 Total: 239625537917270.5300; Orientation: 0.0001; Line Search: 0.0605
    Zero gradient: 3.520801526440956E-17
    F(0.0) = LineSearchPoint{point=PointSample{avg=8.965897225902563E-33}, derivative=-1.2396043388588967E-33}
    New Minimum: 8.965897225902563E-33 > 0.0
    F(0.382583794733383) = LineSearchPoint{point=PointSample{avg=0.0}, derivative=0.0}, delta = -8.965897225902563E-33
    0.0 <= 8.965897225902563E-33
    Converged to right
    Iteration 13 complete. Error: 0.0 Total: 239625556822167.4700; Orientation: 0.0001; Line Search: 0.0135
    
```

Returns: 

```
    0.0
```



Training Converged

Next, we run the same optimization using L-BFGS, which is nearly ideal for purely second-order or quadratic functions.

Code from [LearningTester.java:249](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L249) executed in 0.66 seconds: 
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
    th(0)=39.53045808874145;dx=-210.62760302524623
    Armijo: th(2.154434690031884)=888.5796212237332; dx=998.8150259843746 delta=-849.0491631349918
    Armijo: th(1.077217345015942)=138.346895213525; dx=394.0937114795667 delta=-98.81643712478353
    New Minimum: 39.53045808874145 > 0.0896828097334288
    END: th(0.3590724483386473)=0.0896828097334288; dx=-9.053831523642618 delta=39.440775279008015
    Iteration 1 complete. Error: 0.0896828097334288 Total: 239625601576028.4400; Orientation: 0.0001; Line Search: 0.0274
    LBFGS Accumulation History: 1 points
    th(0)=0.0896828097334288;dx=-0.469086223013145
    Armijo: th(0.7735981389354633)=0.09407567780682526; dx=0.4804431998524147 delta=-0.00439286807339645
    New Minimum: 0.0896828097334288 > 5.9969470969406915E-5
    WOLF (strong): th(0.3867990694677316)=5.9969470969406915E-5; dx=0.005678488419634698 delta=0.0896228402624594
    END: th(0.12893302315591054)=0.03940424646835189; dx=-0.31083131920221885 delta=0.
```
...[skipping 9654 bytes](etc/170.txt)...
```
    1 points
    th(0)=1.5181067577057963E-30;dx=-5.6707832760009476E-30
    New Minimum: 1.5181067577057963E-30 > 7.581789568248265E-31
    WOLF (strong): th(0.6279337062757206)=7.581789568248265E-31; dx=3.236826831615475E-30 delta=7.599278008809698E-31
    New Minimum: 7.581789568248265E-31 > 1.9215425627973256E-31
    END: th(0.3139668531378603)=1.9215425627973256E-31; dx=-7.888645514359285E-31 delta=1.3259525014260638E-30
    Iteration 21 complete. Error: 1.9215425627973256E-31 Total: 239626192160968.8400; Orientation: 0.0001; Line Search: 0.0228
    LBFGS Accumulation History: 1 points
    th(0)=1.9215425627973256E-31;dx=-1.8901718330702618E-31
    New Minimum: 1.9215425627973256E-31 > 1.767450619196314E-31
    WOLF (strong): th(0.676421079920352)=1.767450619196314E-31; dx=1.8212232582783148E-31 delta=1.5409194360101154E-32
    New Minimum: 1.767450619196314E-31 > 0.0
    END: th(0.338210539960176)=0.0; dx=0.0 delta=1.9215425627973256E-31
    Iteration 22 complete. Error: 0.0 Total: 239626222841171.8000; Orientation: 0.0001; Line Search: 0.0213
    
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

![Result](etc/test.98.png)



Code from [LearningTester.java:98](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L98) executed in 0.00 seconds: 
```java
    return TestUtil.compareTime(runs);
```

Returns: 

![Result](etc/test.99.png)



### Performance
Now we execute larger-scale runs to benchmark performance:

Code from [PerformanceTester.java:66](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/PerformanceTester.java#L66) executed in 0.48 seconds: 
```java
    test(component, inputPrototype);
```
Logging: 
```
    100 batches
    Input Dimensions:
    	[100, 100, 1]
    Performance:
    	Evaluation performance: 0.016450s +- 0.001285s [0.014853s - 0.018243s]
    	Learning performance: 0.059942s +- 0.016065s [0.040149s - 0.077772s]
    
```

