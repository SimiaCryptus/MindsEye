# ImgBandBiasLayer
## Float
### Batch Execution
Code from [BatchingTester.java:66](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/BatchingTester.java#L66) executed in 0.01 seconds: 
```java
    return test(reference, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (360#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (360#)}
```



Code from [SingleDerivativeTester.java:77](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/SingleDerivativeTester.java#L77) executed in 0.08 seconds: 
```java
    return test(component, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ 0.44, 0.944 ], [ 1.464, -0.832 ], [ -0.508, 1.764 ] ],
    	[ [ -0.844, 1.84 ], [ 1.12, -1.944 ], [ -0.68, -1.664 ] ],
    	[ [ 1.544, -0.176 ], [ -1.576, 0.116 ], [ -1.204, 1.42 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.04620682489262066, negative=9, min=1.42, max=1.42, mean=0.068, count=18.0, positive=9, stdDev=1.2460540384215553, zeros=0}
    Output: [
    	[ [ 2.032, 2.136 ], [ 3.056, 0.36 ], [ 1.084, 2.956 ] ],
    	[ [ 0.7480000000000001, 3.032 ], [ 2.712, -0.752 ], [ 0.912, -0.472 ] ],
    	[ [ 3.136, 1.016 ], [ 0.016000000000000014, 1.308 ], [ 0.3880000000000001, 2.612 ] ]
    ]
    Outputs Statistics: {meanExponent=0.017421781052214688, negative=2, min=2.612, max=2.612, mean=1.46, count=18.0, positive=16, stdDev=1.2468384908328034, zeros=0}
    Feedback for input 0
    Inputs Values: [
    	[ [ 0.44, 0.944 ], [ 1.464, -0.832 ], [ -0.508, 1.764 ] ],
    	[ [ -0.844, 1.84 ], [ 1.12, -1.944 ], [ -0.68, -1.664 ] ],
    	[ [ 1.544, -0.176 ], [ -1.576, 0.116 ], [ -1.204, 1.42 ] ]
    ]
    Value Statistics: {meanExponent=-0.04620682489262066, n
```
...[skipping 2653 bytes](etc/105.txt)...
```
     0.9999999999998899, 0.9999999999998899, 0.9999999999998899, 0.9999999999998899, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ] ]
    Measured Statistics: {meanExponent=-6.92601415977925E-14, negative=0, min=0.9999999999994458, max=0.9999999999994458, mean=0.4999999999999203, count=36.0, positive=18, stdDev=0.49999999999992023, zeros=18}
    Gradient Error: [ [ -1.1013412404281553E-13, -1.1013412404281553E-13, -1.1013412404281553E-13, -1.1013412404281553E-13, -1.1013412404281553E-13, -1.1013412404281553E-13, -1.1013412404281553E-13, -1.1013412404281553E-13, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ] ]
    Error Statistics: {meanExponent=-12.880104441979023, negative=18, min=-5.542233338928781E-13, max=-5.542233338928781E-13, mean=-7.973868479085568E-14, count=36.0, positive=0, stdDev=1.2687506206273358E-13, zeros=18}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 1.3804e-14 +- 5.8137e-14 [0.0000e+00 - 5.5422e-13] (360#)
    relativeTol: 6.9019e-14 +- 6.4519e-14 [4.4409e-16 - 2.7711e-13] (36#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=1.3804e-14 +- 5.8137e-14 [0.0000e+00 - 5.5422e-13] (360#), relativeTol=6.9019e-14 +- 6.4519e-14 [4.4409e-16 - 2.7711e-13] (36#)}
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
      "class": "com.simiacryptus.mindseye.layers.cudnn.ImgBandBiasLayer",
      "id": "cf252433-3b02-45ff-9183-a67f714c198d",
      "isFrozen": false,
      "name": "ImgBandBiasLayer/cf252433-3b02-45ff-9183-a67f714c198d",
      "bias": [
        1.592,
        1.192
      ]
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
    	[ [ -0.344, 0.004 ], [ 1.376, 0.428 ], [ -0.736, 0.548 ] ],
    	[ [ -0.204, -0.684 ], [ 0.288, 1.636 ], [ 0.176, -0.332 ] ],
    	[ [ -1.78, 0.128 ], [ 0.052, -1.136 ], [ 0.624, -0.74 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 1.2480000000000002, 1.196 ], [ 2.968, 1.6199999999999999 ], [ 0.8560000000000001, 1.74 ] ],
    	[ [ 1.3880000000000001, 0.5079999999999999 ], [ 1.8800000000000001, 2.828 ], [ 1.768, 0.8599999999999999 ] ],
    	[ [ -0.18799999999999994, 1.3199999999999998 ], [ 1.6440000000000001, 0.05600000000000005 ], [ 2.216, 0.45199999999999996 ] ]
    ]
    --------------------
    Derivative: 
    [
    	[ [ 1.0, 1.0 ], [ 1.0, 1.0 ], [ 1.0, 1.0 ] ],
    	[ [ 1.0, 1.0 ], [ 1.0, 1.0 ], [ 1.0, 1.0 ] ],
    	[ [ 1.0, 1.0 ], [ 1.0, 1.0 ], [ 1.0, 1.0 ] ]
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
    	[ [ 1.396, -0.932 ], [ -1.544, 0.18 ], [ 0.5, 1.396 ], [ -1.164, 1.152 ], [ 1.66, -0.076 ], [ -1.992, 0.048 ], [ 1.16, 0.348 ], [ 1.38, 0.72 ], ... ],
    	[ [ -1.464, -1.668 ], [ 0.712, 0.444 ], [ -1.804, -1.576 ], [ 0.656, -0.504 ], [ -0.896, -1.136 ], [ 1.108, 1.092 ], [ -0.932, -0.812 ], [ 1.948, 1.364 ], ... ],
    	[ [ -1.472, -1.148 ], [ 1.716, -1.768 ], [ -1.564, 0.812 ], [ -1.34, 0.408 ], [ 0.376, -1.004 ], [ 1.812, -1.848 ], [ -0.612, -1.432 ], [ 1.84, -0.96 ], ... ],
    	[ [ 1.852, 0.932 ], [ -0.56, -1.484 ], [ -1.836, -1.768 ], [ 1.656, 1.168 ], [ 1.28, -0.092 ], [ -0.524, -0.116 ], [ -1.304, -1.664 ], [ 1.144, 0.48 ], ... ],
    	[ [ -1.024, -1.3 ], [ -0.792, -1.084 ], [ 0.468, -1.332 ], [ 1.784, -0.54 ], [ -1.024, 0.776 ], [ -0.972, -1.508 ], [ 1.052, 1.16 ], [ -0.712, 1.796 ], ... ],
    	[ [ -1.924, -0.372 ], [ 0.956, -1.064 ], [ -0.3, 1.148 ], [ 0.516, 1.112 ], [ 0.184, 0.42 ], [ -1.392, -0.052 ], [ -1.872, 0.728 ], [ -0.632, 0.376 ], ... ],
    	[ [ 0.552, 1.144 ], [ 0.74, 1.296 ], [ -1.356, -1.264 ], [ 0.336, 1.104 ], [ 0.204, -0.268 ], [ 0.272, -1.572 ], [ 0.684, 1.508 ], [ -1.532, -1.608 ], ... ],
    	[ [ 0.712, 1.476 ], [ 0.74, -1.504 ], [ -0.108, -1.42 ], [ -1.784, -0.132 ], [ -0.72, 1.26 ], [ -1.94, -0.244 ], [ 1.476, -0.996 ], [ 1.868, -0.348 ], ... ],
    	...
    ]
```



First, we use a conjugate gradient descent method, which converges the fastest for purely linear functions.

Code from [LearningTester.java:225](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L225) executed in 0.07 seconds: 
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
    F(0.0) = LineSearchPoint{point=PointSample{avg=2.654545324800015}, derivative=-5.3090906496E-4}
    New Minimum: 2.654545324800015 > 2.6545453247999666
    F(1.0E-10) = LineSearchPoint{point=PointSample{avg=2.6545453247999666}, derivative=-5.309090649599947E-4}, delta = -4.8405723873656825E-14
    New Minimum: 2.6545453247999666 > 2.654545324799631
    F(7.000000000000001E-10) = LineSearchPoint{point=PointSample{avg=2.654545324799631}, derivative=-5.309090649599628E-4}, delta = -3.8413716652030416E-13
    New Minimum: 2.654545324799631 > 2.6545453247973665
    F(4.900000000000001E-9) = LineSearchPoint{point=PointSample{avg=2.6545453247973665}, derivative=-5.309090649597399E-4}, delta = -2.6485480475457734E-12
    New Minimum: 2.6545453247973665 > 2.6545453247818425
    F(3.430000000000001E-8) = LineSearchPoint{point=PointSample{avg=2.6545453247818425}, derivative=-5.30909064958179E-4}, delta = -1.8172574556274412E-11
    New Minimum: 2.6545453247818425 > 2.654545324672551
    F(2.4010000000000004E-
```
...[skipping 2406 bytes](etc/106.txt)...
```
    =PointSample{avg=9.426887817391092E-33}, derivative=1.7044831297448895E-34}, delta = -3.054084867428547E-25
    9.426887817391092E-33 <= 3.0540849616974254E-25
    Converged to right
    Iteration 2 complete. Error: 9.426887817391092E-33 Total: 239573208127220.8000; Orientation: 0.0006; Line Search: 0.0042
    Zero gradient: 1.3730905153988276E-18
    F(0.0) = LineSearchPoint{point=PointSample{avg=9.426887817391092E-33}, derivative=-1.8853775634782182E-36}
    New Minimum: 9.426887817391092E-33 > 9.399154426191914E-33
    F(9999.999999996608) = LineSearchPoint{point=PointSample{avg=9.399154426191914E-33}, derivative=1.879830885238383E-36}, delta = -2.7733391199177247E-35
    9.399154426191914E-33 <= 9.426887817391092E-33
    New Minimum: 9.399154426191914E-33 > 0.0
    F(5007.36569875759) = LineSearchPoint{point=PointSample{avg=0.0}, derivative=0.0}, delta = -9.426887817391092E-33
    Right bracket at 5007.36569875759
    Converged to right
    Iteration 3 complete. Error: 0.0 Total: 239573218031074.7800; Orientation: 0.0006; Line Search: 0.0080
    
```

Returns: 

```
    0.0
```



Training Converged

Next, we run the same optimization using L-BFGS, which is nearly ideal for purely second-order or quadratic functions.

Code from [LearningTester.java:249](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L249) executed in 0.04 seconds: 
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
    th(0)=2.654545324800015;dx=-5.3090906496E-4
    New Minimum: 2.654545324800015 > 2.6534016391062663
    WOLFE (weak): th(2.154434690031884)=2.6534016391062663; dx=-5.307946840693199E-4 delta=0.001143685693748786
    New Minimum: 2.6534016391062663 > 2.652258199838684
    WOLFE (weak): th(4.308869380063768)=2.652258199838684; dx=-5.306803031786396E-4 delta=0.0022871249613309885
    New Minimum: 2.652258199838684 > 2.6476869070300477
    WOLFE (weak): th(12.926608140191302)=2.6476869070300477; dx=-5.302227796159187E-4 delta=0.006858417769967318
    New Minimum: 2.6476869070300477 > 2.62716488177045
    WOLFE (weak): th(51.70643256076521)=2.62716488177045; dx=-5.281639235836747E-4 delta=0.027380443029564994
    New Minimum: 2.62716488177045 > 2.5190625243267997
    WOLFE (weak): th(258.53216280382605)=2.5190625243267997; dx=-5.171833580783734E-4 delta=0.13548280047321537
    New Minimum: 2.5190625243267997 > 1.8948765722525358
    END: th(1551.1929768229563)=1.894876572
```
...[skipping 167 bytes](etc/107.txt)...
```
    rch: 0.0180
    LBFGS Accumulation History: 1 points
    th(0)=1.8948765722525358;dx=-3.789753144505069E-4
    New Minimum: 1.8948765722525358 > 0.8399932936786776
    END: th(3341.943960201201)=0.8399932936786776; dx=-2.5232388813118463E-4 delta=1.0548832785738582
    Iteration 2 complete. Error: 0.8399932936786776 Total: 239573248094299.7500; Orientation: 0.0010; Line Search: 0.0044
    LBFGS Accumulation History: 1 points
    th(0)=0.8399932936786776;dx=-1.6799865873573503E-4
    New Minimum: 0.8399932936786776 > 0.06585547422440763
    END: th(7200.000000000001)=0.06585547422440763; dx=-4.703962444600579E-5 delta=0.77413781945427
    Iteration 3 complete. Error: 0.06585547422440763 Total: 239573255390886.7500; Orientation: 0.0010; Line Search: 0.0046
    LBFGS Accumulation History: 1 points
    th(0)=0.06585547422440763;dx=-1.3171094844881614E-5
    MAX ALPHA: th(0)=0.06585547422440763;th'(0)=-1.3171094844881614E-5;
    Iteration 4 failed, aborting. Error: 0.06585547422440763 Total: 239573261985001.7500; Orientation: 0.0010; Line Search: 0.0041
    
```

Returns: 

```
    0.06585547422440763
```



This training run resulted in the following regressed input:

Code from [LearningTester.java:154](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L154) executed in 0.01 seconds: 
```java
    return Arrays.stream(input_lbgfs).map(x -> x.prettyPrint()).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [
    	[ [ 1.0425534712271018, -0.7663022512169837 ], [ -1.4513856689311657, 0.19512070711327895 ], [ 0.6537271889850038, 0.9436388455277349 ], [ -1.1955014731526648, 0.8243846792122871 ], [ 1.2460706427739856, -0.1705044194579942 ], [ -1.9189165822858179, -0.19330128434941163 ], [ 1.1203081438276423, 0.04495582827136557 ], [ 1.1418488629658547, 0.7792227695270095 ], ... ],
    	[ [ -1.3751658457094855, -1.7171422981181568 ], [ 0.7750029463053296, 0.23735033611851952 ], [ -1.4215721159266506, -1.3504494522269206 ], [ 0.3069636774684753, -0.49265946966504076 ], [ -0.5400333533748888, -0.6704082268036156 ], [ 0.6682394347888007, 0.9590637832957548 ], [ -0.6138351211580864, -0.42516190968527745 ], [ 1.8534955805420057, 0.9154190223060545 ], ... ],
    	[ [ -1.242669275448601, -1.1177585857734418 ], [ 1.2478881089514022, -1.323829228547428 ], [ -1.24457506223198, 0.8056997053694671 ], [ -0.875038256266669, 0.5365260104628721 ], [ 0.21534248692141, -1.1010245373102072 ], [ 1.4793444435078607, -1.3049146028480607 ], [ -0.32533
```
...[skipping 926 bytes](etc/108.txt)...
```
    0.5413012843494117, 1.1580804714088526 ], [ 0.24886750766540333, 0.7875348265275538 ], [ 0.1014661403400185, 0.41621982322168016 ], [ -1.255283606517435, -0.043809616980307206 ], [ -1.5884867416260176, 0.67633758402963 ], [ -0.3869185388722686, 0.08240627021716507 ], ... ],
    	[ [ 0.3881923396061435, 0.8825377728328829 ], [ 0.3393012614981048, 1.0616290397441746 ], [ -1.4435740953644078, -1.1241334592021688 ], [ 0.5319391630095744, 0.6925907606261993 ], [ 0.3463866586500443, -0.3360431820097558 ], [ 0.3419332703989158, -1.6079116793940378 ], [ 0.7898449497929536, 1.0871403186803998 ], [ -1.461436700138031, -1.5027850796700999 ], ... ],
    	[ [ 0.6880588804039751, 1.3310932234977426 ], [ 0.3607222632419168, -1.2494680969264693 ], [ -0.22581550959096613, -0.9966202008281864 ], [ -1.3851913498872648, -0.03434543322673936 ], [ -0.38671441404480744, 0.8775721159266506 ], [ -1.7402806602121057, 0.043923464615355456 ], [ 1.1023925284093967, -0.7584788924289081 ], [ 1.3860274607642298, -0.5641001058272799 ], ... ],
    	...
    ]
```



Code from [LearningTester.java:95](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L95) executed in 0.00 seconds: 
```java
    return TestUtil.compare(runs);
```

Returns: 

![Result](etc/test.69.png)



Code from [LearningTester.java:98](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L98) executed in 0.00 seconds: 
```java
    return TestUtil.compareTime(runs);
```

Returns: 

![Result](etc/test.70.png)



### Model Learning
In this test, attempt to train a network to emulate a randomized network given an example input/output. The target state is:

Code from [LearningTester.java:176](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L176) executed in 0.00 seconds: 
```java
    return network_target.state().stream().map(Arrays::toString).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [1.592, 1.192]
```



First, we use a conjugate gradient descent method, which converges the fastest for purely linear functions.

Code from [LearningTester.java:225](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L225) executed in 0.00 seconds: 
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

Returns: 

```
    0.0
```



This training run resulted in the following configuration:

Code from [LearningTester.java:189](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L189) executed in 0.00 seconds: 
```java
    return network_gd.state().stream().map(Arrays::toString).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [1.092, 0.8360000000000001, 1.596, 1.856, 0.15200000000000014, 2.8840000000000003, 0.32400000000000007, 1.6, 1.6320000000000001, -0.22799999999999998, 2.92, 1.9080000000000001, 0.484, 3.16, 2.6, 2.8920000000000003, 2.672, 1.4080000000000001, 1.6840000000000002, 0.03600000000000003, 1.588, 1.056, 3.14, 1.86, 0.4560000000000002, 3.2359999999999998, 2.856, 3.292, 0.788, 2.4000000000000004, 1.624, 3.3680000000000003, 3.3040000000000003, 3.048, 1.324, 2.68, 3.5, 0.20000000000000018, 2.348, 0.07200000000000006, 2.7279999999999998, -0.35199999999999987, 2.716, 2.704, 0.8800000000000001, 0.6040000000000001, 1.344, 1.3800000000000001, 1.568, 2.748, 1.872, 1.264, 1.1560000000000001, 1.6520000000000001, 0.040000000000000036, 0.10000000000000009, 3.156, 3.3280000000000003, 2.172, -0.3679999999999999, 2.484, 0.20400000000000018, 1.86, 1.112, 2.652, 1.04, 3.084, 2.676, -0.10799999999999987, 0.3360000000000001, 3.252, -0.11999999999999988, 0.4520000000000002, 0.1160000000000001, 1.78, 3.084, 2.612, 2.396, -0.095999999999999
```
...[skipping 256542 bytes](etc/109.txt)...
```
     2.964, 1.7719999999999998, -0.06000000000000005, 0.824, 2.26, -0.45599999999999996, 0.08799999999999986, 0.5239999999999999, 2.824, -0.17600000000000016, 2.48, -0.22799999999999998, 0.8759999999999999, 1.26, 0.2759999999999999, -0.29200000000000004, 2.472, 0.5119999999999999, 2.328, 1.548, 2.448, 0.768, 0.43999999999999995, 2.316, 0.62, 0.5519999999999999, 1.412, 2.7119999999999997, -0.6200000000000001, 0.8759999999999999, 2.092, 2.404, -0.44799999999999995, 0.948, 2.152, 2.076, 0.16799999999999993, 2.62, 0.040000000000000036, 2.84, -0.6600000000000001, 1.588, 3.056, 1.548, 1.3519999999999999, 0.008000000000000007, -0.52, 2.208, -0.16400000000000015, 1.0999999999999999, -0.31600000000000006, -0.41200000000000014, 1.944, -0.736, 2.692, 2.6559999999999997, 1.236, 1.228, 2.66, 0.6719999999999999, 0.17999999999999994, 0.42799999999999994, 0.040000000000000036, 1.676, 2.508, -0.14000000000000012, 2.9, 2.46, -0.06000000000000005, 0.05600000000000005, -0.42000000000000015, -0.768, 1.7599999999999998]
    [1.592, 1.192]
```



Next, we run the same optimization using L-BFGS, which is nearly ideal for purely second-order or quadratic functions.

Code from [LearningTester.java:249](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L249) executed in 0.00 seconds: 
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

Returns: 

```
    0.0
```



This training run resulted in the following configuration:

Code from [LearningTester.java:203](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L203) executed in 0.00 seconds: 
```java
    return network_lbfgs.state().stream().map(Arrays::toString).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [1.092, 0.8360000000000001, 1.596, 1.856, 0.15200000000000014, 2.8840000000000003, 0.32400000000000007, 1.6, 1.6320000000000001, -0.22799999999999998, 2.92, 1.9080000000000001, 0.484, 3.16, 2.6, 2.8920000000000003, 2.672, 1.4080000000000001, 1.6840000000000002, 0.03600000000000003, 1.588, 1.056, 3.14, 1.86, 0.4560000000000002, 3.2359999999999998, 2.856, 3.292, 0.788, 2.4000000000000004, 1.624, 3.3680000000000003, 3.3040000000000003, 3.048, 1.324, 2.68, 3.5, 0.20000000000000018, 2.348, 0.07200000000000006, 2.7279999999999998, -0.35199999999999987, 2.716, 2.704, 0.8800000000000001, 0.6040000000000001, 1.344, 1.3800000000000001, 1.568, 2.748, 1.872, 1.264, 1.1560000000000001, 1.6520000000000001, 0.040000000000000036, 0.10000000000000009, 3.156, 3.3280000000000003, 2.172, -0.3679999999999999, 2.484, 0.20400000000000018, 1.86, 1.112, 2.652, 1.04, 3.084, 2.676, -0.10799999999999987, 0.3360000000000001, 3.252, -0.11999999999999988, 0.4520000000000002, 0.1160000000000001, 1.78, 3.084, 2.612, 2.396, -0.095999999999999
```
...[skipping 256542 bytes](etc/110.txt)...
```
     2.964, 1.7719999999999998, -0.06000000000000005, 0.824, 2.26, -0.45599999999999996, 0.08799999999999986, 0.5239999999999999, 2.824, -0.17600000000000016, 2.48, -0.22799999999999998, 0.8759999999999999, 1.26, 0.2759999999999999, -0.29200000000000004, 2.472, 0.5119999999999999, 2.328, 1.548, 2.448, 0.768, 0.43999999999999995, 2.316, 0.62, 0.5519999999999999, 1.412, 2.7119999999999997, -0.6200000000000001, 0.8759999999999999, 2.092, 2.404, -0.44799999999999995, 0.948, 2.152, 2.076, 0.16799999999999993, 2.62, 0.040000000000000036, 2.84, -0.6600000000000001, 1.588, 3.056, 1.548, 1.3519999999999999, 0.008000000000000007, -0.52, 2.208, -0.16400000000000015, 1.0999999999999999, -0.31600000000000006, -0.41200000000000014, 1.944, -0.736, 2.692, 2.6559999999999997, 1.236, 1.228, 2.66, 0.6719999999999999, 0.17999999999999994, 0.42799999999999994, 0.040000000000000036, 1.676, 2.508, -0.14000000000000012, 2.9, 2.46, -0.06000000000000005, 0.05600000000000005, -0.42000000000000015, -0.768, 1.7599999999999998]
    [1.592, 1.192]
```



Code from [LearningTester.java:95](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L95) executed in 0.00 seconds: 
```java
    return TestUtil.compare(runs);
```

Code from [LearningTester.java:98](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L98) executed in 0.00 seconds: 
```java
    return TestUtil.compareTime(runs);
```

### Performance
Now we execute larger-scale runs to benchmark performance:

Code from [PerformanceTester.java:66](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/PerformanceTester.java#L66) executed in 0.49 seconds: 
```java
    test(component, inputPrototype);
```
Logging: 
```
    100 batches
    Input Dimensions:
    	[100, 100, 2]
    Performance:
    	Evaluation performance: 0.018402s +- 0.011363s [0.011811s - 0.041105s]
    	Learning performance: 0.065620s +- 0.017234s [0.053805s - 0.099411s]
    
```

