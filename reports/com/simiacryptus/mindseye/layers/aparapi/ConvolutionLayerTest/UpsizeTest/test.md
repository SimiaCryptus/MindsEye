# ConvolutionLayer
## UpsizeTest
### Batch Execution
Code from [BatchingTester.java:66](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/BatchingTester.java#L66) executed in 0.09 seconds: 
```java
    return test(reference, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (210#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (50#)}
```



Code from [SingleDerivativeTester.java:77](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/SingleDerivativeTester.java#L77) executed in 0.34 seconds: 
```java
    return test(component, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ 0.296, 1.312 ], [ -1.932, -0.556 ], [ 0.824, -1.636 ] ],
    	[ [ 1.528, 1.096 ], [ 1.312, -1.272 ], [ -1.248, -1.312 ] ],
    	[ [ 0.596, 0.208 ], [ -1.864, 0.524 ], [ -0.032, -1.46 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.10204939612624073, negative=9, min=-1.46, max=-1.46, mean=-0.20088888888888887, count=18.0, positive=9, stdDev=1.1764855804418743, zeros=0}
    Output: [
    	[ [ 0.46963200000000016, -0.21094400000000002, -0.80432 ] ]
    ]
    Outputs Statistics: {meanExponent=-0.36621542456773937, negative=2, min=-0.80432, max=-0.80432, mean=-0.1818773333333333, count=3.0, positive=1, stdDev=0.5204946864341866, zeros=0}
    Feedback for input 0
    Inputs Values: [
    	[ [ 0.296, 1.312 ], [ -1.932, -0.556 ], [ 0.824, -1.636 ] ],
    	[ [ 1.528, 1.096 ], [ 1.312, -1.272 ], [ -1.248, -1.312 ] ],
    	[ [ 0.596, 0.208 ], [ -1.864, 0.524 ], [ -0.032, -1.46 ] ]
    ]
    Value Statistics: {meanExponent=-0.10204939612624073, negative=9, min=-1.46, max=-1.46, mean=-0.20088888888888887, count=18.0, positive=9, stdDev=1.1764855804418743, ze
```
...[skipping 1557 bytes](etc/14.txt)...
```
    0.0, mean=0.029777777777777778, count=162.0, positive=6, stdDev=0.18058943271929437, zeros=156}
    Measured Gradient: [ [ 0.296000000000185, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], ... ]
    Measured Statistics: {meanExponent=-0.2053872269508272, negative=0, min=0.0, max=0.0, mean=0.02977777777777055, count=162.0, positive=6, stdDev=0.1805894327192527, zeros=156}
    Gradient Error: [ [ 1.8501866705378234E-13, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], ... ]
    Error Statistics: {meanExponent=-12.776483278228902, negative=5, min=0.0, max=0.0, mean=-7.22775748716618E-15, count=162.0, positive=1, stdDev=6.041836998767345E-14, zeros=156}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 1.1625e-14 +- 6.1303e-14 [0.0000e+00 - 5.7532e-13] (216#)
    relativeTol: 3.9347e-13 +- 4.2401e-13 [7.7005e-15 - 1.3674e-12] (12#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=1.1625e-14 +- 6.1303e-14 [0.0000e+00 - 5.7532e-13] (216#), relativeTol=3.9347e-13 +- 4.2401e-13 [7.7005e-15 - 1.3674e-12] (12#)}
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
      "class": "com.simiacryptus.mindseye.layers.aparapi.ConvolutionLayer",
      "id": "5e9ce124-a298-4c62-85b3-965e606cf015",
      "isFrozen": false,
      "name": "ConvolutionLayer/5e9ce124-a298-4c62-85b3-965e606cf015",
      "filter": [
        [
          [
            -1.392,
            -0.968,
            -0.072
          ],
          [
            -0.7,
            -1.436,
            0.668
          ],
          [
            -0.08,
            -1.536,
            -0.516
          ]
        ],
        [
          [
            0.032,
            1.468,
            -1.86
          ],
          [
            -0.544,
            1.696,
            -1.004
          ],
          [
            0.628,
            0.444,
            -0.656
          ]
        ],
        [
          [
            0.084,
            -1.224,
            1.712
          ],
          [
            -1.012,
            -1.22,
            -1.408
          ],
          [
            -1.988,
            -1.192,
            -1.428
          ]
        ],
        [
          [
            0.672,
            1.84,
            1.38
          ],
          [
            0.908,
            -1.064,
            -0.228
          ],
          [
            0.16,
            1.52,
            -0.436
          ]
        ],
        [
          [
            -0.168,
            1.968,
            1.244
          ],
          [
            -1.272,
            1.924,
            -0.308
          ],
          [
            0.624,
            0.428,
            -1.572
          ]
        ],
        [
          [
            -0.632,
            0.364,
            1.62
          ],
          [
            -0.584,
            0.684,
            0.484
          ],
          [
            0.852,
            0.364,
            -1.992
          ]
        ]
      ],
      "skip": [
        [
          0.0
        ]
      ],
      "simple": false
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
    	[ [ -0.688, 0.088 ], [ -0.276, 1.9 ], [ -1.812, 0.268 ] ],
    	[ [ -0.624, -1.468 ], [ 1.256, 1.708 ], [ -1.756, 0.324 ] ],
    	[ [ 1.208, 0.976 ], [ 1.072, -1.156 ], [ -1.672, 0.232 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 1.016832, -0.0368, -0.113408 ] ]
    ]
    --------------------
    Derivative: 
    [
    	[ [ -1.2759999999999998, -0.128 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ] ],
    	[ [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ] ],
    	[ [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ] ]
    ]
```



### Input Learning
In this test, we use a network to learn this target input, given it's pre-evaluated output:

Code from [LearningTester.java:127](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L127) executed in 0.01 seconds: 
```java
    return Arrays.stream(input_target).map(x -> x.prettyPrint()).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [
    	[ [ 0.292, 0.708, -0.508 ], [ -1.04, 0.164, -0.724 ], [ -1.904, -0.416, 0.248 ], [ -0.432, -1.812, -1.596 ], [ -0.404, -1.252, -0.04 ], [ -0.64, -1.112, -1.392 ], [ 1.692, -1.132, 1.836 ], [ 0.232, 1.936, 0.316 ], ... ],
    	[ [ -0.216, 1.888, 1.576 ], [ -0.168, 1.352, -1.716 ], [ -1.424, 1.372, -1.492 ], [ -1.208, 0.428, -0.696 ], [ 0.076, -0.696, 1.352 ], [ 1.808, 0.152, -1.504 ], [ 0.044, 0.924, -0.016 ], [ 1.112, -1.196, 0.076 ], ... ],
    	[ [ 1.692, 1.764, -1.824 ], [ 0.404, -0.776, 1.06 ], [ -0.94, 0.824, 1.264 ], [ 1.324, 0.748, -1.956 ], [ -0.964, -1.776, 1.844 ], [ 1.436, 0.576, -1.84 ], [ -1.792, 0.204, -1.708 ], [ 1.392, -1.056, 0.112 ], ... ],
    	[ [ 1.176, -1.372, 1.068 ], [ 0.32, -1.024, 1.536 ], [ -1.532, 1.452, 0.684 ], [ 1.668, 1.136, 1.924 ], [ 1.476, 0.364, -0.144 ], [ -0.724, 0.168, -0.068 ], [ 0.2, -0.872, 1.428 ], [ -1.284, -0.688, 1.996 ], ... ],
    	[ [ 0.108, -0.684, -1.656 ], [ -1.876, -1.588, -1.12 ], [ -1.248, -0.908, -1.04 ], [ 1.128, -0.784, -1.244 ], [ 1.344, -0.252, -0.88 ], [ 0.992, 0.084, -0.292 ], [ -0.412, -1.336, 0.596 ], [ 1.516, 1.128, 1.216 ], ... ],
    	[ [ -0.276, 0.504, -1.044 ], [ 1.264, 0.1, -1.7 ], [ 1.448, 1.512, -1.428 ], [ -0.552, -1.78, 1.364 ], [ -1.692, 0.96, -0.392 ], [ -1.532, -1.784, -0.992 ], [ 1.2, 1.984, -0.736 ], [ 1.288, -0.068, -1.184 ], ... ],
    	[ [ -1.72, 0.216, 0.932 ], [ -0.752, -1.264, 1.452 ], [ 1.5, -0.432, 1.612 ], [ 1.124, -0.408, 1.88 ], [ -1.468, 0.572, -0.768 ], [ -1.676, -0.828, 0.436 ], [ -1.216, 0.556, 0.468 ], [ -0.868, 1.776, 0.356 ], ... ],
    	[ [ 0.364, 1.624, 1.388 ], [ -0.08, -1.548, -0.444 ], [ 0.368, 1.252, 1.824 ], [ 0.468, 1.912, 0.196 ], [ 0.076, 0.788, -0.792 ], [ 0.224, 1.128, -1.052 ], [ 1.236, 1.384, -1.648 ], [ 1.216, -0.108, -0.1 ], ... ],
    	...
    ]
```



First, we use a conjugate gradient descent method, which converges the fastest for purely linear functions.

Code from [LearningTester.java:225](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L225) executed in 17.65 seconds: 
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
    F(0.0) = LineSearchPoint{point=PointSample{avg=89.57919140108089}, derivative=-0.967333358361435}
    New Minimum: 89.57919140108089 > 89.57919140098461
    F(1.0E-10) = LineSearchPoint{point=PointSample{avg=89.57919140098461}, derivative=-0.9673333583608168}, delta = -9.627854069549358E-11
    New Minimum: 89.57919140098461 > 89.57919140040366
    F(7.000000000000001E-10) = LineSearchPoint{point=PointSample{avg=89.57919140040366}, derivative=-0.9673333583571078}, delta = -6.772324923076667E-10
    New Minimum: 89.57919140040366 > 89.57919139634157
    F(4.900000000000001E-9) = LineSearchPoint{point=PointSample{avg=89.57919139634157}, derivative=-0.9673333583311443}, delta = -4.739320047519868E-9
    New Minimum: 89.57919139634157 > 89.57919136790191
    F(3.430000000000001E-8) = LineSearchPoint{point=PointSample{avg=89.57919136790191}, derivative=-0.9673333581494002}, delta = -3.3178977787429176E-8
    New Minimum: 89.57919136790191 > 89.57919116882445
    F(2.4010000000000004E-7) = LineSearchPoin
```
...[skipping 295500 bytes](etc/15.txt)...
```
    : 239424285952973.8000; Orientation: 0.0008; Line Search: 0.0875
    F(0.0) = LineSearchPoint{point=PointSample{avg=1.257607052313293E-4}, derivative=-3.060300230755859E-9}
    New Minimum: 1.257607052313293E-4 > 1.253935638407073E-4
    F(135.9454108712557) = LineSearchPoint{point=PointSample{avg=1.253935638407073E-4}, derivative=-2.341005900401667E-9}, delta = -3.671413906220075E-7
    New Minimum: 1.253935638407073E-4 > 1.2524419552580848E-4
    F(951.6178760987898) = LineSearchPoint{point=PointSample{avg=1.2524419552580848E-4}, derivative=1.9747600817232495E-9}, delta = -5.165097055208118E-7
    1.2524419552580848E-4 <= 1.257607052313293E-4
    New Minimum: 1.2524419552580848E-4 > 1.248756793138989E-4
    F(578.3915633738718) = LineSearchPoint{point=PointSample{avg=1.248756793138989E-4}, derivative=3.708665191964912E-23}, delta = -8.850259174303951E-7
    Right bracket at 578.3915633738718
    Converged to right
    Iteration 250 complete. Error: 1.248756793138989E-4 Total: 239424347959711.7200; Orientation: 0.0009; Line Search: 0.0468
    
```

Returns: 

```
    1.248756793138989E-4
```



This training run resulted in the following regressed input:

Code from [LearningTester.java:144](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L144) executed in 0.01 seconds: 
```java
    return Arrays.stream(input_gd).map(x -> x.prettyPrint()).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [
    	[ [ 0.3415346435781033, 0.8055237456950946, -0.48122481933834194 ], [ -1.1349898675729677, 0.24646595880272923, -0.7000600646861052 ], [ -1.8955569959454452, -0.40064322119435125, 0.2682489147867502 ], [ -0.19733721098778198, -1.4906857114473728, -1.4830348688180968 ], [ -0.6992625348612392, -1.0007367125279862, -0.003158474155506385 ], [ -0.6572612908570185, -1.0801986223958147, -1.187646913766341 ], [ 2.193985474758862, -0.7554402428252086, 1.533449094931032 ], [ 0.2708046386935112, 2.1921721097937423, 0.4615851066694088 ], ... ],
    	[ [ -0.26717899061192896, 1.7916260209101924, 1.4581004334407894 ], [ -0.37591168856935786, 1.3206253223329263, -1.2645905609331536 ], [ -2.4092983623529762, 1.0628241031628214, -1.3107962808707563 ], [ -0.45949815123442667, 0.446206797775974, -1.3462549549741196 ], [ -0.28226375612232096, -0.9619266416972916, 1.3870992936214768 ], [ 0.8430649692482851, -1.0361754518429236, -0.7360184025359766 ], [ 0.20790650775903877, 0.8738912695422139, -0.8863065309075985 ], [ 0.77436278226
```
...[skipping 2241 bytes](etc/16.txt)...
```
    0.7689793618760711 ], [ -0.09090873859412435, -0.9347401845646419, 1.416810780282614 ], [ 2.8892495330892, -0.5251230276114689, 2.017406151848169 ], [ -0.49666035854850693, -1.159649276083148, 2.4318425618972124 ], [ -2.3368616992895643, -1.072921161786203, -1.9809392470337932 ], [ -0.396386364503429, -0.572153818221506, 0.38467130064095756 ], [ -1.3966105261399175, 0.8210398272097625, -0.8508237448144611 ], [ -2.1536142454875833, 0.42087287899212056, 1.0119014410826173 ], ... ],
    	[ [ 0.20805583747611495, 1.8177988535345337, 1.5089674217313382 ], [ -0.07183683953602639, -0.10925022047297953, -0.6783657996212928 ], [ 0.7879151462204063, 1.360461540902077, 0.40316442183451245 ], [ 0.7308430322139192, 0.5862057594805001, 1.016705793943968 ], [ 0.24420970486688276, 0.3663903914545707, -0.8669851634967696 ], [ 1.162627582219503, 0.9392443226393499, -2.0766470623953244 ], [ 1.3606335793788986, 1.8126894413032693, -0.659140280819904 ], [ 0.8612115614793983, -0.01981461087790733, -0.03344304177751365 ], ... ],
    	...
    ]
```



Next, we run the same optimization using L-BFGS, which is nearly ideal for purely second-order or quadratic functions.

Code from [LearningTester.java:249](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L249) executed in 2.93 seconds: 
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
    th(0)=89.57919140108089;dx=-0.967333358361435
    New Minimum: 89.57919140108089 > 87.50948147458736
    WOLFE (weak): th(2.154434690031884)=87.50948147458736; dx=-0.954015138364646 delta=2.0697099264935304
    New Minimum: 87.50948147458736 > 85.46846478326499
    WOLFE (weak): th(4.308869380063768)=85.46846478326499; dx=-0.9406969183678571 delta=4.110726617815899
    New Minimum: 85.46846478326499 > 77.59133036967785
    WOLFE (weak): th(12.926608140191302)=77.59133036967785; dx=-0.8874240383807012 delta=11.987861031403042
    New Minimum: 77.59133036967785 > 47.825486072308344
    END: th(51.70643256076521)=47.825486072308344; dx=-0.6476960784384997 delta=41.75370532877255
    Iteration 1 complete. Error: 47.825486072308344 Total: 239424454806943.6200; Orientation: 0.0016; Line Search: 0.0496
    LBFGS Accumulation History: 1 points
    th(0)=47.825486072308344;dx=-0.4420978076934984
    New Minimum: 47.825486072308344 > 14.659999318746927
    END: th(111.39813200670
```
...[skipping 31293 bytes](etc/17.txt)...
```
    E (weak): th(242.76840175892335)=0.0015181888817006259; dx=-1.2507804155719428E-7 delta=3.061998051914522E-5
    New Minimum: 0.0015181888817006259 > 0.0014880788697318147
    WOLFE (weak): th(485.5368035178467)=0.0014880788697318147; dx=-1.2297740347322718E-7 delta=6.072999248795644E-5
    New Minimum: 0.0014880788697318147 > 0.001372738507359741
    WOLFE (weak): th(1456.61041055354)=0.001372738507359741; dx=-1.1457485113735872E-7 delta=1.760703548600302E-4
    New Minimum: 0.001372738507359741 > 9.546806496484613E-4
    END: th(5826.44164221416)=9.546806496484613E-4; dx=-7.676336562595074E-8 delta=5.941282125713098E-4
    Iteration 64 complete. Error: 9.546806496484613E-4 Total: 239427260073797.8000; Orientation: 0.0016; Line Search: 0.0583
    LBFGS Accumulation History: 1 points
    th(0)=9.546806496484613E-4;dx=-1.391779223904071E-7
    MAX ALPHA: th(0)=9.546806496484613E-4;th'(0)=-1.391779223904071E-7;
    Iteration 65 failed, aborting. Error: 9.546806496484613E-4 Total: 239427303843343.7500; Orientation: 0.0016; Line Search: 0.0322
    
```

Returns: 

```
    9.546806496484613E-4
```



This training run resulted in the following regressed input:

Code from [LearningTester.java:154](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L154) executed in 0.01 seconds: 
```java
    return Arrays.stream(input_lbgfs).map(x -> x.prettyPrint()).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [
    	[ [ 0.40722769594608743, 0.8822689762867152, -0.5331218595954014 ], [ -1.0475208641783909, 0.4046557078809027, -0.6318516938668526 ], [ -2.004660127527806, -0.33052546636534524, 0.2637859249726999 ], [ -0.14892290941036873, -1.4050041807542304, -1.2363445309397914 ], [ -0.40835834676247507, -0.6365286640343973, -0.33417504820917526 ], [ -0.6473110714082285, -0.8805210073839096, -0.8523817782268379 ], [ 2.1355380745757544, -0.5913091224763168, 1.4745120831817282 ], [ 0.48313114461747, 2.3728599227059908, 0.6198473712636621 ], ... ],
    	[ [ -0.4890345087007196, 1.8116956500454366, 1.5316245977816696 ], [ -0.5180503693667171, 1.2897334844083284, -1.269650130582284 ], [ -2.4072845933513096, 1.2246463366730256, -1.0993149862502556 ], [ -0.5694721298041465, 0.4353517475860804, -1.499056057232647 ], [ -0.6276837213515251, -1.0061383101240804, 1.4445562920846677 ], [ 0.7555296373956912, -0.8259697670415199, -0.5094928383112124 ], [ 0.029012825025298852, 1.0640408441739155, -1.0693200996521233 ], [ 0.6656540445954813
```
...[skipping 2242 bytes](etc/18.txt)...
```
    6609736027254 ], [ -0.1605585353419169, -0.9566942212632963, 1.300366533056335 ], [ 2.826276939289889, -0.5791047687535202, 1.8555945517554993 ], [ -0.4357651135190551, -1.2151313115923443, 2.3475062374759945 ], [ -2.3568234414452394, -1.0485207084731976, -1.9777510049613256 ], [ -0.43155951013800536, -0.5273970251381828, 0.44634169206844554 ], [ -1.4015407802476854, 0.83577842520675, -0.8260600169697784 ], [ -2.115347050895496, 0.37720718064326403, 0.9457476009828967 ], ... ],
    	[ [ 0.09851379200892571, 1.5403292421513928, 1.5539647277711857 ], [ -0.16690631215515045, -0.3274014342623744, -0.8841576302639229 ], [ 0.9308942171265445, 1.2848992089772044, 0.1429393950348224 ], [ 0.6649503177664023, 0.5899079976388478, 1.0343159780999114 ], [ 0.17894889374663286, 0.458780864998283, -0.7809188860985763 ], [ 1.1704093880984592, 0.9536280883248553, -2.0909072801546844 ], [ 1.3899073228510215, 1.7690714484325387, -0.6911342643596177 ], [ 0.8753954310876799, -0.04209252133228723, -0.003133337143910112 ], ... ],
    	...
    ]
```



Code from [LearningTester.java:95](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L95) executed in 0.01 seconds: 
```java
    return TestUtil.compare(runs);
```

Returns: 

![Result](etc/test.9.png)



Code from [LearningTester.java:98](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L98) executed in 0.01 seconds: 
```java
    return TestUtil.compareTime(runs);
```

Returns: 

![Result](etc/test.10.png)



### Model Learning
In this test, attempt to train a network to emulate a randomized network given an example input/output. The target state is:

Code from [LearningTester.java:176](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L176) executed in 0.00 seconds: 
```java
    return network_target.state().stream().map(Arrays::toString).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [-0.08, 0.684, -1.192, -1.224, 1.84, 0.624, -1.536, -1.004, 0.364, 1.52, 0.628, -1.992, -0.228, -0.968, -0.308, 1.696, 1.244, 0.16, 1.38, -0.584, -1.428, -1.392, 0.908, 1.468, -0.436, -1.86, 0.084, 0.032, 0.668, 0.672, 0.364, -1.064, -1.272, -0.072, 0.444, -0.656, 1.712, -1.012, -1.572, -0.516, -1.22, -0.544, -0.168, 0.428, 1.968, -1.436, -1.408, 1.62, -0.7, 1.924, 0.852, -0.632, 0.484, -1.988]
```



First, we use a conjugate gradient descent method, which converges the fastest for purely linear functions.

Code from [LearningTester.java:225](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L225) executed in 1.78 seconds: 
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
    F(0.0) = LineSearchPoint{point=PointSample{avg=78.80029440125332}, derivative=-215.18399593931295}
    New Minimum: 78.80029440125332 > 78.80029437973499
    F(1.0E-10) = LineSearchPoint{point=PointSample{avg=78.80029437973499}, derivative=-215.1839959098755}, delta = -2.1518332005143748E-8
    New Minimum: 78.80029437973499 > 78.8002942506242
    F(7.000000000000001E-10) = LineSearchPoint{point=PointSample{avg=78.8002942506242}, derivative=-215.18399573325087}, delta = -1.5062911984387028E-7
    New Minimum: 78.8002942506242 > 78.80029334685128
    F(4.900000000000001E-9) = LineSearchPoint{point=PointSample{avg=78.80029334685128}, derivative=-215.18399449687837}, delta = -1.0544020483393979E-6
    New Minimum: 78.80029334685128 > 78.80028702044261
    F(3.430000000000001E-8) = LineSearchPoint{point=PointSample{avg=78.80028702044261}, derivative=-215.18398584227077}, delta = -7.380810714607833E-6
    New Minimum: 78.80028702044261 > 78.80024273558472
    F(2.4010000000000004E-7) = LineSearchPoint{p
```
...[skipping 18253 bytes](etc/19.txt)...
```
    0.0555
    Zero gradient: 3.337944830428296E-16
    F(0.0) = LineSearchPoint{point=PointSample{avg=2.608278461142645E-31}, derivative=-1.1141875690982987E-31}
    New Minimum: 2.608278461142645E-31 > 1.3928299689437718E-32
    F(0.761166770483209) = LineSearchPoint{point=PointSample{avg=1.3928299689437718E-32}, derivative=3.915343581561136E-34}, delta = -2.468995464248268E-31
    1.3928299689437718E-32 <= 2.608278461142645E-31
    Converged to right
    Iteration 16 complete. Error: 1.3928299689437718E-32 Total: 239429214762801.8000; Orientation: 0.0001; Line Search: 0.0247
    Zero gradient: 2.6098743103813023E-17
    F(0.0) = LineSearchPoint{point=PointSample{avg=1.3928299689437718E-32}, derivative=-6.811443915988278E-34}
    New Minimum: 1.3928299689437718E-32 > 0.0
    F(0.761166770483209) = LineSearchPoint{point=PointSample{avg=0.0}, derivative=0.0}, delta = -1.3928299689437718E-32
    0.0 <= 1.3928299689437718E-32
    Converged to right
    Iteration 17 complete. Error: 0.0 Total: 239429252551790.7800; Orientation: 0.0001; Line Search: 0.0223
    
```

Returns: 

```
    0.0
```



Training Converged

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
    th(0)=63.507286590923094;dx=-168.57024089620973
    Armijo: th(2.154434690031884)=220.55076531036187; dx=314.35649233938193 delta=-157.04347871943878
    New Minimum: 63.507286590923094 > 11.974762597326759
    WOLF (strong): th(1.077217345015942)=11.974762597326759; dx=72.89312572158607 delta=51.532523993596335
    END: th(0.3590724483386473)=17.428831181209496; dx=-88.08245202361114 delta=46.0784554097136
    Iteration 1 complete. Error: 11.974762597326759 Total: 239429390906234.7500; Orientation: 0.0004; Line Search: 0.0769
    LBFGS Accumulation History: 1 points
    th(0)=17.428831181209496;dx=-46.09815051225332
    New Minimum: 17.428831181209496 > 0.04574621709257671
    WOLF (strong): th(0.7735981389354633)=0.04574621709257671; dx=1.1572849924953175 delta=17.383084964116918
    END: th(0.3867990694677316)=4.167699079018748; dx=-22.470432759879 delta=13.261132102190748
    Iteration 2 complete. Error: 0.04574621709257671 Total: 239429436330364.6000; Orienta
```
...[skipping 12074 bytes](etc/20.txt)...
```
    043257553146E-32
    END: th(0.31707238121266507)=1.4287015170957908E-32; dx=-1.966661526188408E-33 delta=1.4205518093761013E-32
    Iteration 25 complete. Error: 1.1146490007165776E-32 Total: 239430524504672.5300; Orientation: 0.0002; Line Search: 0.0384
    LBFGS Accumulation History: 1 points
    th(0)=1.4287015170957908E-32;dx=-1.871314387199633E-33
    New Minimum: 1.4287015170957908E-32 > 1.1288949464943025E-32
    WOLFE (weak): th(0.6831117373355794)=1.1288949464943025E-32; dx=-1.70566753437214E-33 delta=2.9980657060148824E-33
    New Minimum: 1.1288949464943025E-32 > 3.0500441568255E-33
    WOLF (strong): th(1.3662234746711588)=3.0500441568255E-33; dx=3.978888808135603E-35 delta=1.1236971014132407E-32
    WOLF (strong): th(1.024667606003369)=3.0500441568255E-33; dx=3.978888808135603E-35 delta=1.1236971014132407E-32
    New Minimum: 3.0500441568255E-33 > 0.0
    END: th(0.8538896716694742)=0.0; dx=0.0 delta=1.4287015170957908E-32
    Iteration 26 complete. Error: 0.0 Total: 239430576368174.4700; Orientation: 0.0001; Line Search: 0.0432
    
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

![Result](etc/test.11.png)



Code from [LearningTester.java:98](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L98) executed in 0.01 seconds: 
```java
    return TestUtil.compareTime(runs);
```

Returns: 

![Result](etc/test.12.png)



### Performance
Now we execute larger-scale runs to benchmark performance:

Code from [PerformanceTester.java:66](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/PerformanceTester.java#L66) executed in 2.37 seconds: 
```java
    test(component, inputPrototype);
```
Logging: 
```
    100 batches
    Input Dimensions:
    	[100, 100, 3]
    Performance:
    	Evaluation performance: 0.097940s +- 0.003114s [0.092832s - 0.102199s]
    	Learning performance: 0.269253s +- 0.021220s [0.253498s - 0.311263s]
    
```

