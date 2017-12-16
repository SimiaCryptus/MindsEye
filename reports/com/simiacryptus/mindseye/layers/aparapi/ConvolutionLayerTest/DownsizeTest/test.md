# ConvolutionLayer
## DownsizeTest
### Batch Execution
Code from [BatchingTester.java:66](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/BatchingTester.java#L66) executed in 0.09 seconds: 
```java
    return test(reference, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (660#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (100#)}
```



Code from [SingleDerivativeTester.java:77](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/SingleDerivativeTester.java#L77) executed in 1.11 seconds: 
```java
    return test(component, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ -0.432, -1.324, -1.236, 1.716, 1.116, -0.712, -1.464 ], [ 0.368, 1.26, 0.808, -0.084, 1.736, 1.824, -0.552 ], [ 1.716, -1.148, 0.72, 1.804, 0.692, -1.008, 0.528 ] ],
    	[ [ 0.204, -0.316, 0.036, 1.352, -0.368, 1.876, 1.696 ], [ -0.808, 1.336, -0.02, 1.788, -1.648, -1.06, 0.62 ], [ -0.904, -1.852, 0.016, 0.848, 0.216, 0.536, 1.52 ] ],
    	[ [ -1.308, 0.616, -0.876, 1.332, 0.412, -1.036, 1.456 ], [ 0.868, -0.352, 1.9, 0.008, -0.676, 1.22, -1.876 ], [ -0.52, -0.144, 1.216, -1.32, 1.312, -0.136, 0.284 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.18952175473733154, negative=27, min=0.284, max=0.284, mean=0.2186666666666667, count=63.0, positive=36, stdDev=1.0938802348767294, zeros=0}
    Output: [
    	[ [ 2.856496, 1.3469600000000002, 0.45352000000000015 ] ]
    ]
    Outputs Statistics: {meanExponent=0.08059492103919498, negative=0, min=0.45352000000000015, max=0.45352000000000015, mean=1.5523253333333333, count=3.0, positive=3, stdDev=0.9917004266226549, zeros=0}
    Feedback for input 0
    Inputs Values: [
    	[ [ -0.432, -
```
...[skipping 2256 bytes](etc/6.txt)...
```
    =-0.012359788359788359, count=567.0, positive=6, stdDev=0.23320603709492888, zeros=546}
    Measured Gradient: [ [ -0.43199999999909977, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], ... ]
    Measured Statistics: {meanExponent=0.02137039302043678, negative=15, min=0.0, max=0.0, mean=-0.01235978835978349, count=567.0, positive=6, stdDev=0.23320603709486196, zeros=546}
    Gradient Error: [ [ 9.002243395173082E-13, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], ... ]
    Error Statistics: {meanExponent=-12.221075660914638, negative=11, min=0.0, max=0.0, mean=4.869610495752263E-15, count=567.0, positive=10, stdDev=2.5841669667985337E-13, zeros=546}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 5.3239e-14 +- 2.8623e-13 [0.0000e+00 - 2.7154e-12] (756#)
    relativeTol: 5.5321e-13 +- 5.5874e-13 [2.4328e-15 - 2.8130e-12] (42#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=5.3239e-14 +- 2.8623e-13 [0.0000e+00 - 2.7154e-12] (756#), relativeTol=5.5321e-13 +- 5.5874e-13 [2.4328e-15 - 2.8130e-12] (42#)}
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
      "id": "77fabed5-8338-4ff9-bbeb-7a0d333ee202",
      "isFrozen": false,
      "name": "ConvolutionLayer/77fabed5-8338-4ff9-bbeb-7a0d333ee202",
      "filter": [
        [
          [
            1.876,
            0.4,
            -0.612
          ],
          [
            0.176,
            -1.344,
            -0.632
          ],
          [
            -1.916,
            -0.448,
            0.8
          ]
        ],
        [
          [
            1.028,
            -0.912,
            0.312
          ],
          [
            0.952,
            -1.664,
            1.548
          ],
          [
            -0.768,
            -0.044,
            -0.792
          ]
        ],
        [
          [
            0.192,
            -1.388,
            1.296
          ],
          [
            0.844,
            -0.228,
            -1.684
          ],
          [
            -1.548,
            1.128,
            -0.992
          ]
        ],
        [
          [
            0.068,
            0.04,
            -1.424
          ],
          [
            -0.3,
            0.508,
            -1.976
          ],
          [
            -1.664,
            -1.064,
            0.132
          ]
        ],
        [
     
```
...[skipping 2395 bytes](etc/7.txt)...
```
    
            -1.808,
            -0.496
          ],
          [
            0.856,
            0.336,
            -0.188
          ],
          [
            0.692,
            -0.648,
            -1.712
          ]
        ],
        [
          [
            -1.124,
            -0.2,
            0.628
          ],
          [
            1.56,
            0.536,
            -0.68
          ],
          [
            -0.028,
            1.416,
            -1.84
          ]
        ],
        [
          [
            -1.764,
            -1.212,
            -1.8
          ],
          [
            1.464,
            1.5,
            -1.764
          ],
          [
            0.72,
            -0.368,
            -0.332
          ]
        ],
        [
          [
            -1.328,
            -0.972,
            1.156
          ],
          [
            0.336,
            -0.176,
            -0.088
          ],
          [
            -1.288,
            -0.404,
            -0.112
          ]
        ],
        [
          [
            0.244,
            0.068,
            -0.204
          ],
          [
            -1.06,
            0.196,
            0.928
          ],
          [
            0.752,
            -0.44,
            -1.536
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
    	[ [ 1.312, 0.552, -0.884, 1.348, -1.624, 0.8, -0.172 ], [ 1.664, 1.42, -0.372, -0.42, -1.192, -0.708, -0.072 ], [ -0.512, 1.98, 0.892, -1.98, -1.824, -0.7, -1.768 ] ],
    	[ [ -1.368, -0.148, -0.676, 1.228, -0.192, 1.432, 2.0 ], [ -0.232, 1.928, -1.496, -0.088, -1.216, 0.492, -1.564 ], [ -0.444, -1.068, 1.256, 0.82, 1.416, -1.444, -1.544 ] ],
    	[ [ 0.868, 0.764, 1.204, 0.072, -0.108, -1.82, -0.716 ], [ 0.524, 0.848, 1.188, -0.948, 1.772, 1.816, 0.356 ], [ -1.496, 1.364, -1.02, -0.444, 0.84, 0.168, -0.632 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ -0.045232000000000695, 0.33398400000000045, -1.6554880000000005 ] ]
    ]
    --------------------
    Derivative: 
    [
    	[ [ 3.096, 2.92, -4.672, -3.2680000000000002, 2.832, -3.32, -2.848 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ] ],
    	[ [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ] ],
    	[ [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ] ]
    ]
```



### Input Learning
In this test, we use a network to learn this target input, given it's pre-evaluated output:

Code from [LearningTester.java:127](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L127) executed in 0.04 seconds: 
```java
    return Arrays.stream(input_target).map(x -> x.prettyPrint()).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [
    	[ [ 0.78, 1.708, 1.62 ], [ 1.276, 0.216, -0.088 ], [ -0.256, 1.208, -0.316 ], [ -0.548, 0.08, 0.092 ], [ -0.264, 1.592, -1.424 ], [ -1.712, 0.888, -0.544 ], [ 1.136, -1.808, 1.532 ], [ -0.172, -1.144, 0.336 ], ... ],
    	[ [ -0.316, 1.828, 1.216 ], [ -1.112, -1.072, -1.192 ], [ 0.656, -1.44, -0.524 ], [ 0.544, 0.368, 1.984 ], [ 0.912, -0.916, -1.26 ], [ 1.272, 0.092, 1.824 ], [ 0.14, 0.848, 1.744 ], [ 1.964, 0.528, 1.884 ], ... ],
    	[ [ -0.608, -1.544, 1.876 ], [ 1.364, 1.424, 0.092 ], [ -0.18, -0.064, -1.496 ], [ 1.932, -1.296, -1.78 ], [ 0.644, 1.984, -1.112 ], [ -0.032, -0.1, 1.468 ], [ 1.212, -0.028, 0.308 ], [ -1.344, -1.24, 0.108 ], ... ],
    	[ [ 1.436, 1.824, -1.156 ], [ 0.224, 0.196, -0.18 ], [ 1.636, 0.676, -1.004 ], [ 0.188, 1.336, -0.34 ], [ 1.216, 1.276, -0.92 ], [ 1.348, -0.192, -0.532 ], [ -1.196, 1.316, -0.024 ], [ -1.276, 0.248, -1.896 ], ... ],
    	[ [ -1.6, 0.232, -0.912 ], [ -0.868, -0.768, 1.112 ], [ -0.316, -1.484, -1.012 ], [ 0.2, 0.992, 0.084 ], [ 0.212, 1.62, -0.1 ], [ -1.068, -1.852, -1.136 ], [ 0.456, 0.416, -1.904 ], [ -0.34, -0.128, 0.092 ], ... ],
    	[ [ 1.064, -0.272, -1.856 ], [ 0.192, -0.676, -0.984 ], [ -0.152, 1.916, 0.088 ], [ 1.552, -0.84, 0.372 ], [ 1.828, -1.456, 0.04 ], [ -0.392, 1.656, 0.272 ], [ -1.668, 1.872, 0.224 ], [ 1.432, -1.912, 1.808 ], ... ],
    	[ [ 1.584, -1.504, -1.72 ], [ 0.104, 0.204, 1.316 ], [ 0.964, 0.096, 0.816 ], [ 1.044, 0.188, -0.988 ], [ 1.068, 0.66, -0.744 ], [ 0.088, 1.564, -1.724 ], [ 0.56, -1.716, 0.08 ], [ 0.812, -0.18, 1.968 ], ... ],
    	[ [ 1.384, 1.856, 1.376 ], [ -0.24, -0.736, -0.1 ], [ -1.784, 0.276, -0.176 ], [ 1.848, -0.716, -0.556 ], [ -0.38, -1.616, 1.036 ], [ -0.724, -1.476, -1.868 ], [ 0.484, 1.252, 1.428 ], [ 0.936, 0.544, -0.972 ], ... ],
    	...
    ]
```



First, we use a conjugate gradient descent method, which converges the fastest for purely linear functions.

Code from [LearningTester.java:225](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L225) executed in 25.49 seconds: 
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
    F(0.0) = LineSearchPoint{point=PointSample{avg=94.16373115989067}, derivative=-0.7378635655671774}
    New Minimum: 94.16373115989067 > 94.16373115981689
    F(1.0E-10) = LineSearchPoint{point=PointSample{avg=94.16373115981689}, derivative=-0.7378635655668112}, delta = -7.37827576813288E-11
    New Minimum: 94.16373115981689 > 94.16373115937367
    F(7.000000000000001E-10) = LineSearchPoint{point=PointSample{avg=94.16373115937367}, derivative=-0.7378635655646141}, delta = -5.170051053937641E-10
    New Minimum: 94.16373115937367 > 94.1637311562737
    F(4.900000000000001E-9) = LineSearchPoint{point=PointSample{avg=94.1637311562737}, derivative=-0.7378635655492343}, delta = -3.6169751638226444E-9
    New Minimum: 94.1637311562737 > 94.16373113458265
    F(3.430000000000001E-8) = LineSearchPoint{point=PointSample{avg=94.16373113458265}, derivative=-0.7378635654415755}, delta = -2.5308025897174957E-8
    New Minimum: 94.16373113458265 > 94.163730982728
    F(2.4010000000000004E-7) = LineSearchPoint{po
```
...[skipping 281562 bytes](etc/8.txt)...
```
    = LineSearchPoint{point=PointSample{avg=1.3825402501650332E-5}, derivative=-8.184701833310113E-10}, delta = -1.595609368763749E-7
    F(905.2666959070416) = LineSearchPoint{point=PointSample{avg=1.5123993437827106E-5}, derivative=4.1656004063306365E-9}, delta = 1.139029999300399E-6
    F(69.63589968515704) = LineSearchPoint{point=PointSample{avg=1.3885697155608353E-5}, derivative=-1.2018602286896293E-9}, delta = -9.926628291835347E-8
    F(487.4512977960993) = LineSearchPoint{point=PointSample{avg=1.3944193371201349E-5}, derivative=1.4818700888204964E-9}, delta = -4.077006732535768E-8
    1.3944193371201349E-5 <= 1.3984963438526707E-5
    New Minimum: 1.3774138710994318E-5 > 1.3773256465791548E-5
    F(256.7469915289699) = LineSearchPoint{point=PointSample{avg=1.3773256465791548E-5}, derivative=-2.5461653230147883E-23}, delta = -2.1170697273515886E-7
    Left bracket at 256.7469915289699
    Converged to left
    Iteration 250 complete. Error: 1.3773256465791548E-5 Total: 239393493081852.6600; Orientation: 0.0009; Line Search: 0.1346
    
```

Returns: 

```
    1.3773256465791548E-5
```



This training run resulted in the following regressed input:

Code from [LearningTester.java:144](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L144) executed in 0.03 seconds: 
```java
    return Arrays.stream(input_gd).map(x -> x.prettyPrint()).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [
    	[ [ 0.779999855389401, 1.7079995785508306, 1.6200001012923928 ], [ 1.2759996794046347, 0.215999361603518, -0.08799990986973306 ], [ -0.2560004178227934, 1.2079989130653253, -0.31599995035425493 ], [ -0.5480002789548603, 0.07999876083148715, 0.09199979354400273 ], [ -0.26400052745036745, 1.5919984273123355, -1.4240000847960272 ], [ -1.7120009571929031, 0.8879978368621068, -0.5439998536558303 ], [ 1.13599972761786, -1.808002408204292, 1.5319996581512823 ], [ -0.17200088292651614, -1.1440020597402254, 0.3359994212737081 ], ... ],
    	[ [ -0.3160006128988201, 1.8279991572449936, 1.216000004449165 ], [ -1.112001195250676, -1.0720014327834826, -1.1920000221266447 ], [ 0.6559984262453414, -1.4400022782062243, -0.5240002947076802 ], [ 0.5439982371599668, 0.36799740655828983, 1.9839993367752422 ], [ 0.9119976032781918, -0.916003519943942, -1.2600002511282204 ], [ 1.2719967413310922, 0.09199551382481672, 1.8239997745914942 ], [ 0.1399975775336243, 0.8479951636296034, 1.7439988384839142 ], [ 1.9639959435627918, 0.527995
```
...[skipping 2224 bytes](etc/9.txt)...
```
    35044, -1.719999947907691 ], [ 0.10399971517333066, 0.20400054282667324, 1.3159997882557688 ], [ 0.9640000973758552, 0.0960021445301569, 0.8159996428830111 ], [ 1.0439977393755004, 0.1880013028749073, -0.9879991598982648 ], [ 1.067999620442639, 0.6600004418203472, -0.7439989200628916 ], [ 0.0879990908260734, 1.564005028050379, -1.724001855142962 ], [ 0.5599988439958163, -1.715996836853575, 0.08000190574813647 ], [ 0.811996292713302, -0.1799991705489858, 1.9680015961203763 ], ... ],
    	[ [ 1.38400046567423, 1.8560011580086604, 1.376000113589551 ], [ -0.23999790716446157, -0.7359975049730302, -0.09999993447358284 ], [ -1.7839970442958635, 0.2760046747130477, -0.1759994465977301 ], [ 1.848001071665629, -0.715996091212088, -0.5559982894863218 ], [ -0.37999475329420734, -1.615994864093597, 1.0360015183018303 ], [ -0.7239956217604626, -1.475989361802498, -1.8680009088197325 ], [ 0.4840044876796845, 1.2520073829270848, 1.4280040982753621 ], [ 0.9360045062819612, 0.5440074858824152, -0.9719976627956923 ], ... ],
    	...
    ]
```



Next, we run the same optimization using L-BFGS, which is nearly ideal for purely second-order or quadratic functions.

Code from [LearningTester.java:249](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L249) executed in 5.44 seconds: 
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
    th(0)=94.16373115989067;dx=-0.7378635655671774
    New Minimum: 94.16373115989067 > 92.5825507290623
    WOLFE (weak): th(2.154434690031884)=92.5825507290623; dx=-0.7299743207640185 delta=1.5811804308283683
    New Minimum: 92.5825507290623 > 91.01836716091805
    WOLFE (weak): th(4.308869380063768)=91.01836716091805; dx=-0.7220850759608597 delta=3.145363998972627
    New Minimum: 91.01836716091805 > 84.93160151515693
    WOLFE (weak): th(12.926608140191302)=84.93160151515693; dx=-0.6905280967482242 delta=9.232129644733746
    New Minimum: 84.93160151515693 > 60.906534920283754
    END: th(51.70643256076521)=60.906534920283754; dx=-0.548521690291365 delta=33.25719623960692
    Iteration 1 complete. Error: 60.906534920283754 Total: 239393650230050.4700; Orientation: 0.0016; Line Search: 0.0740
    LBFGS Accumulation History: 1 points
    th(0)=60.906534920283754;dx=-0.41448506894853054
    New Minimum: 60.906534920283754 > 26.278204339549976
    END: th(111.3981320067066
```
...[skipping 43578 bytes](etc/10.txt)...
```
    eak): th(271.00882183664726)=9.691671551829056E-5; dx=-9.692670412833779E-9 delta=2.648252448787195E-6
    New Minimum: 9.691671551829056E-5 > 9.431136958901061E-5
    WOLFE (weak): th(542.0176436732945)=9.431136958901061E-5; dx=-9.534348926414188E-9 delta=5.253598378067147E-6
    New Minimum: 9.431136958901061E-5 > 8.431905106694922E-5
    WOLFE (weak): th(1626.0529310198835)=8.431905106694922E-5; dx=-8.90106298073586E-9 delta=1.5245916900128543E-5
    New Minimum: 8.431905106694922E-5 > 4.7849108579868304E-5
    END: th(6504.211724079534)=4.7849108579868304E-5; dx=-6.051276225183346E-9 delta=5.1715859387209455E-5
    Iteration 85 complete. Error: 4.7849108579868304E-5 Total: 239398943510996.1600; Orientation: 0.0015; Line Search: 0.0718
    LBFGS Accumulation History: 1 points
    th(0)=4.7849108579868304E-5;dx=-4.7163784755653764E-9
    MAX ALPHA: th(0)=4.7849108579868304E-5;th'(0)=-4.7163784755653764E-9;
    Iteration 86 failed, aborting. Error: 4.7849108579868304E-5 Total: 239398986380295.0600; Orientation: 0.0015; Line Search: 0.0288
    
```

Returns: 

```
    4.7849108579868304E-5
```



This training run resulted in the following regressed input:

Code from [LearningTester.java:154](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L154) executed in 0.01 seconds: 
```java
    return Arrays.stream(input_lbgfs).map(x -> x.prettyPrint()).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [
    	[ [ 0.7800005196830067, 1.7080007992409563, 1.619999741433507 ], [ 1.2760010473975787, 0.21600156408023863, -0.0880005780634615 ], [ -0.25599937154095076, 1.208002151614294, -0.31599908020221684 ], [ -0.5479988981088917, 0.08000109764064348, 0.09200096540675333 ], [ -0.2639987111450654, 1.5920047201634706, -1.4240015293978463 ], [ -1.7119955917404721, 0.8880029568388657, -0.5439993480560166 ], [ 1.1359967739937777, -1.8079954472011686, 1.5320016437028645 ], [ -0.1719931506782462, -1.143998371531447, 0.3360007942768001 ], ... ],
    	[ [ -0.3159987134006429, 1.8280011209274443, 1.2159996768840653 ], [ -1.1119974332910172, -1.0719973532362348, -1.1920004819430092 ], [ 0.6560016658797455, -1.4399966135919962, -0.523997498101127 ], [ 0.5440047861678905, 0.36800139457079073, 1.984001399843086 ], [ 0.9120024005391046, -0.9159906746072295, -1.2600029232929242 ], [ 1.272009183459564, 0.09200364986063916, 1.8240035743527498 ], [ 0.13999809872079635, 0.8480075391326075, 1.7440024320691871 ], [ 1.9640146309015436, 0.5280
```
...[skipping 2236 bytes](etc/11.txt)...
```
    1.719998225324019 ], [ 0.10399396373342715, 0.2039916400374295, 1.3160015442369581 ], [ 0.9639893967042389, 0.09599166541502596, 0.8159947087373185 ], [ 1.0440093284036132, 0.18799517153981038, -0.9880010224199974 ], [ 1.0679681026986694, 0.6599895977954743, -0.7440059987284974 ], [ 0.08800776475741656, 1.5639671223884979, -1.723986928308325 ], [ 0.5599811210412424, -1.715997899364907, 0.07997403837552323 ], [ 0.8119923893536964, -0.18001934236932296, 1.9680082828739247 ], ... ],
    	[ [ 1.3839974195221887, 1.8559967578225378, 1.376002632630761 ], [ -0.2400087926883892, -0.7360123590496417, -0.09999938624424508 ], [ -1.7840140787355176, 0.2759920132885709, -0.1760092919802233 ], [ 1.8480044107639875, -0.7160093843837959, -0.5559976922714277 ], [ -0.3800446036895194, -1.6160214723754371, 1.035993620275592 ], [ -0.7239936679338829, -1.476034770071859, -1.867991361644922 ], [ 0.48396973488037565, 1.2520004056461485, 1.4279703439370228 ], [ 0.9359754290972149, 0.5439631393355461, -0.9719872215827525 ], ... ],
    	...
    ]
```



Code from [LearningTester.java:95](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L95) executed in 0.01 seconds: 
```java
    return TestUtil.compare(runs);
```

Returns: 

![Result](etc/test.5.png)



Code from [LearningTester.java:98](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L98) executed in 0.01 seconds: 
```java
    return TestUtil.compareTime(runs);
```

Returns: 

![Result](etc/test.6.png)



### Model Learning
In this test, attempt to train a network to emulate a randomized network given an example input/output. The target state is:

Code from [LearningTester.java:176](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L176) executed in 0.00 seconds: 
```java
    return network_target.state().stream().map(Arrays::toString).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [-1.288, 0.844, 0.068, -0.3, -1.316, -1.916, 1.608, 0.312, 0.256, -1.548, -0.044, -1.664, -1.096, -0.176, -0.992, -0.448, -1.068, -0.188, 1.18, 1.896, 1.428, 0.616, -0.364, -1.936, -1.764, -1.868, 0.22, -1.4, -1.164, -1.34, 0.188, 1.276, -0.068, 0.576, -0.808, 0.208, -0.956, -1.032, -1.588, -0.792, 0.72, -0.848, -1.2, -0.912, -0.808, -0.216, 0.336, -1.88, -0.332, 1.128, -1.836, -1.068, 1.368, 1.028, -1.436, -0.9, -0.828, 0.04, 0.26, -0.204, 1.348, 0.692, 1.3, -0.632, 0.508, -0.792, -1.788, -0.612, 0.336, -1.8, 0.744, 1.536, -1.808, -0.26, -1.004, 1.08, 1.548, 0.576, -0.44, 0.928, 0.096, 1.5, -0.768, -1.84, 1.98, -1.96, -0.112, -0.16, 0.992, -0.972, -1.676, -0.044, -1.168, 0.04, 0.752, 1.156, -1.948, 0.12, 1.176, -1.424, -1.992, -0.496, -1.776, -1.648, -1.744, 0.176, -1.064, 0.4, 0.688, 1.296, 1.172, 1.504, -1.968, -1.548, -1.268, -1.524, -0.44, -1.132, -1.712, -0.368, 0.816, 0.856, -0.04, -0.18, 1.436, -1.976, 1.436, -1.064, 0.952, -0.476, 0.628, -1.748, -0.476, -1.764, -0.404, 1.028, -1.968, -1.664, -0.68, -1.032, -1.504, -1.536, 1.876, 0.132, 1.464, -1.584, -0.172, 0.0, 0.84, 0.076, -1.212, -1.328, -0.088, 0.196, 1.872, 1.416, 0.068, 0.536, -0.068, -0.648, -0.36, -1.952, -1.84, 0.8, 0.784, -1.452, 1.48, -1.344, -0.872, -1.868, 1.952, 0.192, -0.228, 1.56, 1.304, -0.112, -1.388, -1.168, -1.48, 0.204, 0.396, 1.02, -0.028, -1.06, -1.124, 0.244, -1.46, -1.684, -0.2]
```



First, we use a conjugate gradient descent method, which converges the fastest for purely linear functions.

Code from [LearningTester.java:225](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L225) executed in 2.08 seconds: 
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
    F(0.0) = LineSearchPoint{point=PointSample{avg=85.02366825709197}, derivative=-63.95824459789392}
    New Minimum: 85.02366825709197 > 85.02366825069493
    F(1.0E-10) = LineSearchPoint{point=PointSample{avg=85.02366825069493}, derivative=-63.95824459548063}, delta = -6.397044671757612E-9
    New Minimum: 85.02366825069493 > 85.02366821232125
    F(7.000000000000001E-10) = LineSearchPoint{point=PointSample{avg=85.02366821232125}, derivative=-63.95824458100089}, delta = -4.477071513520059E-8
    New Minimum: 85.02366821232125 > 85.02366794369716
    F(4.900000000000001E-9) = LineSearchPoint{point=PointSample{avg=85.02366794369716}, derivative=-63.95824447964267}, delta = -3.133948069944381E-7
    New Minimum: 85.02366794369716 > 85.0236660633251
    F(3.430000000000001E-8) = LineSearchPoint{point=PointSample{avg=85.0236660633251}, derivative=-63.958243770135205}, delta = -2.193766874825087E-6
    New Minimum: 85.0236660633251 > 85.02365290071884
    F(2.4010000000000004E-7) = LineSearchPoint{point=P
```
...[skipping 23156 bytes](etc/12.txt)...
```
    intSample{avg=5.8670561761543686E-36}, derivative=-1.2391147241876186E-38}, delta = -2.451707558704601E-33
    Left bracket at 3.367795028013252
    Converged to left
    Iteration 19 complete. Error: 5.8670561761543686E-36 Total: 239401160421440.9700; Orientation: 0.0002; Line Search: 0.1240
    Zero gradient: 1.1131553010194123E-19
    F(0.0) = LineSearchPoint{point=PointSample{avg=5.8670561761543686E-36}, derivative=-1.2391147241876186E-38}
    F(3.367795028013252) = LineSearchPoint{point=PointSample{avg=5.8670561761543686E-36}, derivative=-1.2391147241876186E-38}, delta = 0.0
    F(23.574565196092763) = LineSearchPoint{point=PointSample{avg=5.8670561761543686E-36}, derivative=-1.2391147241876186E-38}, delta = 0.0
    New Minimum: 5.8670561761543686E-36 > 0.0
    F(165.02195637264936) = LineSearchPoint{point=PointSample{avg=0.0}, derivative=0.0}, delta = -5.8670561761543686E-36
    0.0 <= 5.8670561761543686E-36
    Converged to right
    Iteration 20 complete. Error: 0.0 Total: 239401226744745.8400; Orientation: 0.0003; Line Search: 0.0525
    
```

Returns: 

```
    0.0
```



Training Converged

Next, we run the same optimization using L-BFGS, which is nearly ideal for purely second-order or quadratic functions.

Code from [LearningTester.java:249](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L249) executed in 1.59 seconds: 
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
    th(0)=86.39682240730848;dx=-64.52559482913409
    New Minimum: 86.39682240730848 > 3.4929801144309653
    END: th(2.154434690031884)=3.4929801144309653; dx=-12.435514900908522 delta=82.90384229287751
    Iteration 1 complete. Error: 3.4929801144309653 Total: 239401338375055.7800; Orientation: 0.0003; Line Search: 0.0474
    LBFGS Accumulation History: 1 points
    th(0)=3.4929801144309653;dx=-2.536114799108579
    New Minimum: 3.4929801144309653 > 1.675518137026345
    WOLF (strong): th(4.641588833612779)=1.675518137026345; dx=1.7529941727216685 delta=1.8174619774046203
    New Minimum: 1.675518137026345 > 0.09571408700419715
    END: th(2.3207944168063896)=0.09571408700419715; dx=-0.3915603131934551 delta=3.397266027426768
    Iteration 2 complete. Error: 0.09571408700419715 Total: 239401405271738.6600; Orientation: 0.0001; Line Search: 0.0489
    LBFGS Accumulation History: 1 points
    th(0)=0.09571408700419715;dx=-0.06705522685914096
    New Minimum: 0.0957140870041
```
...[skipping 11330 bytes](etc/13.txt)...
```
    450274E-35;dx=-4.817878191522467E-37
    New Minimum: 6.780345975450274E-35 > 2.0958683098020197E-35
    END: th(20.29263239761056)=2.0958683098020197E-35; dx=-1.5523974489659094E-37 delta=4.684477665648254E-35
    Iteration 25 complete. Error: 2.0958683098020197E-35 Total: 239402720280295.3000; Orientation: 0.0002; Line Search: 0.0277
    LBFGS Accumulation History: 1 points
    th(0)=2.0958683098020197E-35;dx=-8.08995881865166E-38
    New Minimum: 2.0958683098020197E-35 > 3.678369204190532E-36
    END: th(43.719151189477074)=3.678369204190532E-36; dx=-1.7789510616257504E-38 delta=1.7280313893829666E-35
    Iteration 26 complete. Error: 3.678369204190532E-36 Total: 239402763005965.2800; Orientation: 0.0002; Line Search: 0.0284
    LBFGS Accumulation History: 1 points
    th(0)=3.678369204190532E-36;dx=-1.0756812354761205E-38
    New Minimum: 3.678369204190532E-36 > 0.0
    END: th(94.19005594135811)=0.0; dx=0.0 delta=3.678369204190532E-36
    Iteration 27 complete. Error: 0.0 Total: 239402824907261.2500; Orientation: 0.0001; Line Search: 0.0487
    
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

![Result](etc/test.7.png)



Code from [LearningTester.java:98](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L98) executed in 0.01 seconds: 
```java
    return TestUtil.compareTime(runs);
```

Returns: 

![Result](etc/test.8.png)



### Performance
Now we execute larger-scale runs to benchmark performance:

Code from [PerformanceTester.java:66](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/PerformanceTester.java#L66) executed in 3.28 seconds: 
```java
    test(component, inputPrototype);
```
Logging: 
```
    100 batches
    Input Dimensions:
    	[100, 100, 3]
    Performance:
    	Evaluation performance: 0.235490s +- 0.012651s [0.220234s - 0.252505s]
    	Learning performance: 0.195506s +- 0.005234s [0.189204s - 0.201361s]
    
```

