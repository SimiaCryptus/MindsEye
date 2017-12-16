# PipelineNetwork
## Double
### Network Diagram
This is a network with the following layout:

Code from [StandardLayerTests.java:72](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/StandardLayerTests.java#L72) executed in 1.81 seconds: 
```java
    return Graphviz.fromGraph(TestUtil.toGraph((DAGNetwork) layer))
      .height(400).width(600).render(Format.PNG).toImage();
```

Returns: 

![Result](etc/test.55.png)



### Batch Execution
Code from [BatchingTester.java:66](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/BatchingTester.java#L66) executed in 0.01 seconds: 
```java
    return test(reference, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (60#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (30#)}
```



Code from [SingleDerivativeTester.java:77](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/SingleDerivativeTester.java#L77) executed in 0.02 seconds: 
```java
    return test(component, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ 1.172, -0.784, 0.048 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.45183836275263417, negative=1, min=0.048, max=0.048, mean=0.1453333333333333, count=3.0, positive=2, stdDev=0.8014941602338029, zeros=0}
    Output: [
    	[ [ 0.688, 0.05599999999999994, 0.0 ] ]
    ]
    Outputs Statistics: {meanExponent=-0.7071117673791444, negative=0, min=0.0, max=0.0, mean=0.24799999999999997, count=3.0, positive=2, stdDev=0.311965810092495, zeros=1}
    Feedback for input 0
    Inputs Values: [
    	[ [ 1.172, -0.784, 0.048 ] ]
    ]
    Value Statistics: {meanExponent=-0.45183836275263417, negative=1, min=0.048, max=0.048, mean=0.1453333333333333, count=3.0, positive=2, stdDev=0.8014941602338029, zeros=0}
    Implemented Feedback: [ [ 1.0, 0.0, 0.0 ], [ 0.0, 1.0, 0.0 ], [ 0.0, 0.0, 0.0 ] ]
    Implemented Statistics: {meanExponent=0.0, negative=0, min=0.0, max=0.0, mean=0.2222222222222222, count=9.0, positive=2, stdDev=0.41573970964154905, zeros=7}
    Measured Feedback: [ [ 0.9999999999998899, 0.0, 0.0 ], [ 0.0, 0.9999999999998899, 0.0 ], [ 0.0, 
```
...[skipping 534 bytes](etc/85.txt)...
```
    484, 0.84, -0.696 ]
    Implemented Gradient: [ [ 1.0, 0.0, 0.0 ], [ 0.0, 1.0, 0.0 ], [ 0.0, 0.0, 0.0 ] ]
    Implemented Statistics: {meanExponent=0.0, negative=0, min=0.0, max=0.0, mean=0.2222222222222222, count=9.0, positive=2, stdDev=0.41573970964154905, zeros=7}
    Measured Gradient: [ [ 0.9999999999998899, 0.0, 0.0 ], [ 0.0, 0.9999999999998899, 0.0 ], [ 0.0, 0.0, 0.0 ] ]
    Measured Statistics: {meanExponent=-4.7830642341045674E-14, negative=0, min=0.0, max=0.0, mean=0.22222222222219776, count=9.0, positive=2, stdDev=0.41573970964150325, zeros=7}
    Gradient Error: [ [ -1.1013412404281553E-13, 0.0, 0.0 ], [ 0.0, -1.1013412404281553E-13, 0.0 ], [ 0.0, 0.0, 0.0 ] ]
    Error Statistics: {meanExponent=-12.958078098036825, negative=2, min=0.0, max=0.0, mean=-2.447424978729234E-14, count=9.0, positive=0, stdDev=4.5787128751186467E-14, zeros=7}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 2.4474e-14 +- 4.5787e-14 [0.0000e+00 - 1.1013e-13] (18#)
    relativeTol: 5.5067e-14 +- 0.0000e+00 [5.5067e-14 - 5.5067e-14] (4#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=2.4474e-14 +- 4.5787e-14 [0.0000e+00 - 1.1013e-13] (18#), relativeTol=5.5067e-14 +- 0.0000e+00 [5.5067e-14 - 5.5067e-14] (4#)}
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
      "class": "com.simiacryptus.mindseye.network.PipelineNetwork",
      "id": "92bef547-8c5c-4bbb-b81d-65c6fc85ce08",
      "isFrozen": false,
      "name": "PipelineNetwork/92bef547-8c5c-4bbb-b81d-65c6fc85ce08",
      "inputs": [
        "b1d98807-a076-412e-b357-3f87055d1e25"
      ],
      "nodes": {
        "14f0e7a1-cf44-4165-81df-2b942afa8e7d": "3829d9d0-f73a-43c7-8163-dd806d67f994",
        "58b68b52-b3fe-48ad-89dc-639b27879c45": "8b09446b-3428-491f-985e-e707139c595b",
        "f1b21cb8-3009-4980-8b74-6ed813a9a3f2": "5698f0af-bcd7-4cdd-b7c2-5bbcac6d5644"
      },
      "layers": {
        "3829d9d0-f73a-43c7-8163-dd806d67f994": {
          "class": "com.simiacryptus.mindseye.layers.cudnn.ImgConcatLayer",
          "id": "3829d9d0-f73a-43c7-8163-dd806d67f994",
          "isFrozen": false,
          "name": "ImgConcatLayer/3829d9d0-f73a-43c7-8163-dd806d67f994",
          "maxBands": -1
        },
        "8b09446b-3428-491f-985e-e707139c595b": {
          "class": "com.simiacryptus.mindseye.layers.cudnn.ImgBandBiasLayer",
          "id": "8b09446b-3428-491f-985e-e707139c595b",
          "isFrozen": false,
          "name": "ImgBandBiasLayer/8b09446b-3428-491f-985e-e707139c595b",
          "bias": [
            -0.484,
            0.84,
            -0.696
          ]
        },
        "5698f0af-bcd7-4cdd-b7c2-5bbcac6d5644": {
          "class": "com.simiacryptus.mindseye.layers.cudnn.ActivationLayer",
          "id": "5698f0af-bcd7-4cdd-b7c2-5bbcac6d5644",
          "isFrozen": false,
          "name": "ActivationLayer/5698f0af-bcd7-4cdd-b7c2-5bbcac6d5644",
          "mode": 1
        }
      },
      "links": {
        "14f0e7a1-cf44-4165-81df-2b942afa8e7d": [
          "b1d98807-a076-412e-b357-3f87055d1e25"
        ],
        "58b68b52-b3fe-48ad-89dc-639b27879c45": [
          "14f0e7a1-cf44-4165-81df-2b942afa8e7d"
        ],
        "f1b21cb8-3009-4980-8b74-6ed813a9a3f2": [
          "58b68b52-b3fe-48ad-89dc-639b27879c45"
        ]
      },
      "labels": {},
      "head": "f1b21cb8-3009-4980-8b74-6ed813a9a3f2"
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
    	[ [ -1.176, 1.176, 1.176 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 0.0, 2.016, 0.48 ] ]
    ]
    --------------------
    Derivative: 
    [
    	[ [ 0.0, 1.0, 1.0 ] ]
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
    	[ [ 0.416, -0.32, 1.376 ], [ -1.084, 0.26, -1.916 ], [ -0.552, -0.328, 0.708 ], [ 0.84, -1.256, 1.02 ], [ 0.636, 0.084, 1.408 ], [ 1.556, -1.848, -0.356 ], [ 1.836, -0.988, 1.644 ], [ -0.52, -0.688, -0.904 ], ... ],
    	[ [ 0.86, 1.124, 0.328 ], [ 0.78, 1.896, -1.06 ], [ 1.524, -0.568, 1.352 ], [ 0.02, 1.26, -0.828 ], [ -0.608, 1.972, 1.948 ], [ 0.252, -1.452, -1.196 ], [ 0.432, 1.388, 0.26 ], [ -0.66, 1.32, -1.736 ], ... ],
    	[ [ -1.42, 0.06, -1.164 ], [ -1.66, -1.492, 0.456 ], [ 0.588, 1.096, -0.06 ], [ 1.26, -0.236, 0.644 ], [ -1.38, 0.304, -1.856 ], [ -0.088, 0.048, -0.932 ], [ 0.676, 1.292, 0.812 ], [ 0.448, 0.056, 1.828 ], ... ],
    	[ [ -0.1, -0.22, 1.496 ], [ 1.236, 0.392, -1.592 ], [ -0.328, -0.428, -0.224 ], [ 0.82, -0.968, -1.472 ], [ 0.336, -0.028, -1.36 ], [ -1.008, -1.264, -1.748 ], [ -0.324, 1.72, -0.188 ], [ 0.752, 0.776, -0.404 ], ... ],
    	[ [ -0.98, 0.136, 0.952 ], [ -1.64, 0.448, 0.684 ], [ -0.164, 0.26, 1.568 ], [ 0.992, -0.912, 1.6 ], [ -0.892, -1.716, 1.676 ], [ -1.196, 0.608, -0.72 ], [ -0.668, 0.896, -1.252 ], [ 0.12, 1.316, 0.956 ], ... ],
    	[ [ -0.972, 0.98, 1.852 ], [ 0.376, 1.6, 1.504 ], [ 0.048, -1.66, -1.764 ], [ 1.196, -0.972, -0.368 ], [ 0.884, -0.996, -0.816 ], [ 1.404, 0.564, 0.128 ], [ -0.64, -1.896, 1.436 ], [ 0.924, -1.932, -0.324 ], ... ],
    	[ [ 0.016, 0.78, -0.34 ], [ -1.992, 0.112, -0.048 ], [ 1.956, -1.468, -0.116 ], [ -0.808, -1.36, 0.88 ], [ -0.436, 0.144, 0.712 ], [ 1.564, 1.096, -0.112 ], [ -1.324, 1.312, -0.888 ], [ 1.28, -1.488, -1.796 ], ... ],
    	[ [ 1.112, 0.436, -0.104 ], [ -0.04, -0.924, -0.824 ], [ 1.016, -1.384, 1.148 ], [ 0.084, -1.664, -0.2 ], [ -1.804, -0.28, -0.2 ], [ 0.368, 1.664, -0.48 ], [ -1.952, 1.948, 1.768 ], [ 0.348, -1.956, 0.744 ], ... ],
    	...
    ]
```



First, we use a conjugate gradient descent method, which converges the fastest for purely linear functions.

Code from [LearningTester.java:225](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L225) executed in 0.13 seconds: 
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
    F(0.0) = LineSearchPoint{point=PointSample{avg=0.8291731850666618}, derivative=-7.194980096000001E-5}
    New Minimum: 0.8291731850666618 > 0.8291731850666538
    F(1.0E-10) = LineSearchPoint{point=PointSample{avg=0.8291731850666538}, derivative=-7.194980095999952E-5}, delta = -7.993605777301127E-15
    New Minimum: 0.8291731850666538 > 0.8291731850666157
    F(7.000000000000001E-10) = LineSearchPoint{point=PointSample{avg=0.8291731850666157}, derivative=-7.194980095999665E-5}, delta = -4.6074255521943996E-14
    New Minimum: 0.8291731850666157 > 0.8291731850663137
    F(4.900000000000001E-9) = LineSearchPoint{point=PointSample{avg=0.8291731850663137}, derivative=-7.19498009599765E-5}, delta = -3.481659405224491E-13
    New Minimum: 0.8291731850663137 > 0.829173185064195
    F(3.430000000000001E-8) = LineSearchPoint{point=PointSample{avg=0.829173185064195}, derivative=-7.194980095983548E-5}, delta = -2.4668045384146353E-12
    New Minimum: 0.829173185064195 > 0.8291731850493888
    F(2.401000000000
```
...[skipping 1600 bytes](etc/86.txt)...
```
    583338759 > 0.8290736018754944
    F(1.3841287201) = LineSearchPoint{point=PointSample{avg=0.8290736018754944}, derivative=-7.194316177427239E-5}, delta = -9.958319116742942E-5
    Loops = 12
    New Minimum: 0.8290736018754944 > 0.2895496778666659
    F(14999.99999998484) = LineSearchPoint{point=PointSample{avg=0.2895496778666659}, derivative=-7.271146254773777E-17}, delta = -0.539623507199996
    Right bracket at 14999.99999998484
    Converged to right
    Iteration 1 complete. Error: 0.2895496778666659 Total: 239566111298126.0000; Orientation: 0.0011; Line Search: 0.0885
    Zero gradient: 8.572116888256246E-15
    F(0.0) = LineSearchPoint{point=PointSample{avg=0.2895496778666659}, derivative=-7.348118794592795E-29}
    F(14999.99999998484) = LineSearchPoint{point=PointSample{avg=0.2895496778666659}, derivative=2.1244791125704148E-35}, delta = 0.0
    0.2895496778666659 <= 0.2895496778666659
    Converged to right
    Iteration 2 failed, aborting. Error: 0.2895496778666659 Total: 239566127888427.8800; Orientation: 0.0012; Line Search: 0.0106
    
```

Returns: 

```
    0.2895496778666659
```



This training run resulted in the following regressed input:

Code from [LearningTester.java:144](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L144) executed in 0.01 seconds: 
```java
    return Arrays.stream(input_gd).map(x -> x.prettyPrint()).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [
    	[ [ -0.676, 0.696, -1.0 ], [ -1.988, -1.776, -0.84 ], [ -1.172, -0.692, 0.708 ], [ 0.84, 0.164, 1.02 ], [ 0.636, -1.748, -1.976 ], [ -1.36, -1.256, -0.356 ], [ -1.304, 0.048, -1.944 ], [ 0.0, 0.46, -0.84 ], ... ],
    	[ [ -1.784, 0.456, -1.796 ], [ -1.776, 1.896, -0.84 ], [ -1.472, 0.696, 1.3520000000000003 ], [ 0.484, -1.408, -0.828 ], [ -0.256, 1.972, -1.008 ], [ -0.892, -1.348, -0.84 ], [ 0.484, 0.288, 0.26 ], [ 0.38, -1.912, -0.84 ], ... ],
    	[ [ 0.484, -1.812, -0.84 ], [ -1.864, 0.444, 0.45600000000000007 ], [ 0.408, -1.816, -0.05999999999999994 ], [ 0.248, 0.696, 0.6440000000000001 ], [ 0.484, -0.724, -0.84 ], [ 0.484, -0.396, -0.992 ], [ -1.436, 1.292, 0.812 ], [ -1.036, -0.632, 1.828 ], ... ],
    	[ [ -1.868, 0.696, 1.496 ], [ -0.876, -0.968, -0.84 ], [ -1.576, 0.068, -0.22399999999999998 ], [ -1.392, -1.68, -0.84 ], [ 0.484, -0.136, -1.164 ], [ 0.24, -0.376, -0.84 ], [ 0.484, -1.428, -0.18800000000000006 ], [ -0.548, -1.632, -0.404 ], ... ],
    	[ [ 0.484, 0.696, -1.552 ], [ 0.484, 0.696, -1.852 ], [ -0.572, 0.696, 1.5679999999999998 ], [ 0.992, -1.884, -1.708 ], [ 0.484, 0.696, 1.6760000000000002 ], [ 0.484, 0.696, -0.72 ], [ 0.484, 0.896, -0.84 ], [ -1.372, 0.516, 0.9559999999999997 ], ... ],
    	[ [ 0.484, 0.452, -1.212 ], [ -0.684, 1.6, 1.5039999999999998 ], [ 0.032, 0.696, -0.84 ], [ -1.272, -1.528, -0.368 ], [ 0.884, 0.696, -1.724 ], [ -0.272, 0.06, -0.9 ], [ 0.364, 0.696, -0.864 ], [ 0.144, 0.508, -0.32399999999999995 ], ... ],
    	[ [ -0.168, 0.78, -0.34 ], [ -1.816, 0.616, -0.04800000000000004 ], [ 1.956, 0.472, -0.11599999999999999 ], [ 0.484, -0.92, -1.556 ], [ 0.484, -1.568, -1.972 ], [ 1.564, 0.44, -0.11200000000000004 ], [ -1.848, -0.228, -1.928 ], [ 1.28, 0.696, -0.84 ], ... ],
    	[ [ -0.78, -0.648, -0.10400000000000001 ], [ -1.736, 0.696, -0.824 ], [ -1.656, -1.844, -1.08 ], [ 0.484, 0.696, -0.20000000000000007 ], [ -1.264, 0.516, -0.20000000000000007 ], [ -0.948, -1.912, -1.288 ], [ -0.176, 1.948, 1.768 ], [ 0.112, 0.108, 0.7440000000000001 ], ... ],
    	...
    ]
```



Next, we run the same optimization using L-BFGS, which is nearly ideal for purely second-order or quadratic functions.

Code from [LearningTester.java:249](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L249) executed in 0.15 seconds: 
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
    th(0)=0.8291731850666618;dx=-7.194980096000001E-5
    New Minimum: 0.8291731850666618 > 0.8290181850515842
    WOLFE (weak): th(2.154434690031884)=0.8290181850515842; dx=-7.193946688352474E-5 delta=1.5500001507762207E-4
    New Minimum: 0.8290181850515842 > 0.8288632073005938
    WOLFE (weak): th(4.308869380063768)=0.8288632073005938; dx=-7.192913280704946E-5 delta=3.0997776606800365E-4
    New Minimum: 0.8288632073005938 > 0.8282435189375674
    WOLFE (weak): th(12.926608140191302)=0.8282435189375674; dx=-7.188779650114836E-5 delta=9.296661290943975E-4
    New Minimum: 0.8282435189375674 > 0.8254593295943113
    WOLFE (weak): th(51.70643256076521)=0.8254593295943113; dx=-7.170178312459342E-5 delta=0.00371385547235048
    New Minimum: 0.8254593295943113 > 0.8107321488796784
    WOLFE (weak): th(258.53216280382605)=0.8107321488796784; dx=-7.07097117829671E-5 delta=0.01844103618698345
    New Minimum: 0.8107321488796784 > 0.7233360119998486
    END: th(1551.19297682295
```
...[skipping 184 bytes](etc/87.txt)...
```
    h: 0.0380
    LBFGS Accumulation History: 1 points
    th(0)=0.7233360119998486;dx=-5.783817788442417E-5
    New Minimum: 0.7233360119998486 > 0.5515764248283268
    END: th(3341.943960201201)=0.5515764248283268; dx=-4.495204793443123E-5 delta=0.17175958717152184
    Iteration 2 complete. Error: 0.5515764248283268 Total: 239566201611481.8000; Orientation: 0.0014; Line Search: 0.0100
    LBFGS Accumulation History: 1 points
    th(0)=0.5515764248283268;dx=-3.4936899594888076E-5
    New Minimum: 0.5515764248283268 > 0.36040171024509665
    END: th(7200.000000000001)=0.36040171024509665; dx=-1.8167187789341796E-5 delta=0.19117471458323015
    Iteration 3 complete. Error: 0.36040171024509665 Total: 239566215698533.8000; Orientation: 0.0015; Line Search: 0.0093
    LBFGS Accumulation History: 1 points
    th(0)=0.36040171024509665;dx=-9.446937650457732E-6
    MAX ALPHA: th(0)=0.36040171024509665;th'(0)=-9.446937650457732E-6;
    Iteration 4 failed, aborting. Error: 0.36040171024509665 Total: 239566291018606.7800; Orientation: 0.0539; Line Search: 0.0174
    
```

Returns: 

```
    0.36040171024509665
```



This training run resulted in the following regressed input:

Code from [LearningTester.java:154](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L154) executed in 0.01 seconds: 
```java
    return Arrays.stream(input_lbgfs).map(x -> x.prettyPrint()).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [
    	[ [ -0.676, 0.810503248700546, -1.0 ], [ -1.988, -1.776, -0.7356426087792491 ], [ -1.172, -0.692, 0.561609770648669 ], [ 1.160319214719249, 0.164, 0.9518778140642321 ], [ 0.7505032487005461, -1.748, -1.976 ], [ -1.36, -1.256, 0.11360826049337858 ], [ -1.304, 0.048, -1.944 ], [ 0.0, 0.46, -0.19501334592730407 ], ... ],
    	[ [ -1.784, 0.456, -1.796 ], [ -1.776, 1.9017976328455972, -0.03267962625058052 ], [ -1.472, 1.1235754223627983, 1.2432943841450514 ], [ 0.878239033500614, -1.408, 0.17499048228832714 ], [ -0.256, 1.5893562321905805, -1.008 ], [ -0.892, -1.348, -0.13268879283713342 ], [ 0.5245834299191808, 0.288, 0.8644032241535151 ], [ 0.38, -1.912, -0.432716292596792 ], ... ],
    	[ [ 0.8854860745576107, -1.812, 0.07457658139296874 ], [ -1.864, 0.444, 0.23134172723310598 ], [ 0.408, -1.816, 0.5009209778115357 ], [ 0.248, 1.1119801566716039, 0.28164794715017083 ], [ 0.9927422822011601, -0.724, -0.13558760925993218 ], [ 0.7985215818736517, -0.396, -0.992 ], [ -1.436, 1.2905505917886009, 0.6366216064206828 ], [ 
```
...[skipping 893 bytes](etc/88.txt)...
```
    99908334134473, 1.2749935025989079 ], [ 0.032, 0.8771760264249145, -0.753035507316041 ], [ -1.272, -1.528, -0.5172890457741296 ], [ 1.1376464369948804, 0.9409499877264844, -1.724 ], [ -0.272, 0.06, -0.9 ], [ 0.364, 0.7989079830093515, -0.864 ], [ 0.144, 0.508, 0.174596424721365 ], ... ],
    	[ [ -0.168, 1.1611943595980203, -0.04142190845174076 ], [ -1.816, 0.616, 0.3723283813058019 ], [ 1.7806216064206828, 0.472, 0.44202216138873673 ], [ 0.942012994802184, -0.92, -1.556 ], [ 0.8217121132560408, -1.568, -1.972 ], [ 1.326297053330512, 0.44, 0.07787247569331052 ], [ -1.848, -0.228, -1.928 ], [ 1.2944940821139932, 1.0569026446384298, -0.6327346257698977 ], ... ],
    	[ [ -0.78, -0.648, -0.1576281038217747 ], [ -1.736, 0.9264559056124914, -0.09639707787754315 ], [ -1.656, -1.844, -1.08 ], [ 0.7956227654508531, 1.016319214719249, 0.013063007075699462 ], [ -1.264, 0.516, -0.12173195658443692 ], [ -0.948, -1.912, -1.288 ], [ -0.176, 1.655219541297338, 1.4752195412973381 ], [ 0.112, 0.108, 1.009241702686075 ], ... ],
    	...
    ]
```



Code from [LearningTester.java:95](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L95) executed in 0.01 seconds: 
```java
    return TestUtil.compare(runs);
```

Returns: 

![Result](etc/test.56.png)



Code from [LearningTester.java:98](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L98) executed in 0.00 seconds: 
```java
    return TestUtil.compareTime(runs);
```

Returns: 

![Result](etc/test.57.png)



### Model Learning
In this test, attempt to train a network to emulate a randomized network given an example input/output. The target state is:

Code from [LearningTester.java:176](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L176) executed in 0.00 seconds: 
```java
    return network_target.state().stream().map(Arrays::toString).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [0.84, -0.696, -0.484]
```



First, we use a conjugate gradient descent method, which converges the fastest for purely linear functions.

Code from [LearningTester.java:225](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L225) executed in 0.39 seconds: 
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
    F(0.0) = LineSearchPoint{point=PointSample{avg=0.01037562826666505}, derivative=-0.004646664070399998}
    New Minimum: 0.01037562826666505 > 0.010375628266200435
    F(1.0E-10) = LineSearchPoint{point=PointSample{avg=0.010375628266200435}, derivative=-0.004657316397543177}, delta = -4.646161927412962E-13
    New Minimum: 0.010375628266200435 > 0.010375628263407061
    F(7.000000000000001E-10) = LineSearchPoint{point=PointSample{avg=0.010375628263407061}, derivative=-0.004657316396882256}, delta = -3.2579893644024693E-12
    New Minimum: 0.010375628263407061 > 0.0103756282438472
    F(4.900000000000001E-9) = LineSearchPoint{point=PointSample{avg=0.0103756282438472}, derivative=-0.004657316392255808}, delta = -2.281785003999115E-11
    New Minimum: 0.0103756282438472 > 0.010375628106919249
    F(3.430000000000001E-8) = LineSearchPoint{point=PointSample{avg=0.010375628106919249}, derivative=-0.004657316359870673}, delta = -1.597458016561193E-10
    New Minimum: 0.010375628106919249 > 0.01037562714
```
...[skipping 19181 bytes](etc/89.txt)...
```
    5810868881199E-33 <= 2.963420701708862E-31
    F(4.214982396945403) = LineSearchPoint{point=PointSample{avg=1.6531155479983025E-33}, derivative=-4.12983926976733E-33}, delta = -2.946889546228879E-31
    Left bracket at 4.214982396945403
    F(4.311971746282389) = LineSearchPoint{point=PointSample{avg=3.666148884003692E-33}, derivative=2.810667853606653E-33}, delta = -2.926759212868825E-31
    Right bracket at 4.311971746282389
    Converged to right
    Iteration 14 complete. Error: 1.6531155479983025E-33 Total: 239566888534792.1200; Orientation: 0.0000; Line Search: 0.0259
    Zero gradient: 3.6649577704928083E-17
    F(0.0) = LineSearchPoint{point=PointSample{avg=3.666148884003692E-33}, derivative=-1.3431915459495618E-33}
    New Minimum: 3.666148884003692E-33 > 0.0
    F(4.311971746282389) = LineSearchPoint{point=PointSample{avg=0.0}, derivative=0.0}, delta = -3.666148884003692E-33
    0.0 <= 3.666148884003692E-33
    Converged to right
    Iteration 15 complete. Error: 0.0 Total: 239566895138881.1200; Orientation: 0.0001; Line Search: 0.0038
    
```

Returns: 

```
    0.0
```



Training Converged

Next, we run the same optimization using L-BFGS, which is nearly ideal for purely second-order or quadratic functions.

Code from [LearningTester.java:249](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L249) executed in 0.47 seconds: 
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
    th(0)=0.7154140282666372;dx=-0.3929288790300442
    New Minimum: 0.7154140282666372 > 0.14397585902209914
    END: th(2.154434690031884)=0.14397585902209914; dx=-0.1452707474456885 delta=0.571438169244538
    Iteration 1 complete. Error: 0.14397585902209914 Total: 239566907742905.1000; Orientation: 0.0001; Line Search: 0.0048
    LBFGS Accumulation History: 1 points
    th(0)=0.14397585902209914;dx=-0.0848835980395377
    New Minimum: 0.14397585902209914 > 0.05451080541017006
    WOLF (strong): th(4.641588833612779)=0.05451080541017006; dx=0.06647200227189372 delta=0.08946505361192908
    New Minimum: 0.05451080541017006 > 0.009761658846692457
    END: th(2.3207944168063896)=0.009761658846692457; dx=-0.024584120435497306 delta=0.1342142001754067
    Iteration 2 complete. Error: 0.009761658846692457 Total: 239566916587779.1000; Orientation: 0.0001; Line Search: 0.0063
    LBFGS Accumulation History: 1 points
    th(0)=0.009761658846692457;dx=-0.007236245222068819
    Arm
```
...[skipping 18757 bytes](etc/90.txt)...
```
    5572)=1.2623828808810202E-32; dx=-1.2309389125899143E-32 delta=6.094279184876158E-32
    Iteration 39 complete. Error: 1.2623828808810202E-32 Total: 239567356777654.6600; Orientation: 0.0001; Line Search: 0.0040
    LBFGS Accumulation History: 1 points
    th(0)=1.2623828808810202E-32;dx=-5.318325413614744E-33
    New Minimum: 1.2623828808810202E-32 > 1.5238984882628816E-33
    WOLF (strong): th(6.116365378330731)=1.5238984882628816E-33; dx=1.672738298669654E-33 delta=1.109993032054732E-32
    END: th(3.0581826891653656)=1.5238984882628816E-33; dx=-1.672738298669654E-33 delta=1.109993032054732E-32
    Iteration 40 complete. Error: 1.5238984882628816E-33 Total: 239567364740520.6600; Orientation: 0.0000; Line Search: 0.0062
    LBFGS Accumulation History: 1 points
    th(0)=1.5238984882628816E-33;dx=-5.2611549655711545E-34
    New Minimum: 1.5238984882628816E-33 > 0.0
    END: th(6.588654873992858)=0.0; dx=0.0 delta=1.5238984882628816E-33
    Iteration 41 complete. Error: 0.0 Total: 239567370877245.6200; Orientation: 0.0001; Line Search: 0.0040
    
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

![Result](etc/test.58.png)



Code from [LearningTester.java:98](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L98) executed in 0.00 seconds: 
```java
    return TestUtil.compareTime(runs);
```

Returns: 

![Result](etc/test.59.png)



### Performance
Adding performance wrappers

Code from [TestUtil.java:287](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/TestUtil.java#L287) executed in 0.00 seconds: 
```java
    network.visitNodes(node -> {
      if (!(node.getLayer() instanceof MonitoringWrapperLayer)) {
        node.setLayer(new MonitoringWrapperLayer(node.getLayer()).shouldRecordSignalMetrics(false));
      }
      else {
        ((MonitoringWrapperLayer) node.getLayer()).shouldRecordSignalMetrics(false);
      }
    });
```

Now we execute larger-scale runs to benchmark performance:

Code from [PerformanceTester.java:66](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/PerformanceTester.java#L66) executed in 1.17 seconds: 
```java
    test(component, inputPrototype);
```
Logging: 
```
    100 batches
    Input Dimensions:
    	[100, 100, 3]
    Performance:
    	Evaluation performance: 0.087577s +- 0.019176s [0.066690s - 0.112421s]
    	Learning performance: 0.081295s +- 0.012552s [0.071925s - 0.106144s]
    
```

Per-layer Performance Metrics:

Code from [TestUtil.java:252](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/TestUtil.java#L252) executed in 0.00 seconds: 
```java
    Map<NNLayer, MonitoringWrapperLayer> metrics = new HashMap<>();
    network.visitNodes(node -> {
      if ((node.getLayer() instanceof MonitoringWrapperLayer)) {
        MonitoringWrapperLayer layer = node.getLayer();
        metrics.put(layer.getInner(), layer);
      }
    });
    System.out.println("Forward Performance: \n\t" + metrics.entrySet().stream().map(e -> {
      PercentileStatistics performance = e.getValue().getForwardPerformance();
      return String.format("%s -> %.6fs +- %.6fs (%s)", e.getKey(), performance.getMean(), performance.getStdDev(), performance.getCount());
    }).reduce((a, b) -> a + "\n\t" + b));
    System.out.println("Backward Performance: \n\t" + metrics.entrySet().stream().map(e -> {
      PercentileStatistics performance = e.getValue().getBackwardPerformance();
      return String.format("%s -> %.6fs +- %.6fs (%s)", e.getKey(), performance.getMean(), performance.getStdDev(), performance.getCount());
    }).reduce((a, b) -> a + "\n\t" + b));
```
Logging: 
```
    Forward Performance: 
    	Optional[ImgConcatLayer/3829d9d0-f73a-43c7-8163-dd806d67f994 -> 0.020408s +- 0.009818s (11.0)
    	ActivationLayer/5698f0af-bcd7-4cdd-b7c2-5bbcac6d5644 -> 0.027562s +- 0.014768s (11.0)
    	ImgBandBiasLayer/8b09446b-3428-491f-985e-e707139c595b -> 0.020945s +- 0.006945s (11.0)]
    Backward Performance: 
    	Optional[ImgConcatLayer/3829d9d0-f73a-43c7-8163-dd806d67f994 -> 0.000092s +- 0.000000s (1.0)
    	ActivationLayer/5698f0af-bcd7-4cdd-b7c2-5bbcac6d5644 -> 0.000197s +- 0.000007s (6.0)
    	ImgBandBiasLayer/8b09446b-3428-491f-985e-e707139c595b -> 0.000067s +- 0.000082s (6.0)]
    
```

Removing performance wrappers

Code from [TestUtil.java:270](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/TestUtil.java#L270) executed in 0.00 seconds: 
```java
    network.visitNodes(node -> {
      if (node.getLayer() instanceof MonitoringWrapperLayer) {
        node.setLayer(node.<MonitoringWrapperLayer>getLayer().getInner());
      }
    });
```

