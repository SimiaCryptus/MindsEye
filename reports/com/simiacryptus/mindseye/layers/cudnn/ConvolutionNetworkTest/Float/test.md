# PipelineNetwork
## Float
### Network Diagram
This is a network with the following layout:

Code from [StandardLayerTests.java:72](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/StandardLayerTests.java#L72) executed in 0.15 seconds: 
```java
    return Graphviz.fromGraph(TestUtil.toGraph((DAGNetwork) layer))
      .height(400).width(600).render(Format.PNG).toImage();
```

Returns: 

![Result](etc/test.60.png)



### Batch Execution
Code from [BatchingTester.java:66](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/BatchingTester.java#L66) executed in 0.01 seconds: 
```java
    return test(reference, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (60#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (34#)}
```



Code from [SingleDerivativeTester.java:77](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/SingleDerivativeTester.java#L77) executed in 0.02 seconds: 
```java
    return test(component, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ 0.348, 1.024, 0.0 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.22406039970680355, negative=0, min=0.0, max=0.0, mean=0.4573333333333333, count=3.0, positive=2, stdDev=0.4251347498800299, zeros=1}
    Output: [
    	[ [ 0.6679999999999999, 1.792, 0.0 ] ]
    ]
    Outputs Statistics: {meanExponent=0.03905723390082601, negative=0, min=0.0, max=0.0, mean=0.82, count=3.0, positive=2, stdDev=0.7394340177910852, zeros=1}
    Feedback for input 0
    Inputs Values: [
    	[ [ 0.348, 1.024, 0.0 ] ]
    ]
    Value Statistics: {meanExponent=-0.22406039970680355, negative=0, min=0.0, max=0.0, mean=0.4573333333333333, count=3.0, positive=2, stdDev=0.4251347498800299, zeros=1}
    Implemented Feedback: [ [ 1.0, 0.0, 0.0 ], [ 0.0, 1.0, 0.0 ], [ 0.0, 0.0, 0.0 ] ]
    Implemented Statistics: {meanExponent=0.0, negative=0, min=0.0, max=0.0, mean=0.2222222222222222, count=9.0, positive=2, stdDev=0.41573970964154905, zeros=7}
    Measured Feedback: [ [ 1.000000000000112, 0.0, 0.0 ], [ 0.0, 0.9999999999998899, 0.0 ], [ 0.0, 0.0, 0.0 ] ]
    Measured Statist
```
...[skipping 493 bytes](etc/91.txt)...
```
    : [ 0.32, 0.768, -1.688 ]
    Implemented Gradient: [ [ 1.0, 0.0, 0.0 ], [ 0.0, 1.0, 0.0 ], [ 0.0, 0.0, 0.0 ] ]
    Implemented Statistics: {meanExponent=0.0, negative=0, min=0.0, max=0.0, mean=0.2222222222222222, count=9.0, positive=2, stdDev=0.41573970964154905, zeros=7}
    Measured Gradient: [ [ 1.000000000000112, 0.0, 0.0 ], [ 0.0, 1.000000000000112, 0.0 ], [ 0.0, 0.0, 0.0 ] ]
    Measured Statistics: {meanExponent=4.860210431428295E-14, negative=0, min=0.0, max=0.0, mean=0.22222222222224708, count=9.0, positive=2, stdDev=0.41573970964159557, zeros=7}
    Gradient Error: [ [ 1.1191048088221578E-13, 0.0, 0.0 ], [ 0.0, 1.1191048088221578E-13, 0.0 ], [ 0.0, 0.0, 0.0 ] ]
    Error Statistics: {meanExponent=-12.951129238081498, negative=0, min=0.0, max=0.0, mean=2.4868995751603507E-14, count=9.0, positive=2, stdDev=4.6525630827818513E-14, zeros=7}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 2.4770e-14 +- 4.6342e-14 [0.0000e+00 - 1.1191e-13] (18#)
    relativeTol: 5.5733e-14 +- 3.8459e-16 [5.5067e-14 - 5.5955e-14] (4#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=2.4770e-14 +- 4.6342e-14 [0.0000e+00 - 1.1191e-13] (18#), relativeTol=5.5733e-14 +- 3.8459e-16 [5.5067e-14 - 5.5955e-14] (4#)}
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
      "id": "8cd90880-e00a-4041-9e59-825cc1334110",
      "isFrozen": false,
      "name": "PipelineNetwork/8cd90880-e00a-4041-9e59-825cc1334110",
      "inputs": [
        "cc099a81-7b3f-42c5-bd98-8101ab31ccc8"
      ],
      "nodes": {
        "f19add4e-cb7c-4dfb-bbcd-568a04aa8555": "37da8232-5c66-4b01-a1ab-08e1ed8d1257",
        "2ca69941-befa-48be-8d98-e485fc180722": "8bd60bee-d23e-44a6-87ce-0833d2ffca27",
        "a5d40692-e053-4291-be0c-ca37bc135909": "e0835c5a-2431-4e2b-9d22-686f533c4347"
      },
      "layers": {
        "37da8232-5c66-4b01-a1ab-08e1ed8d1257": {
          "class": "com.simiacryptus.mindseye.layers.cudnn.ImgConcatLayer",
          "id": "37da8232-5c66-4b01-a1ab-08e1ed8d1257",
          "isFrozen": false,
          "name": "ImgConcatLayer/37da8232-5c66-4b01-a1ab-08e1ed8d1257",
          "maxBands": -1
        },
        "8bd60bee-d23e-44a6-87ce-0833d2ffca27": {
          "class": "com.simiacryptus.mindseye.layers.cudnn.ImgBandBiasLayer",
          "id": "8bd60bee-d23e-44a6-87ce-0833d2ffca27",
          "isFrozen": false,
          "name": "ImgBandBiasLayer/8bd60bee-d23e-44a6-87ce-0833d2ffca27",
          "bias": [
            0.32,
            0.768,
            -1.688
          ]
        },
        "e0835c5a-2431-4e2b-9d22-686f533c4347": {
          "class": "com.simiacryptus.mindseye.layers.cudnn.ActivationLayer",
          "id": "e0835c5a-2431-4e2b-9d22-686f533c4347",
          "isFrozen": false,
          "name": "ActivationLayer/e0835c5a-2431-4e2b-9d22-686f533c4347",
          "mode": 1
        }
      },
      "links": {
        "f19add4e-cb7c-4dfb-bbcd-568a04aa8555": [
          "cc099a81-7b3f-42c5-bd98-8101ab31ccc8"
        ],
        "2ca69941-befa-48be-8d98-e485fc180722": [
          "f19add4e-cb7c-4dfb-bbcd-568a04aa8555"
        ],
        "a5d40692-e053-4291-be0c-ca37bc135909": [
          "2ca69941-befa-48be-8d98-e485fc180722"
        ]
      },
      "labels": {},
      "head": "a5d40692-e053-4291-be0c-ca37bc135909"
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
    	[ [ -1.136, 1.988, 0.472 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 0.0, 2.7560000000000002, 0.0 ] ]
    ]
    --------------------
    Derivative: 
    [
    	[ [ 0.0, 1.0, 0.0 ] ]
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
    	[ [ 0.988, 1.936, 1.808 ], [ 0.376, 0.516, 0.916 ], [ 0.288, -0.28, -1.856 ], [ 0.7, -1.216, -0.116 ], [ 0.5, -0.22, 0.888 ], [ -0.3, -0.52, 0.14 ], [ 0.34, 0.656, -0.504 ], [ 1.764, 1.196, 0.828 ], ... ],
    	[ [ 1.2, 0.696, 1.184 ], [ -0.472, 0.548, 1.6 ], [ 0.456, 0.632, 0.096 ], [ -0.62, -1.664, -1.244 ], [ -0.552, 0.0, 0.956 ], [ -0.732, -0.628, 1.952 ], [ 0.232, -1.796, -1.076 ], [ 0.004, 1.376, -1.26 ], ... ],
    	[ [ 1.552, -1.156, 0.448 ], [ -1.636, -0.724, 1.844 ], [ -1.76, 0.68, 1.296 ], [ 0.892, 1.376, 0.2 ], [ 0.488, -1.072, 1.56 ], [ -0.692, -0.64, 1.312 ], [ -1.42, -0.084, 0.312 ], [ 0.132, 1.368, 0.676 ], ... ],
    	[ [ -0.54, -1.492, -1.684 ], [ -0.468, 1.908, -1.208 ], [ 0.756, -0.792, -0.76 ], [ -1.124, -0.376, -0.304 ], [ -0.008, -0.9, 0.908 ], [ -0.94, 0.308, -1.56 ], [ 0.68, -1.472, 0.448 ], [ -1.872, -1.9, 0.988 ], ... ],
    	[ [ -0.8, 1.648, 0.244 ], [ -1.192, 0.388, -0.444 ], [ -0.188, 0.692, -0.724 ], [ 1.484, -0.496, 1.952 ], [ 1.644, 0.496, -1.556 ], [ 1.812, -0.632, -0.492 ], [ -1.14, -1.524, -1.576 ], [ -1.488, 0.996, 1.28 ], ... ],
    	[ [ -0.704, 1.328, -0.528 ], [ 0.568, 1.86, -0.524 ], [ 0.456, -0.484, -0.076 ], [ -0.92, -1.108, -1.52 ], [ -0.472, 1.828, 0.524 ], [ -1.376, -0.068, -1.744 ], [ 0.028, 1.208, 0.936 ], [ -0.468, 1.176, -0.044 ], ... ],
    	[ [ -0.636, -0.096, -1.26 ], [ 0.576, 1.844, 1.296 ], [ -1.188, -1.892, 1.98 ], [ -1.632, 0.812, -0.472 ], [ 0.0, 0.16, -1.652 ], [ -1.56, -0.712, -1.532 ], [ -0.568, 0.736, 0.656 ], [ -1.68, -1.012, 0.944 ], ... ],
    	[ [ 1.788, 1.744, -1.056 ], [ -1.708, 1.516, 1.86 ], [ -0.436, -1.868, 0.78 ], [ 1.356, 0.58, 0.008 ], [ 1.252, -0.24, -0.04 ], [ -1.224, 1.844, -1.5 ], [ 1.676, -1.248, -0.496 ], [ 0.656, -0.668, -1.216 ], ... ],
    	...
    ]
```



First, we use a conjugate gradient descent method, which converges the fastest for purely linear functions.

Code from [LearningTester.java:225](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L225) executed in 0.19 seconds: 
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
    F(0.0) = LineSearchPoint{point=PointSample{avg=0.9445442223999853}, derivative=-8.304618069333335E-5}
    New Minimum: 0.9445442223999853 > 0.9445442223999752
    F(1.0E-10) = LineSearchPoint{point=PointSample{avg=0.9445442223999752}, derivative=-8.304618069333279E-5}, delta = -1.0103029524088925E-14
    New Minimum: 0.9445442223999752 > 0.9445442223999313
    F(7.000000000000001E-10) = LineSearchPoint{point=PointSample{avg=0.9445442223999313}, derivative=-8.304618069332947E-5}, delta = -5.395683899678261E-14
    New Minimum: 0.9445442223999313 > 0.9445442223995791
    F(4.900000000000001E-9) = LineSearchPoint{point=PointSample{avg=0.9445442223995791}, derivative=-8.304618069330621E-5}, delta = -4.0611958240788226E-13
    New Minimum: 0.9445442223995791 > 0.9445442223971393
    F(3.430000000000001E-8) = LineSearchPoint{point=PointSample{avg=0.9445442223971393}, derivative=-8.304618069314345E-5}, delta = -2.8459457013241263E-12
    New Minimum: 0.9445442223971393 > 0.9445442223800561
    F(2.4010000
```
...[skipping 2759 bytes](etc/92.txt)...
```
    racket at 27222.095829360514
    F(19612.342773160366) = LineSearchPoint{point=PointSample{avg=0.32169786720000004}, derivative=2.1725583627146156E-30}, delta = 0.0
    Right bracket at 19612.342773160366
    F(17107.382632570083) = LineSearchPoint{point=PointSample{avg=0.32169786720000004}, derivative=9.926625330648372E-31}, delta = 0.0
    Right bracket at 17107.382632570083
    F(16034.614296250038) = LineSearchPoint{point=PointSample{avg=0.32169786720000004}, derivative=4.873566481461958E-31}, delta = 0.0
    Right bracket at 16034.614296250038
    F(15524.678717128207) = LineSearchPoint{point=PointSample{avg=0.32169786720000004}, derivative=2.471683647057991E-31}, delta = 0.0
    Right bracket at 15524.678717128207
    F(15270.296843185479) = LineSearchPoint{point=PointSample{avg=0.32169786720000004}, derivative=1.273503407284706E-31}, delta = 0.0
    Right bracket at 15270.296843185479
    Converged to right
    Iteration 2 failed, aborting. Error: 0.32169786720000004 Total: 239569122065745.9700; Orientation: 0.0009; Line Search: 0.0740
    
```

Returns: 

```
    0.32169786720000004
```



This training run resulted in the following regressed input:

Code from [LearningTester.java:144](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L144) executed in 0.01 seconds: 
```java
    return Arrays.stream(input_gd).map(x -> x.prettyPrint()).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [
    	[ [ 0.9879999999999939, 0.328, -1.108 ], [ -1.256, -0.808, 0.9160000000000086 ], [ 0.2879999999999889, 0.472, -0.32000000000001644 ], [ -1.728, 1.572, -0.688 ], [ 0.5000000000000092, 1.496, 0.8879999999999968 ], [ -0.3000000000000037, -1.708, -1.82 ], [ 0.3399999999999995, 1.6879999999999988, -0.320000000000001 ], [ 1.7640000000000189, -0.3, 0.8279999999999978 ], ... ],
    	[ [ -0.916, 0.188, -1.14 ], [ -0.4719999999999983, -0.488, 1.6000000000000099 ], [ -1.28, 0.248, -0.428 ], [ -0.620000000000012, -0.228, -0.32000000000000517 ], [ -1.82, -0.912, -0.624 ], [ -0.7320000000000093, -0.244, 1.9520000000000097 ], [ -0.8, -0.632, -0.3200000000000081 ], [ -1.956, -1.412, -1.968 ], ... ],
    	[ [ -1.068, 1.576, -0.792 ], [ -0.7680000000000048, 1.6879999999999995, 1.8440000000000156 ], [ -1.808, -0.24, -1.82 ], [ 0.8920000000000015, -0.524, 0.19999999999999282 ], [ 0.48799999999999266, 0.22, 1.5600000000000085 ], [ -0.6920000000000178, 1.248, -1.088 ], [ -0.7680000000000046, 0.308, -1.216 ], [ 0.13200000000000014, -0.6
```
...[skipping 791 bytes](etc/93.txt)...
```
    848 ], [ 0.567999999999989, -1.196, -0.32000000000001666 ], [ 0.4560000000000006, 0.42, -0.07600000000001235 ], [ -1.22, -0.496, -0.716 ], [ -1.54, 1.224, 0.5239999999999908 ], [ -0.7680000000000206, 0.052, -0.3200000000000013 ], [ 0.02799999999999378, -0.756, -0.488 ], [ -0.468000000000016, -0.544, -0.852 ], ... ],
    	[ [ -0.6360000000000012, 1.6879999999999995, -0.3200000000000081 ], [ -1.452, -0.068, 1.2960000000000107 ], [ -1.848, 1.176, -1.22 ], [ -1.116, 1.484, -0.32000000000001744 ], [ -0.992, 0.8, -1.568 ], [ -0.7680000000000182, -0.256, -0.86 ], [ -0.864, 0.004, 0.6559999999999967 ], [ -0.7680000000000132, -1.852, 0.9440000000000093 ], ... ],
    	[ [ 1.7880000000000074, -1.14, -1.024 ], [ -0.7680000000000075, -1.144, -0.684 ], [ -1.584, -1.584, -1.744 ], [ 1.3559999999999957, -1.984, 0.007999999999989007 ], [ 1.2520000000000153, 0.592, -0.040000000000010916 ], [ -0.7680000000000149, -0.228, -0.32000000000001644 ], [ -1.268, -0.448, -0.568 ], [ 0.6559999999999976, 1.096, -0.320000000000005 ], ... ],
    	...
    ]
```



Next, we run the same optimization using L-BFGS, which is nearly ideal for purely second-order or quadratic functions.

Code from [LearningTester.java:249](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L249) executed in 0.09 seconds: 
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
    th(0)=0.9445442223999853;dx=-8.304618069333335E-5
    New Minimum: 0.9445442223999853 > 0.944365317676309
    WOLFE (weak): th(2.154434690031884)=0.944365317676309; dx=-8.303425285516265E-5 delta=1.789047236762098E-4
    New Minimum: 0.944365317676309 > 0.9441864386503653
    WOLFE (weak): th(4.308869380063768)=0.9441864386503653; dx=-8.302232501699196E-5 delta=3.577837496199221E-4
    New Minimum: 0.9441864386503653 > 0.9434711795241039
    WOLFE (weak): th(12.926608140191302)=0.9434711795241039; dx=-8.29746136643092E-5 delta=0.0010730428758813915
    New Minimum: 0.9434711795241039 > 0.9402576016100761
    WOLFE (weak): th(51.70643256076521)=0.9402576016100761; dx=-8.275991257723674E-5 delta=0.004286620789909157
    New Minimum: 0.9402576016100761 > 0.9232591374807455
    WOLFE (weak): th(258.53216280382605)=0.9232591374807455; dx=-8.161484011285035E-5 delta=0.02128508491923975
    New Minimum: 0.9232591374807455 > 0.8223844265241942
    END: th(1551.1929768229563)
```
...[skipping 180 bytes](etc/94.txt)...
```
    ch: 0.0355
    LBFGS Accumulation History: 1 points
    th(0)=0.8223844265241942;dx=-6.675820790989225E-5
    New Minimum: 0.8223844265241942 > 0.6241354039870998
    END: th(3341.943960201201)=0.6241354039870998; dx=-5.188472859533755E-5 delta=0.19824902253709442
    Iteration 2 complete. Error: 0.6241354039870998 Total: 239569195323145.8000; Orientation: 0.0015; Line Search: 0.0103
    LBFGS Accumulation History: 1 points
    th(0)=0.6241354039870998;dx=-4.032500490494643E-5
    New Minimum: 0.6241354039870998 > 0.40347697714722874
    END: th(7200.000000000001)=0.40347697714722874; dx=-2.0969002550572138E-5 delta=0.22065842683987102
    Iteration 3 complete. Error: 0.40347697714722874 Total: 239569209592298.8000; Orientation: 0.0014; Line Search: 0.0102
    LBFGS Accumulation History: 1 points
    th(0)=0.40347697714722874;dx=-1.090388132629751E-5
    MAX ALPHA: th(0)=0.40347697714722874;th'(0)=-1.090388132629751E-5;
    Iteration 4 failed, aborting. Error: 0.40347697714722874 Total: 239569223294059.7800; Orientation: 0.0014; Line Search: 0.0096
    
```

Returns: 

```
    0.40347697714722874
```



This training run resulted in the following regressed input:

Code from [LearningTester.java:154](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L154) executed in 0.01 seconds: 
```java
    return Arrays.stream(input_lbgfs).map(x -> x.prettyPrint()).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [
    	[ [ 1.2778816422798631, 0.328, -1.108 ], [ -1.256, -0.808, 0.5029186597511948 ], [ 0.8184834053721497, 0.472, 0.462680434155631 ], [ -1.728, 1.572, -0.688 ], [ 0.0608293119460071, 1.496, 1.0372890457741295 ], [ -0.12317219820928335, -1.708, -1.82 ], [ 0.3646399395937884, 1.7401786956103755, -0.2736189372352219 ], [ 0.8784115828350174, -0.3, 0.9222115337409555 ], ... ],
    	[ [ -0.916, 0.188, -1.14 ], [ -0.5502680434155631, -0.488, 1.1390881887750175 ], [ -1.28, 0.248, -0.428 ], [ -0.051831981131467864, -0.228, -0.07505001227351549 ], [ -1.82, -0.912, -0.624 ], [ -0.2899304955232084, -0.244, 1.4838411477180204 ], [ -0.8, -0.632, 0.06554258423221809 ], [ -1.956, -1.412, -1.968 ], ... ],
    	[ [ -1.068, 1.576, -0.792 ], [ -0.538993502598908, 1.7097411231709898, 1.1033524039749494 ], [ -1.808, -0.24, -1.82 ], [ 0.8311248551212288, -0.524, 0.53916152146744 ], [ 0.8373073789472352, 0.22, 1.1599633336537887 ], [ 0.16025202830279806, 1.248, -1.088 ], [ -0.5534875847129012, 0.308, -1.216 ], [ 0.12475295894300342, -0.684,
```
...[skipping 774 bytes](etc/95.txt)...
```
    1, 0.288, -0.848 ], [ 1.085438731469556, -1.196, 0.4728262916354261 ], [ 0.4270118357720136, 0.42, 0.5110103256167231 ], [ -1.22, -0.496, -0.716 ], [ -1.54, 1.224, 0.963170688053993 ], [ 0.21179995090593806, 0.052, -0.260574263332628 ], [ 0.32512868333685985, -0.756, -0.488 ], [ 0.2914899027732418, -0.544, -0.852 ], ... ],
    	[ [ -0.5794730797554266, 1.706842306748191, 0.06554258423221809 ], [ -1.452, -0.068, 0.7901565342216386 ], [ -1.848, 1.176, -1.22 ], [ -1.116, 1.484, 0.509061496920409 ], [ -0.992, 0.8, -1.568 ], [ 0.097296702205392, -0.256, -0.86 ], [ -0.864, 0.004, 0.8110866786197268 ], [ -0.14185565267549527, -1.852, 0.5019304955232085 ], ... ],
    	[ [ 1.4502878867439593, -1.14, -1.024 ], [ -0.41434439641856674, -1.144, -0.684 ], [ -1.584, -1.584, -1.744 ], [ 1.5705124152870988, -1.984, 0.5312363643151532 ], [ 0.5345429353573383, 0.592, 0.4803375478923545 ], [ -0.060688792837133576, -0.228, 0.4612310259442317 ], [ -1.268, -0.448, -0.568 ], [ 0.7661550240663481, 1.096, -0.08229705333051207 ], ... ],
    	...
    ]
```



Code from [LearningTester.java:95](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L95) executed in 0.00 seconds: 
```java
    return TestUtil.compare(runs);
```

Returns: 

![Result](etc/test.61.png)



Code from [LearningTester.java:98](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L98) executed in 0.00 seconds: 
```java
    return TestUtil.compareTime(runs);
```

Returns: 

![Result](etc/test.62.png)



### Model Learning
In this test, attempt to train a network to emulate a randomized network given an example input/output. The target state is:

Code from [LearningTester.java:176](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L176) executed in 0.00 seconds: 
```java
    return network_target.state().stream().map(Arrays::toString).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [0.768, -1.688, 0.32]
```



First, we use a conjugate gradient descent method, which converges the fastest for purely linear functions.

Code from [LearningTester.java:225](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L225) executed in 0.20 seconds: 
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
    F(0.0) = LineSearchPoint{point=PointSample{avg=1.1084951973333128}, derivative=-0.4002336445741511}
    New Minimum: 1.1084951973333128 > 1.108495197293261
    F(1.0E-10) = LineSearchPoint{point=PointSample{avg=1.108495197293261}, derivative=-0.4005095050961371}, delta = -4.005173970256237E-11
    New Minimum: 1.108495197293261 > 1.108495197052968
    F(7.000000000000001E-10) = LineSearchPoint{point=PointSample{avg=1.108495197052968}, derivative=-0.40050950498965243}, delta = -2.8034485843875245E-10
    New Minimum: 1.108495197052968 > 1.1084951953708477
    F(4.900000000000001E-9) = LineSearchPoint{point=PointSample{avg=1.1084951953708477}, derivative=-0.4005095042442605}, delta = -1.9624650793304E-9
    New Minimum: 1.1084951953708477 > 1.1084951835958285
    F(3.430000000000001E-8) = LineSearchPoint{point=PointSample{avg=1.1084951835958285}, derivative=-0.4005094990265173}, delta = -1.3737484261255872E-8
    New Minimum: 1.1084951835958285 > 1.1084951011710233
    F(2.4010000000000004E-7) = Line
```
...[skipping 8746 bytes](etc/96.txt)...
```
    {point=PointSample{avg=8.786111999999998E-4}, derivative=5.878385219585322E-20}, delta = 4.3368086899420177E-19
    F(1.1932206268497567) = LineSearchPoint{point=PointSample{avg=8.786111999999994E-4}, derivative=-4.515405859250771E-21}, delta = 0.0
    F(8.352544387948297) = LineSearchPoint{point=PointSample{avg=8.786111999999996E-4}, derivative=2.713021752386244E-20}, delta = 2.1684043449710089E-19
    F(0.6425034144575613) = LineSearchPoint{point=PointSample{avg=8.786111999999994E-4}, derivative=-6.94599055844508E-21}, delta = 0.0
    F(4.497523901202929) = LineSearchPoint{point=PointSample{avg=8.786111999999994E-4}, derivative=1.0085958261277455E-20}, delta = 0.0
    8.786111999999994E-4 <= 8.786111999999994E-4
    F(2.21432060907452) = LineSearchPoint{point=PointSample{avg=8.786111999999994E-4}, derivative=-8.789236018905123E-24}, delta = 0.0
    Left bracket at 2.21432060907452
    Converged to left
    Iteration 5 failed, aborting. Error: 8.786111999999994E-4 Total: 239569551082758.5000; Orientation: 0.0000; Line Search: 0.0577
    
```

Returns: 

```
    8.786111999999994E-4
```



This training run resulted in the following configuration:

Code from [LearningTester.java:189](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L189) executed in 0.01 seconds: 
```java
    return network_gd.state().stream().map(Arrays::toString).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [0.7679999999998031, -14.449505699546025, 0.32]
    [0.44, 1.62, 1.3479999999999999, 1.1520000000000001, 0.0, 0.19600000000000006, 0.0, 0.0, 0.0, 0.0, 0.0, 0.09999999999999998, 2.3280000000000003, 1.848, 1.02, 2.036, 0.0, 2.596, 2.252, 2.64, 1.0, 0.0, 2.372, 0.0, 1.588, 2.7199999999999998, 0.8200000000000001, 0.0, 1.572, 0.928, 0.0, 0.612, 0.0, 0.0, 2.044, 0.0, 0.0, 0.0, 0.524, 1.204, 1.384, 2.3040000000000003, 0.0, 0.0, 1.56, 0.0, 0.0, 1.428, 1.308, 0.0, 0.0, 0.604, 1.3, 1.932, 0.6, 1.46, 2.416, 2.1, 2.3680000000000003, 0.0, 1.6760000000000002, 0.8280000000000001, 0.864, 0.06400000000000006, 1.6720000000000002, 0.304, 2.716, 0.392, 0.948, 1.284, 0.05600000000000005, 0.04800000000000004, 1.396, 0.0, 1.3519999999999999, 0.724, 1.1360000000000001, 0.436, 1.3319999999999999, 0.0, 0.0, 0.0, 0.656, 1.268, 2.3600000000000003, 1.736, 0.4, 1.708, 2.072, 0.0, 0.904, 1.9, 0.364, 1.2, 0.0, 2.3680000000000003, 0.0, 0.0, 0.0, 0.0, 1.6320000000000001, 1.508, 0.652, 0.0, 1.232, 1.8, 0.0, 2.512, 0.388, 0.856, 1.3319999999999999,
```
...[skipping 224892 bytes](etc/97.txt)...
```
    2.056, 1.82, 1.164, 0.11600000000000002, 0.0, 0.0, 0.0, 1.816, 0.0, 1.9040000000000001, 1.2, 2.02, 1.284, 0.784, 0.0, 0.472, 1.032, 0.476, 0.168, 0.0, 1.344, 0.0, 0.72, 0.0, 2.172, 0.0, 2.2039999999999997, 0.972, 1.96, 0.5720000000000001, 0.372, 0.0, 0.0, 0.0, 0.0, 1.004, 0.0, 0.5920000000000001, 2.024, 0.0, 0.0, 0.0, 2.296, 2.2199999999999998, 0.092, 0.0, 0.0, 0.0, 2.192, 0.0, 2.2239999999999998, 0.976, 1.832, 0.0, 0.0, 0.0, 0.0, 1.508, 2.272, 1.496, 0.0, 0.556, 1.452, 0.0, 0.0040000000000000036, 0.0, 0.484, 0.352, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.1959999999999997, 0.764, 0.228, 0.0, 0.0, 0.0, 1.6840000000000002, 0.0, 0.0, 1.696, 0.0, 0.0, 0.0, 0.784, 0.0, 0.968, 0.324, 0.428, 1.848, 0.16, 1.4400000000000002, 0.0, 0.0, 2.004, 1.12, 0.0, 1.5, 0.0, 1.4160000000000001, 1.6520000000000001, 0.132, 0.328, 0.0, 0.8400000000000001, 0.0, 0.096, 1.448, 0.0, 2.2079999999999997, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.688, 1.368, 0.0, 0.0, 2.128, 1.088, 1.6, 0.0, 0.392, 0.0, 0.6839999999999999, 0.0]
```



Next, we run the same optimization using L-BFGS, which is nearly ideal for purely second-order or quadratic functions.

Code from [LearningTester.java:249](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L249) executed in 1.11 seconds: 
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
    th(0)=0.6515030810666814;dx=-0.19780587496256
    New Minimum: 0.6515030810666814 > 0.3622643744782102
    END: th(2.154434690031884)=0.3622643744782102; dx=-0.08069658090290857 delta=0.28923870658847123
    Iteration 1 complete. Error: 0.3622643744782102 Total: 239569608499064.4400; Orientation: 0.0005; Line Search: 0.0182
    LBFGS Accumulation History: 1 points
    th(0)=0.3622643744782102;dx=-0.0479029296219696
    New Minimum: 0.3622643744782102 > 0.15221281246434087
    END: th(4.641588833612779)=0.15221281246434087; dx=-0.039778215188595215 delta=0.21005156201386932
    Iteration 2 complete. Error: 0.15221281246434087 Total: 239569626442158.4000; Orientation: 0.0001; Line Search: 0.0088
    LBFGS Accumulation History: 1 points
    th(0)=0.15221281246434087;dx=-0.04889643565940023
    Armijo: th(10.000000000000002)=0.2917770056997743; dx=0.1198685453228727 delta=-0.1395641932354334
    New Minimum: 0.15221281246434087 > 0.002063306666064437
    WOLF (strong): th(5
```
...[skipping 45692 bytes](etc/98.txt)...
```
    1456051374234E-34 delta=0.0
    Armijo: th(1.0420611727187088E-9)=5.1013005204292096E-33; dx=-5.2781456051374234E-34 delta=0.0
    Armijo: th(8.187623499932712E-10)=5.1013005204292096E-33; dx=-5.2781456051374234E-34 delta=0.0
    Armijo: th(7.071129386305524E-10)=5.1013005204292096E-33; dx=-5.2781456051374234E-34 delta=0.0
    Armijo: th(6.512882329491931E-10)=5.1013005204292096E-33; dx=-5.2781456051374234E-34 delta=0.0
    WOLFE (weak): th(6.233758801085133E-10)=5.1013005204292096E-33; dx=-5.2781456051374234E-34 delta=0.0
    WOLFE (weak): th(6.373320565288532E-10)=5.1013005204292096E-33; dx=-5.2781456051374234E-34 delta=0.0
    WOLFE (weak): th(6.443101447390231E-10)=5.1013005204292096E-33; dx=-5.2781456051374234E-34 delta=0.0
    WOLFE (weak): th(6.47799188844108E-10)=5.1013005204292096E-33; dx=-5.2781456051374234E-34 delta=0.0
    mu /= nu: th(0)=5.1013005204292096E-33;th'(0)=-5.2781456051374234E-34;
    Iteration 95 failed, aborting. Error: 5.1013005204292096E-33 Total: 239570677987339.4700; Orientation: 0.0000; Line Search: 0.1471
    
```

Returns: 

```
    5.1013005204292096E-33
```



Training Converged

Code from [LearningTester.java:95](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L95) executed in 0.00 seconds: 
```java
    return TestUtil.compare(runs);
```

Returns: 

![Result](etc/test.63.png)



Code from [LearningTester.java:98](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L98) executed in 0.00 seconds: 
```java
    return TestUtil.compareTime(runs);
```

Returns: 

![Result](etc/test.64.png)



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
    	Evaluation performance: 0.085654s +- 0.028180s [0.062365s - 0.140293s]
    	Learning performance: 0.080662s +- 0.016146s [0.072050s - 0.112947s]
    
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
    	Optional[ImgBandBiasLayer/8bd60bee-d23e-44a6-87ce-0833d2ffca27 -> 0.028679s +- 0.020002s (11.0)
    	ActivationLayer/e0835c5a-2431-4e2b-9d22-686f533c4347 -> 0.023034s +- 0.008348s (11.0)
    	ImgConcatLayer/37da8232-5c66-4b01-a1ab-08e1ed8d1257 -> 0.017621s +- 0.006566s (11.0)]
    Backward Performance: 
    	Optional[ImgBandBiasLayer/8bd60bee-d23e-44a6-87ce-0833d2ffca27 -> 0.000065s +- 0.000078s (6.0)
    	ActivationLayer/e0835c5a-2431-4e2b-9d22-686f533c4347 -> 0.000285s +- 0.000205s (6.0)
    	ImgConcatLayer/37da8232-5c66-4b01-a1ab-08e1ed8d1257 -> 0.000094s +- 0.000000s (1.0)]
    
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

