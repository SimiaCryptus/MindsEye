# ReLuActivationLayer
## ReLuActivationLayerTest
### Batch Execution
Code from [BatchingTester.java:66](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/BatchingTester.java#L66) executed in 0.00 seconds: 
```java
    return test(reference, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (120#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (58#)}
```



Code from [SingleDerivativeTester.java:77](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/SingleDerivativeTester.java#L77) executed in 0.00 seconds: 
```java
    return test(component, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ -0.852 ], [ 1.952 ], [ 1.284 ] ],
    	[ [ -1.9 ], [ 1.076 ], [ 0.296 ] ]
    ]
    Inputs Statistics: {meanExponent=0.01855700252872426, negative=2, min=0.296, max=0.296, mean=0.3093333333333334, count=6.0, positive=4, stdDev=1.3207405330175779, zeros=0}
    Output: [
    	[ [ 0.0 ], [ 1.952 ], [ 1.284 ] ],
    	[ [ 0.0 ], [ 1.076 ], [ 0.296 ] ]
    ]
    Outputs Statistics: {meanExponent=-0.02446279513679589, negative=0, min=0.296, max=0.296, mean=0.7680000000000001, count=6.0, positive=4, stdDev=0.7263387639387008, zeros=2}
    Feedback for input 0
    Inputs Values: [
    	[ [ -0.852 ], [ 1.952 ], [ 1.284 ] ],
    	[ [ -1.9 ], [ 1.076 ], [ 0.296 ] ]
    ]
    Value Statistics: {meanExponent=0.01855700252872426, negative=2, min=0.296, max=0.296, mean=0.3093333333333334, count=6.0, positive=4, stdDev=1.3207405330175779, zeros=0}
    Implemented Feedback: [ [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 1.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 1.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 1.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0,
```
...[skipping 1306 bytes](etc/321.txt)...
```
    296 ] ]
    Implemented Statistics: {meanExponent=-0.02446279513679589, negative=0, min=0.296, max=0.296, mean=0.7680000000000001, count=6.0, positive=4, stdDev=0.7263387639387008, zeros=2}
    Measured Gradient: [ [ 0.0, 0.0, 1.95200000000062, 1.0759999999998549, 1.2839999999991747, 0.296000000000185 ] ]
    Measured Statistics: {meanExponent=-0.024462795136778, negative=0, min=0.296000000000185, max=0.296000000000185, mean=0.7679999999999724, count=6.0, positive=4, stdDev=0.7263387639387414, zeros=2}
    Gradient Error: [ [ 0.0, 0.0, 6.199485369506874E-13, -1.4521717162097048E-13, -8.253397965063414E-13, 1.8501866705378234E-13 ] ]
    Error Statistics: {meanExponent=-12.465444513101101, negative=2, min=1.8501866705378234E-13, max=1.8501866705378234E-13, mean=-2.7598294020473684E-14, count=6.0, positive=2, stdDev=4.31329709012208E-13, zeros=2}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 5.2763e-14 +- 1.5830e-13 [0.0000e+00 - 8.2534e-13] (42#)
    relativeTol: 1.3506e-13 +- 1.1008e-13 [5.5067e-14 - 3.2139e-13] (8#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=5.2763e-14 +- 1.5830e-13 [0.0000e+00 - 8.2534e-13] (42#), relativeTol=1.3506e-13 +- 1.1008e-13 [5.5067e-14 - 3.2139e-13] (8#)}
```



### Json Serialization
Code from [JsonTest.java:36](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/JsonTest.java#L36) executed in 0.00 seconds: 
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
      "class": "com.simiacryptus.mindseye.layers.java.ReLuActivationLayer",
      "id": "ad27d921-54b5-4014-b44f-08c0a43a5294",
      "isFrozen": true,
      "name": "ReLuActivationLayer/ad27d921-54b5-4014-b44f-08c0a43a5294",
      "weights": [
        1.0
      ]
    }
```



### Example Input/Output Pair
Code from [ReferenceIO.java:68](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/ReferenceIO.java#L68) executed in 0.00 seconds: 
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
    	[ [ 1.556 ], [ 0.164 ], [ 1.12 ] ],
    	[ [ -0.364 ], [ -1.14 ], [ 0.908 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 1.556 ], [ 0.164 ], [ 1.12 ] ],
    	[ [ 0.0 ], [ 0.0 ], [ 0.908 ] ]
    ]
    --------------------
    Derivative: 
    [
    	[ [ 1.0 ], [ 1.0 ], [ 1.0 ] ],
    	[ [ 0.0 ], [ 0.0 ], [ 1.0 ] ]
    ]
```



### Input Learning
In this test, we use a network to learn this target input, given it's pre-evaluated output:

Code from [LearningTester.java:127](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L127) executed in 0.00 seconds: 
```java
    return Arrays.stream(input_target).map(x -> x.prettyPrint()).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [
    	[ [ -0.532 ], [ -0.868 ], [ 1.68 ], [ -1.192 ], [ 0.864 ], [ 0.692 ], [ 1.08 ], [ -1.164 ], ... ],
    	[ [ -1.172 ], [ -1.06 ], [ 0.096 ], [ -1.124 ], [ 0.18 ], [ 0.1 ], [ -0.436 ], [ 0.452 ], ... ],
    	[ [ -1.032 ], [ -0.792 ], [ 1.064 ], [ 0.824 ], [ -0.86 ], [ -1.824 ], [ -1.612 ], [ 0.492 ], ... ],
    	[ [ 1.976 ], [ 1.292 ], [ -0.028 ], [ -0.82 ], [ 0.476 ], [ -0.976 ], [ -0.38 ], [ 0.56 ], ... ],
    	[ [ 0.352 ], [ 1.144 ], [ -1.948 ], [ -0.696 ], [ -1.472 ], [ 0.036 ], [ 0.42 ], [ -0.492 ], ... ],
    	[ [ -1.516 ], [ 0.176 ], [ 0.668 ], [ 1.98 ], [ 0.936 ], [ 0.42 ], [ -0.2 ], [ 0.232 ], ... ],
    	[ [ -1.272 ], [ -0.78 ], [ -0.592 ], [ -1.172 ], [ -1.18 ], [ 0.868 ], [ -0.992 ], [ 0.316 ], ... ],
    	[ [ 1.7 ], [ -0.192 ], [ -1.008 ], [ 0.092 ], [ -0.784 ], [ -1.784 ], [ 1.292 ], [ -0.348 ], ... ],
    	...
    ]
```



First, we use a conjugate gradient descent method, which converges the fastest for purely linear functions.

Code from [LearningTester.java:225](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L225) executed in 0.14 seconds: 
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
    F(0.0) = LineSearchPoint{point=PointSample{avg=0.804625235199995}, derivative=-1.9427283776000002E-4}
    New Minimum: 0.804625235199995 > 0.8046252351999732
    F(1.0E-10) = LineSearchPoint{point=PointSample{avg=0.8046252351999732}, derivative=-1.9427283775999614E-4}, delta = -2.1760371282653068E-14
    New Minimum: 0.8046252351999732 > 0.8046252351998601
    F(7.000000000000001E-10) = LineSearchPoint{point=PointSample{avg=0.8046252351998601}, derivative=-1.9427283775997283E-4}, delta = -1.3489209749195652E-13
    New Minimum: 0.8046252351998601 > 0.8046252351990476
    F(4.900000000000001E-9) = LineSearchPoint{point=PointSample{avg=0.8046252351990476}, derivative=-1.9427283775980963E-4}, delta = -9.47353306912646E-13
    New Minimum: 0.8046252351990476 > 0.8046252351933316
    F(3.430000000000001E-8) = LineSearchPoint{point=PointSample{avg=0.8046252351933316}, derivative=-1.9427283775866731E-4}, delta = -6.6633365491952645E-12
    New Minimum: 0.8046252351933316 > 0.8046252351533533
    F(2.40100
```
...[skipping 1602 bytes](etc/322.txt)...
```
    0.8045868218718109 > 0.8043563738047442
    F(1.3841287201) = LineSearchPoint{point=PointSample{avg=0.8043563738047442}, derivative=-1.9421905803714424E-4}, delta = -2.6886139525073016E-4
    Loops = 12
    New Minimum: 0.8043563738047442 > 0.31894314079999986
    F(5000.000000001327) = LineSearchPoint{point=PointSample{avg=0.31894314079999986}, derivative=1.7117989149495473E-17}, delta = -0.4856820943999951
    Right bracket at 5000.000000001327
    Converged to right
    Iteration 1 complete. Error: 0.31894314079999986 Total: 239709116250337.0000; Orientation: 0.0004; Line Search: 0.1264
    Zero gradient: 2.13163116971849E-15
    F(0.0) = LineSearchPoint{point=PointSample{avg=0.31894314079999986}, derivative=-4.543851443715417E-30}
    F(5000.000000001327) = LineSearchPoint{point=PointSample{avg=0.31894314079999986}, derivative=0.0}, delta = 0.0
    0.31894314079999986 <= 0.31894314079999986
    Converged to right
    Iteration 2 failed, aborting. Error: 0.31894314079999986 Total: 239709125730427.8800; Orientation: 0.0013; Line Search: 0.0053
    
```

Returns: 

```
    0.31894314079999986
```



This training run resulted in the following regressed input:

Code from [LearningTester.java:144](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L144) executed in 0.01 seconds: 
```java
    return Arrays.stream(input_gd).map(x -> x.prettyPrint()).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [
    	[ [ -4.289901767151605E-13 ], [ -0.828 ], [ 1.68 ], [ -0.74 ], [ -1.56 ], [ -1.776 ], [ 1.08 ], [ -8.709699628184353E-14 ], ... ],
    	[ [ -3.2374103398069565E-13 ], [ -0.94 ], [ 0.096 ], [ -4.246603069191224E-15 ], [ 0.18 ], [ -0.124 ], [ -1.532 ], [ 0.452 ], ... ],
    	[ [ -1.384 ], [ -4.3942627314663696E-13 ], [ 1.064 ], [ 0.824 ], [ -1.564 ], [ -1.0513812043200232E-13 ], [ -1.336 ], [ -1.5 ], ... ],
    	[ [ -0.944 ], [ -1.088 ], [ -9.764411501578252E-14 ], [ -0.804 ], [ -1.82 ], [ -2.049471703458039E-13 ], [ -5.095923683029469E-13 ], [ -1.896 ], ... ],
    	[ [ 0.352 ], [ -0.868 ], [ -0.8 ], [ -4.0878411766698264E-13 ], [ -0.252 ], [ -1.86 ], [ -0.312 ], [ -0.344 ], ... ],
    	[ [ -1.1785017406396037E-13 ], [ 0.176 ], [ 0.668 ], [ 1.98 ], [ -0.648 ], [ 0.42 ], [ -4.0545344859310717E-13 ], [ -1.668 ], ... ],
    	[ [ -3.2795988147427124E-13 ], [ -1.1146639167236572E-13 ], [ -0.032 ], [ -7.749356711883593E-14 ], [ -1.3801459974871477E-14 ], [ -0.74 ], [ -1.228 ], [ 0.316 ], ... ],
    	[ [ -1.284 ], [ -4.1300296516055823E-13 ], [ -0.508 ], [ -1.032 ], [ -5.064837438339964E-13 ], [ -1.484 ], [ 1.292 ], [ -1.824 ], ... ],
    	...
    ]
```



Next, we run the same optimization using L-BFGS, which is nearly ideal for purely second-order or quadratic functions.

Code from [LearningTester.java:249](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L249) executed in 0.08 seconds: 
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
    th(0)=0.804625235199995;dx=-1.9427283776000002E-4
    New Minimum: 0.804625235199995 > 0.8042067772324637
    WOLFE (weak): th(2.154434690031884)=0.8042067772324637; dx=-1.9418912813179981E-4 delta=4.184579675312605E-4
    New Minimum: 0.8042067772324637 > 0.8037884996118453
    WOLFE (weak): th(4.308869380063768)=0.8037884996118453; dx=-1.9410541850359959E-4 delta=8.36735588149673E-4
    New Minimum: 0.8037884996118453 > 0.8021171925986776
    WOLFE (weak): th(12.926608140191302)=0.8021171925986776; dx=-1.9377057999079872E-4 delta=0.0025080426013173174
    New Minimum: 0.8021171925986776 > 0.7946320197309151
    WOLFE (weak): th(51.70643256076521)=0.7946320197309151; dx=-1.9226380668319482E-4 delta=0.00999321546907983
    New Minimum: 0.7946320197309151 > 0.755697956153434
    WOLFE (weak): th(258.53216280382605)=0.755697956153434; dx=-1.842276823759741E-4 delta=0.04892727904656091
    New Minimum: 0.755697956153434 > 0.5500164971276089
    END: th(1551.192976822956
```
...[skipping 2408 bytes](etc/323.txt)...
```
    0.000000000002)=0.318943167951225; dx=-1.0860489538867689E-10 delta=2.687971160264535E-6
    Iteration 6 complete. Error: 0.318943167951225 Total: 239709217779872.7800; Orientation: 0.0007; Line Search: 0.0051
    LBFGS Accumulation History: 1 points
    th(0)=0.318943167951225;dx=-1.0860489538867551E-11
    New Minimum: 0.318943167951225 > 0.3189431487466073
    WOLF (strong): th(9694.956105143481)=0.3189431487466073; dx=3.3851684070155963E-12 delta=1.920461772941806E-8
    New Minimum: 0.3189431487466073 > 0.3189431408252632
    END: th(4847.478052571741)=0.3189431408252632; dx=-3.312926028984554E-13 delta=2.7125961810092747E-8
    Iteration 7 complete. Error: 0.3189431408252632 Total: 239709224822828.7800; Orientation: 0.0009; Line Search: 0.0051
    LBFGS Accumulation History: 1 points
    th(0)=0.3189431408252632;dx=-1.0105878592529637E-14
    MAX ALPHA: th(0)=0.3189431408252632;th'(0)=-1.0105878592529637E-14;
    Iteration 8 failed, aborting. Error: 0.3189431408252632 Total: 239709230471955.7800; Orientation: 0.0007; Line Search: 0.0040
    
```

Returns: 

```
    0.3189431408252632
```



This training run resulted in the following regressed input:

Code from [LearningTester.java:154](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L154) executed in 0.01 seconds: 
```java
    return Arrays.stream(input_lbgfs).map(x -> x.prettyPrint()).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [
    	[ [ 1.1655270377889412E-5 ], [ -0.828 ], [ 1.6799971438817638 ], [ -0.74 ], [ -1.56 ], [ -1.776 ], [ 1.0800021925756158 ], [ 2.365673690561706E-6 ], ... ],
    	[ [ 8.799152141723409E-6 ], [ -0.94 ], [ 0.09601061668192838 ], [ 1.1539871661276586E-7 ], [ 0.18001087632904075 ], [ -0.124 ], [ -1.532 ], [ 0.452003577360215 ], ... ],
    	[ [ -1.384 ], [ 1.1943767169421242E-5 ], [ 1.0639997115032085 ], [ 0.8239989614115505 ], [ -1.564 ], [ 2.8561182361659627E-6 ], [ -1.336 ], [ -1.5 ], ... ],
    	[ [ -0.944 ], [ -1.088 ], [ 2.654170482093617E-6 ], [ -0.804 ], [ -1.82 ], [ 5.567988076565956E-6 ], [ 1.3847845993531947E-5 ], [ -1.896 ], ... ],
    	[ [ 0.3520077605636922 ], [ -0.868 ], [ -0.8 ], [ 1.1107126473978751E-5 ], [ -0.252 ], [ -1.86 ], [ -0.312 ], [ -0.344 ], ... ],
    	[ [ 3.2023143860042505E-6 ], [ 0.1760017886801075 ], [ 0.6680032600137443 ], [ 1.9799875946379641 ], [ -0.648 ], [ 0.4199969707836889 ], [ 1.1020577436519159E-5 ], [ -1.668 ], ... ],
    	[ [ 8.914550858336162E-6 ], [ 3.02921631108512E-6 ], [ -0.032 ], [ 2.106026578182983E-6 ], [ 3.750458289914898E-7 ], [ -0.74 ], [ -1.228 ], [ 0.31600250992208634 ], ... ],
    	[ [ -1.284 ], [ 1.122252519059145E-5 ], [ -0.508 ], [ -1.032 ], [ 1.3761296956072355E-5 ], [ -1.484 ], [ 1.291993595371228 ], [ -1.824 ], ... ],
    	...
    ]
```



Code from [LearningTester.java:95](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L95) executed in 0.00 seconds: 
```java
    return TestUtil.compare(runs);
```

Returns: 

![Result](etc/test.225.png)



Code from [LearningTester.java:98](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L98) executed in 0.00 seconds: 
```java
    return TestUtil.compareTime(runs);
```

Returns: 

![Result](etc/test.226.png)



### Model Learning
In this test, attempt to train a network to emulate a randomized network given an example input/output. The target state is:

Code from [LearningTester.java:176](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L176) executed in 0.00 seconds: 
```java
    return network_target.state().stream().map(Arrays::toString).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [1.0]
```



First, we use a conjugate gradient descent method, which converges the fastest for purely linear functions.

Code from [LearningTester.java:225](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L225) executed in 0.00 seconds: 
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

Code from [LearningTester.java:189](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L189) executed in 0.00 seconds: 
```java
    return network_gd.state().stream().map(Arrays::toString).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [1.0]
    [0.532, 0.0, 0.0, 0.748, 1.116, 0.0, 0.0, 0.892, 1.772, 1.812, 0.0, 0.74, 0.0, 0.0, 0.0, 1.308, 0.0, 1.508, 0.892, 0.0, 1.736, 0.0, 0.884, 0.0, 1.452, 0.024, 0.0, 0.592, 0.74, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.212, 0.504, 0.624, 0.576, 0.0, 1.0, 0.744, 0.0, 0.0, 1.136, 0.0, 0.812, 1.716, 0.0, 1.368, 0.516, 0.056, 1.928, 0.0, 0.336, 0.0, 0.952, 0.0, 0.104, 1.884, 0.0, 0.884, 0.024, 0.0, 0.0, 0.656, 1.824, 0.708, 0.0, 0.0, 0.0, 0.0, 1.924, 0.0, 0.0, 1.776, 0.0, 0.1, 0.0, 0.344, 0.456, 0.0, 0.0, 0.396, 0.0, 0.58, 0.0, 0.0, 0.0, 0.0, 0.316, 0.996, 0.0, 1.124, 0.0, 0.0, 0.0, 0.0, 0.0, 1.784, 1.096, 0.0, 0.0, 1.096, 1.656, 0.0, 0.256, 0.0, 0.0, 0.0, 1.92, 0.9, 0.756, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.004, 1.404, 0.0, 0.0, 0.636, 1.692, 0.0, 1.1, 0.0, 0.0, 0.0, 1.556, 1.34, 0.0, 0.0, 0.0, 0.0, 0.0, 0.012, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.0, 1.448, 0.0, 0.0, 1.512, 0.0, 0.0, 0.0, 0.0, 0.276, 0.0, 1.544, 0.0, 0.116, 1.996, 0.0, 1.788, 0.0, 0.0, 0.88, 0.0, 1.896, 0.0, 0.0, 1.228, 1.044, 0
```
...[skipping 56541 bytes](etc/324.txt)...
```
    6, 1.78, 1.34, 0.032, 0.0, 0.18, 0.372, 0.0, 1.152, 1.0, 1.748, 0.0, 0.0, 0.0, 0.0, 0.692, 1.032, 0.0, 0.884, 0.756, 0.0, 0.0, 0.716, 0.0, 1.516, 0.0, 0.0, 0.86, 0.0, 0.0, 0.0, 0.0, 0.104, 0.616, 0.0, 0.0, 0.0, 0.0, 0.0, 0.584, 0.46, 0.404, 0.0, 0.544, 0.0, 1.148, 0.0, 0.0, 1.108, 1.508, 0.0, 0.0, 0.0, 1.476, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.592, 0.9, 0.668, 0.82, 1.58, 1.336, 0.544, 0.0, 0.0, 0.804, 0.0, 0.772, 0.0, 1.376, 0.0, 1.576, 0.156, 0.0, 0.212, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.692, 0.092, 0.0, 1.864, 0.868, 0.0, 0.0, 0.864, 1.716, 0.732, 0.7, 0.0, 0.0, 0.204, 1.948, 0.06, 0.0, 0.0, 1.632, 0.308, 1.152, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.756, 0.624, 0.0, 0.0, 1.932, 0.968, 0.0, 0.972, 0.268, 1.484, 0.0, 0.0, 0.244, 0.0, 0.0, 0.252, 0.0, 0.0, 0.0, 1.9, 0.0, 0.0, 0.168, 1.04, 0.0, 0.96, 1.096, 0.0, 0.0, 0.0, 0.88, 0.0, 0.764, 0.0, 1.372, 0.772, 0.0, 1.392, 1.42, 0.232, 0.0, 0.0, 0.0, 0.0, 0.0, 1.284, 0.0, 0.0, 1.596, 0.0, 0.904, 0.0, 0.0, 0.0, 0.0, 0.0]
```



Next, we run the same optimization using L-BFGS, which is nearly ideal for purely second-order or quadratic functions.

Code from [LearningTester.java:249](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L249) executed in 0.00 seconds: 
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

Code from [LearningTester.java:203](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L203) executed in 0.00 seconds: 
```java
    return network_lbfgs.state().stream().map(Arrays::toString).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [0.532, 0.0, 0.0, 0.748, 1.116, 0.0, 0.0, 0.892, 1.772, 1.812, 0.0, 0.74, 0.0, 0.0, 0.0, 1.308, 0.0, 1.508, 0.892, 0.0, 1.736, 0.0, 0.884, 0.0, 1.452, 0.024, 0.0, 0.592, 0.74, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.212, 0.504, 0.624, 0.576, 0.0, 1.0, 0.744, 0.0, 0.0, 1.136, 0.0, 0.812, 1.716, 0.0, 1.368, 0.516, 0.056, 1.928, 0.0, 0.336, 0.0, 0.952, 0.0, 0.104, 1.884, 0.0, 0.884, 0.024, 0.0, 0.0, 0.656, 1.824, 0.708, 0.0, 0.0, 0.0, 0.0, 1.924, 0.0, 0.0, 1.776, 0.0, 0.1, 0.0, 0.344, 0.456, 0.0, 0.0, 0.396, 0.0, 0.58, 0.0, 0.0, 0.0, 0.0, 0.316, 0.996, 0.0, 1.124, 0.0, 0.0, 0.0, 0.0, 0.0, 1.784, 1.096, 0.0, 0.0, 1.096, 1.656, 0.0, 0.256, 0.0, 0.0, 0.0, 1.92, 0.9, 0.756, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.004, 1.404, 0.0, 0.0, 0.636, 1.692, 0.0, 1.1, 0.0, 0.0, 0.0, 1.556, 1.34, 0.0, 0.0, 0.0, 0.0, 0.0, 0.012, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.0, 1.448, 0.0, 0.0, 1.512, 0.0, 0.0, 0.0, 0.0, 0.276, 0.0, 1.544, 0.0, 0.116, 1.996, 0.0, 1.788, 0.0, 0.0, 0.88, 0.0, 1.896, 0.0, 0.0, 1.228, 1.044, 0.0, 0.
```
...[skipping 56541 bytes](etc/325.txt)...
```
    8, 1.34, 0.032, 0.0, 0.18, 0.372, 0.0, 1.152, 1.0, 1.748, 0.0, 0.0, 0.0, 0.0, 0.692, 1.032, 0.0, 0.884, 0.756, 0.0, 0.0, 0.716, 0.0, 1.516, 0.0, 0.0, 0.86, 0.0, 0.0, 0.0, 0.0, 0.104, 0.616, 0.0, 0.0, 0.0, 0.0, 0.0, 0.584, 0.46, 0.404, 0.0, 0.544, 0.0, 1.148, 0.0, 0.0, 1.108, 1.508, 0.0, 0.0, 0.0, 1.476, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.592, 0.9, 0.668, 0.82, 1.58, 1.336, 0.544, 0.0, 0.0, 0.804, 0.0, 0.772, 0.0, 1.376, 0.0, 1.576, 0.156, 0.0, 0.212, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.692, 0.092, 0.0, 1.864, 0.868, 0.0, 0.0, 0.864, 1.716, 0.732, 0.7, 0.0, 0.0, 0.204, 1.948, 0.06, 0.0, 0.0, 1.632, 0.308, 1.152, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.756, 0.624, 0.0, 0.0, 1.932, 0.968, 0.0, 0.972, 0.268, 1.484, 0.0, 0.0, 0.244, 0.0, 0.0, 0.252, 0.0, 0.0, 0.0, 1.9, 0.0, 0.0, 0.168, 1.04, 0.0, 0.96, 1.096, 0.0, 0.0, 0.0, 0.88, 0.0, 0.764, 0.0, 1.372, 0.772, 0.0, 1.392, 1.42, 0.232, 0.0, 0.0, 0.0, 0.0, 0.0, 1.284, 0.0, 0.0, 1.596, 0.0, 0.904, 0.0, 0.0, 0.0, 0.0, 0.0]
    [1.0]
```



Code from [LearningTester.java:95](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L95) executed in 0.00 seconds: 
```java
    return TestUtil.compare(runs);
```

Code from [LearningTester.java:98](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L98) executed in 0.00 seconds: 
```java
    return TestUtil.compareTime(runs);
```

### Performance
Now we execute larger-scale runs to benchmark performance:

Code from [PerformanceTester.java:66](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/PerformanceTester.java#L66) executed in 0.36 seconds: 
```java
    test(component, inputPrototype);
```
Logging: 
```
    100 batches
    Input Dimensions:
    	[100, 100, 1]
    Performance:
    	Evaluation performance: 0.022120s +- 0.000414s [0.021715s - 0.022850s]
    	Learning performance: 0.019456s +- 0.001081s [0.018451s - 0.021301s]
    
```

### Function Plots
Code from [ActivationLayerTestBase.java:110](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/java/ActivationLayerTestBase.java#L110) executed in 0.00 seconds: 
```java
    return plot("Value Plot", plotData, x -> new double[]{x[0], x[1]});
```

Returns: 

![Result](etc/test.227.png)



Code from [ActivationLayerTestBase.java:114](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/java/ActivationLayerTestBase.java#L114) executed in 0.00 seconds: 
```java
    return plot("Derivative Plot", plotData, x -> new double[]{x[0], x[2]});
```

Returns: 

![Result](etc/test.228.png)



