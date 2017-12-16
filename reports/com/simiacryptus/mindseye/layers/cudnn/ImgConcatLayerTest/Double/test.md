# ImgConcatLayer
## Double
### Batch Execution
Code from [BatchingTester.java:66](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/BatchingTester.java#L66) executed in 0.02 seconds: 
```java
    return test(reference, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (160#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (160#)}
```



Code from [SingleDerivativeTester.java:77](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/SingleDerivativeTester.java#L77) executed in 0.03 seconds: 
```java
    return test(component, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ -1.508 ], [ -0.2 ] ],
    	[ [ 1.7 ], [ -0.768 ] ]
    ],
    [
    	[ [ 1.868 ], [ 0.304 ] ],
    	[ [ -1.976 ], [ -0.316 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.10118963034811938, negative=3, min=-0.768, max=-0.768, mean=-0.194, count=4.0, positive=1, stdDev=1.187786176043483, zeros=0},
    {meanExponent=-0.11256888040678964, negative=2, min=-0.316, max=-0.316, mean=-0.02999999999999997, count=4.0, positive=2, stdDev=1.376832596941255, zeros=0}
    Output: [
    	[ [ -1.508, 1.868 ], [ -0.2, 0.304 ] ],
    	[ [ 1.7, -1.976 ], [ -0.768, -0.316 ] ]
    ]
    Outputs Statistics: {meanExponent=-0.10687925537745452, negative=5, min=-0.316, max=-0.316, mean=-0.11199999999999999, count=8.0, positive=3, stdDev=1.2884005588325396, zeros=0}
    Feedback for input 0
    Inputs Values: [
    	[ [ -1.508 ], [ -0.2 ] ],
    	[ [ 1.7 ], [ -0.768 ] ]
    ]
    Value Statistics: {meanExponent=-0.10118963034811938, negative=3, min=-0.768, max=-0.768, mean=-0.194, count=4.0, positive=1, stdDev=1.187786176043483, zeros=0}
    Implemented Feedback: [ [ 1.0, 0.0, 0.0, 0.0, 0.
```
...[skipping 1936 bytes](etc/115.txt)...
```
    998899, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9999999999998899, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9999999999998899 ] ]
    Measured Statistics: {meanExponent=-4.7830642341045674E-14, negative=0, min=0.9999999999998899, max=0.9999999999998899, mean=0.12499999999998623, count=32.0, positive=4, stdDev=0.3307189138830374, zeros=28}
    Feedback Error: [ [ 0.0, 0.0, 0.0, 0.0, -1.1013412404281553E-13, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, -1.1013412404281553E-13, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.1013412404281553E-13, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.1013412404281553E-13 ] ]
    Error Statistics: {meanExponent=-12.958078098036825, negative=4, min=-1.1013412404281553E-13, max=-1.1013412404281553E-13, mean=-1.3766765505351941E-14, count=32.0, positive=0, stdDev=3.6423437884903677E-14, zeros=28}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 1.3767e-14 +- 3.6423e-14 [0.0000e+00 - 1.1013e-13] (64#)
    relativeTol: 5.5067e-14 +- 0.0000e+00 [5.5067e-14 - 5.5067e-14] (8#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=1.3767e-14 +- 3.6423e-14 [0.0000e+00 - 1.1013e-13] (64#), relativeTol=5.5067e-14 +- 0.0000e+00 [5.5067e-14 - 5.5067e-14] (8#)}
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
      "class": "com.simiacryptus.mindseye.layers.cudnn.ImgConcatLayer",
      "id": "c539fa48-1c4c-4e24-84d7-414249573901",
      "isFrozen": false,
      "name": "ImgConcatLayer/c539fa48-1c4c-4e24-84d7-414249573901",
      "maxBands": -1
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
    	[ [ 0.084 ], [ 1.632 ] ],
    	[ [ -1.904 ], [ 1.568 ] ]
    ],
    [
    	[ [ 1.556 ], [ -1.636 ] ],
    	[ [ -1.96 ], [ -1.968 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 0.084, 1.556 ], [ 1.632, -1.636 ] ],
    	[ [ -1.904, -1.96 ], [ 1.568, -1.968 ] ]
    ]
    --------------------
    Derivative: 
    [
    	[ [ 1.0 ], [ 1.0 ] ],
    	[ [ 1.0 ], [ 1.0 ] ]
    ],
    [
    	[ [ 1.0 ], [ 1.0 ] ],
    	[ [ 1.0 ], [ 1.0 ] ]
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
    	[ [ 0.48 ], [ -1.644 ], [ -1.332 ], [ -0.34 ], [ 0.016 ], [ 1.264 ], [ -0.456 ], [ -1.808 ], ... ],
    	[ [ -1.208 ], [ 1.36 ], [ 1.788 ], [ -0.308 ], [ -1.488 ], [ -0.204 ], [ -0.056 ], [ 1.656 ], ... ],
    	[ [ 1.268 ], [ -1.952 ], [ -0.608 ], [ -0.52 ], [ 0.272 ], [ 0.788 ], [ 1.372 ], [ -0.78 ], ... ],
    	[ [ 0.592 ], [ 0.152 ], [ 1.58 ], [ 1.524 ], [ -0.52 ], [ -1.5 ], [ 1.376 ], [ 1.44 ], ... ],
    	[ [ -1.14 ], [ -1.468 ], [ -1.676 ], [ 1.792 ], [ 1.28 ], [ 1.6 ], [ 0.052 ], [ -0.14 ], ... ],
    	[ [ 1.98 ], [ -0.412 ], [ -0.384 ], [ 0.568 ], [ 1.12 ], [ -0.348 ], [ 0.576 ], [ 1.456 ], ... ],
    	[ [ -1.912 ], [ 0.868 ], [ -1.372 ], [ 0.688 ], [ -0.936 ], [ 0.872 ], [ 0.564 ], [ -1.312 ], ... ],
    	[ [ 1.308 ], [ 1.776 ], [ 1.424 ], [ -0.904 ], [ 1.628 ], [ 1.416 ], [ 0.904 ], [ -0.588 ], ... ],
    	...
    ]
    [
    	[ [ 1.904 ], [ -1.736 ], [ -1.692 ], [ 0.352 ], [ 0.172 ], [ 1.492 ], [ 0.312 ], [ 0.116 ], ... ],
    	[ [ -0.172 ], [ -0.9 ], [ -0.964 ], [ 1.948 ], [ 1.144 ], [ 0.436 ], [ -0.76 ], [ -0.968 ], ... ],
    	[ [ -1.772 ], [ -0.12 ], [ 0.28 ], [ -1.148 ], [ 0.772 ], [ -1.008 ], [ 0.952 ], [ -0.756 ], ... ],
    	[ [ 0.788 ], [ 1.568 ], [ 0.864 ], [ 1.108 ], [ 0.384 ], [ 1.956 ], [ 0.444 ], [ -1.628 ], ... ],
    	[ [ 0.896 ], [ 0.864 ], [ -1.596 ], [ -1.108 ], [ -1.232 ], [ 0.252 ], [ -0.568 ], [ 1.568 ], ... ],
    	[ [ -0.448 ], [ 0.4 ], [ 1.268 ], [ -1.64 ], [ -1.94 ], [ -1.176 ], [ -0.844 ], [ 0.308 ], ... ],
    	[ [ -1.616 ], [ -0.748 ], [ 1.388 ], [ -0.148 ], [ -0.672 ], [ -1.64 ], [ -1.572 ], [ -0.232 ], ... ],
    	[ [ 1.476 ], [ 0.98 ], [ 0.272 ], [ -0.768 ], [ -1.58 ], [ 1.604 ], [ 1.724 ], [ -1.48 ], ... ],
    	...
    ]
```



First, we use a conjugate gradient descent method, which converges the fastest for purely linear functions.

Code from [LearningTester.java:225](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L225) executed in 0.08 seconds: 
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
    F(0.0) = LineSearchPoint{point=PointSample{avg=2.664920297599992}, derivative=-8.025670278400001E-4}
    New Minimum: 2.664920297599992 > 2.6649202975999082
    F(1.0E-10) = LineSearchPoint{point=PointSample{avg=2.6649202975999082}, derivative=-8.02567027839984E-4}, delta = -8.393286066166183E-14
    New Minimum: 2.6649202975999082 > 2.6649202975994406
    F(7.000000000000001E-10) = LineSearchPoint{point=PointSample{avg=2.6649202975994406}, derivative=-8.025670278398877E-4}, delta = -5.515587986337778E-13
    New Minimum: 2.6649202975994406 > 2.6649202975960686
    F(4.900000000000001E-9) = LineSearchPoint{point=PointSample{avg=2.6649202975960686}, derivative=-8.025670278392136E-4}, delta = -3.923528169025303E-12
    New Minimum: 2.6649202975960686 > 2.664920297572477
    F(3.430000000000001E-8) = LineSearchPoint{point=PointSample{avg=2.664920297572477}, derivative=-8.025670278344945E-4}, delta = -2.751532335310003E-11
    New Minimum: 2.664920297572477 > 2.6649202974072983
    F(2.4010000000000004
```
...[skipping 3388 bytes](etc/116.txt)...
```
    0
    F(420.3284198732498) = LineSearchPoint{point=PointSample{avg=0.6585027280000004}, derivative=-6.506027655704595E-32}, delta = 0.0
    F(2942.2989391127485) = LineSearchPoint{point=PointSample{avg=0.6585027280000013}, derivative=-2.9235060130780217E-32}, delta = 8.881784197001252E-16
    F(226.3306876240576) = LineSearchPoint{point=PointSample{avg=0.6585027280000005}, derivative=-6.781236924826794E-32}, delta = 1.1102230246251565E-16
    F(17.410052894158277) = LineSearchPoint{point=PointSample{avg=0.6585027280000004}, derivative=-7.08907892235216E-32}, delta = 0.0
    F(121.87037025910794) = LineSearchPoint{point=PointSample{avg=0.6585027280000004}, derivative=-6.930018333844101E-32}, delta = 0.0
    Loops = 12
    F(5000.850613177336) = LineSearchPoint{point=PointSample{avg=0.658502728000001}, derivative=4.173472954443352E-35}, delta = 5.551115123125783E-16
    0.658502728000001 > 0.6585027280000004
    Iteration 2 failed, aborting. Error: 0.6585027280000004 Total: 239575050477459.0000; Orientation: 0.0004; Line Search: 0.0375
    
```

Returns: 

```
    0.6585027280000004
```



This training run resulted in the following regressed input:

Code from [LearningTester.java:144](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L144) executed in 0.01 seconds: 
```java
    return Arrays.stream(input_gd).map(x -> x.prettyPrint()).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [
    	[ [ 1.192 ], [ -1.69 ], [ -1.512 ], [ 0.005999999999999966 ], [ 0.094 ], [ 1.3780000000000001 ], [ -0.07200000000000001 ], [ -0.8460000000000001 ], ... ],
    	[ [ -0.69 ], [ 0.23000000000000012 ], [ 0.4120000000000001 ], [ 0.82 ], [ -0.17200000000000004 ], [ 0.11600000000000002 ], [ -0.40800000000000003 ], [ 0.344 ], ... ],
    	[ [ -0.2519999999999999 ], [ -1.036 ], [ -0.16399999999999998 ], [ -0.834 ], [ 0.522 ], [ -0.11000000000000004 ], [ 1.162 ], [ -0.768 ], ... ],
    	[ [ 0.69 ], [ 0.86 ], [ 1.222 ], [ 1.316 ], [ -0.06799999999999999 ], [ 0.228 ], [ 0.9099999999999999 ], [ -0.09400000000000004 ], ... ],
    	[ [ -0.12199999999999989 ], [ -0.30200000000000005 ], [ -1.6360000000000001 ], [ 0.3419999999999999 ], [ 0.023999999999999907 ], [ 0.926 ], [ -0.25799999999999995 ], [ 0.7140000000000001 ], ... ],
    	[ [ 0.766 ], [ -0.005999999999999987 ], [ 0.44199999999999995 ], [ -0.536 ], [ -0.40999999999999986 ], [ -0.7619999999999999 ], [ -0.13400000000000004 ], [ 0.882 ], ... ],
    	[ [ -1.764 ], [ 0.060000000000000026 ], [ 
```
...[skipping 34 bytes](etc/117.txt)...
```
    9999999996 ], [ -0.804 ], [ -0.38399999999999995 ], [ -0.504 ], [ -0.772 ], ... ],
    	[ [ 1.392 ], [ 1.3780000000000001 ], [ 0.848 ], [ -0.836 ], [ 0.02399999999999999 ], [ 1.51 ], [ 1.314 ], [ -1.034 ], ... ],
    	...
    ]
    [
    	[ [ 1.86 ], [ -0.332 ], [ 1.108 ], [ -1.988 ], [ 0.9 ], [ -1.64 ], [ -1.084 ], [ 0.484 ], ... ],
    	[ [ 1.456 ], [ 1.38 ], [ -1.968 ], [ 0.264 ], [ 0.536 ], [ -0.612 ], [ 0.196 ], [ 1.012 ], ... ],
    	[ [ -1.236 ], [ -1.384 ], [ 0.344 ], [ 0.028 ], [ 1.22 ], [ 0.464 ], [ -0.124 ], [ -1.484 ], ... ],
    	[ [ -0.616 ], [ -1.932 ], [ -1.492 ], [ 0.268 ], [ -1.324 ], [ 1.46 ], [ 1.42 ], [ 0.02 ], ... ],
    	[ [ -1.856 ], [ 1.408 ], [ 1.44 ], [ -1.9 ], [ 1.844 ], [ -1.132 ], [ 1.448 ], [ -1.844 ], ... ],
    	[ [ -0.636 ], [ -0.424 ], [ 1.056 ], [ -1.3 ], [ 0.284 ], [ 1.216 ], [ -0.344 ], [ -1.496 ], ... ],
    	[ [ -1.776 ], [ -1.476 ], [ 1.78 ], [ 0.044 ], [ 0.892 ], [ -1.948 ], [ 0.796 ], [ -0.176 ], ... ],
    	[ [ -0.464 ], [ 0.964 ], [ -1.484 ], [ -1.156 ], [ -0.372 ], [ 0.248 ], [ -0.964 ], [ 1.04 ], ... ],
    	...
    ]
```



Next, we run the same optimization using L-BFGS, which is nearly ideal for purely second-order or quadratic functions.

Code from [LearningTester.java:249](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L249) executed in 0.14 seconds: 
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
    th(0)=2.664920297599992;dx=-8.025670278400001E-4
    New Minimum: 2.664920297599992 > 2.6631915918727675
    WOLFE (weak): th(2.154434690031884)=2.6631915918727675; dx=-8.022212121908292E-4 delta=0.001728705727224611
    New Minimum: 2.6631915918727675 > 2.6614636311827637
    WOLFE (weak): th(4.308869380063768)=2.6614636311827637; dx=-8.018753965416584E-4 delta=0.0034566664172284156
    New Minimum: 2.6614636311827637 > 2.654559238795009
    WOLFE (weak): th(12.926608140191302)=2.654559238795009; dx=-8.00492133944975E-4 delta=0.01036105880498317
    New Minimum: 2.654559238795009 > 2.6236369904220034
    WOLFE (weak): th(51.70643256076521)=2.6236369904220034; dx=-7.942674522598995E-4 delta=0.04128330717798878
    New Minimum: 2.6236369904220034 > 2.4627951761601743
    WOLFE (weak): th(258.53216280382605)=2.4627951761601743; dx=-7.610691499394973E-4 delta=0.20212512143981787
    New Minimum: 2.4627951761601743 > 1.6130976108418615
    END: th(1551.1929768229563)=1.6
```
...[skipping 2450 bytes](etc/118.txt)...
```
    00000002)=0.6585028401653313; dx=-4.4866132139709644E-10 delta=1.1104367699465989E-5
    Iteration 6 complete. Error: 0.6585028401653313 Total: 239575139570822.8800; Orientation: 0.0006; Line Search: 0.0079
    LBFGS Accumulation History: 1 points
    th(0)=0.6585028401653313;dx=-4.4866132139709473E-11
    New Minimum: 0.6585028401653313 > 0.6585028268966807
    WOLF (strong): th(9694.956105143481)=0.6585028268966807; dx=4.212890420070083E-11 delta=1.3268650622677569E-8
    New Minimum: 0.6585028268966807 > 0.6585027281043727
    END: th(4847.478052571741)=0.6585027281043727; dx=-1.3686139695041957E-12 delta=1.1206095862359433E-7
    Iteration 7 complete. Error: 0.6585027281043727 Total: 239575185900044.8800; Orientation: 0.0005; Line Search: 0.0439
    LBFGS Accumulation History: 1 points
    th(0)=0.6585027281043727;dx=-4.1748733581253175E-14
    MAX ALPHA: th(0)=0.6585027281043727;th'(0)=-4.1748733581253175E-14;
    Iteration 8 failed, aborting. Error: 0.6585027281043727 Total: 239575205119271.8000; Orientation: 0.0008; Line Search: 0.0139
    
```

Returns: 

```
    0.6585027281043727
```



This training run resulted in the following regressed input:

Code from [LearningTester.java:154](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L154) executed in 0.01 seconds: 
```java
    return Arrays.stream(input_lbgfs).map(x -> x.prettyPrint()).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [
    	[ [ 1.1919869310953435 ], [ -1.6899791561068118 ], [ -1.5119766894592443 ], [ 0.005989917037135957 ], [ 0.09398515684007568 ], [ 1.3779778290215707 ], [ -0.07201099172775735 ], [ -0.845985964631092 ], ... ],
    	[ [ -0.6899908113771898 ], [ 0.23000907322409372 ], [ 0.41199844211732584 ], [ 0.8199971727314429 ], [ -0.17198557516042356 ], [ 0.1159900757103713 ], [ -0.408009981988987 ], [ 0.3439919797891953 ], ... ],
    	[ [ -0.25200135593492023 ], [ -1.0359820554995667 ], [ -0.16401116482583225 ], [ -0.8339922250114682 ], [ 0.5219920230637142 ], [ -0.11000656330200727 ], [ 1.1619838009051555 ], [ -0.767990219958767 ], ... ],
    	[ [ 0.6900057266613119 ], [ 0.8599928452795701 ], [ 1.2219879264092743 ], [ 1.3159769779560357 ], [ -0.06800230797433224 ], [ 0.2279871330430977 ], [ 0.9099889361480447 ], [ -0.09400223585013441 ], ... ],
    	[ [ -0.12200486117093724 ], [ -0.3019921384624307 ], [ -1.635977410701223 ], [ 0.3420027839940382 ], [ 0.024002538771765586 ], [ 0.9260058132103495 ], [ -0.25799141722045194 ], [ 0.71400321
```
...[skipping 436 bytes](etc/119.txt)...
```
     ],
    	[ [ 1.3919852578139527 ], [ 1.3779936674954258 ], [ 0.8479803533684966 ], [ -0.8360051352428893 ], [ 0.023987681187001515 ], [ 1.5099806274404486 ], [ 1.3139770500802337 ], [ -1.033980310093978 ], ... ],
    	...
    ]
    [
    	[ [ 1.86 ], [ -0.332 ], [ 1.108 ], [ -1.988 ], [ 0.9 ], [ -1.64 ], [ -1.084 ], [ 0.484 ], ... ],
    	[ [ 1.456 ], [ 1.38 ], [ -1.968 ], [ 0.264 ], [ 0.536 ], [ -0.612 ], [ 0.196 ], [ 1.012 ], ... ],
    	[ [ -1.236 ], [ -1.384 ], [ 0.344 ], [ 0.028 ], [ 1.22 ], [ 0.464 ], [ -0.124 ], [ -1.484 ], ... ],
    	[ [ -0.616 ], [ -1.932 ], [ -1.492 ], [ 0.268 ], [ -1.324 ], [ 1.46 ], [ 1.42 ], [ 0.02 ], ... ],
    	[ [ -1.856 ], [ 1.408 ], [ 1.44 ], [ -1.9 ], [ 1.844 ], [ -1.132 ], [ 1.448 ], [ -1.844 ], ... ],
    	[ [ -0.636 ], [ -0.424 ], [ 1.056 ], [ -1.3 ], [ 0.284 ], [ 1.216 ], [ -0.344 ], [ -1.496 ], ... ],
    	[ [ -1.776 ], [ -1.476 ], [ 1.78 ], [ 0.044 ], [ 0.892 ], [ -1.948 ], [ 0.796 ], [ -0.176 ], ... ],
    	[ [ -0.464 ], [ 0.964 ], [ -1.484 ], [ -1.156 ], [ -0.372 ], [ 0.248 ], [ -0.964 ], [ 1.04 ], ... ],
    	...
    ]
```



Code from [LearningTester.java:95](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L95) executed in 0.00 seconds: 
```java
    return TestUtil.compare(runs);
```

Returns: 

![Result](etc/test.73.png)



Code from [LearningTester.java:98](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L98) executed in 0.00 seconds: 
```java
    return TestUtil.compareTime(runs);
```

Returns: 

![Result](etc/test.74.png)



### Performance
Now we execute larger-scale runs to benchmark performance:

Code from [PerformanceTester.java:66](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/PerformanceTester.java#L66) executed in 0.45 seconds: 
```java
    test(component, inputPrototype);
```
Logging: 
```
    100 batches
    Input Dimensions:
    	[100, 100, 1]
    	[100, 100, 1]
    Performance:
    	Evaluation performance: 0.013029s +- 0.000657s [0.012119s - 0.013876s]
    	Learning performance: 0.060000s +- 0.018757s [0.046626s - 0.097218s]
    
```

