# LinearActivationLayer
## LinearActivationLayerTest
### Batch Execution
Code from [BatchingTester.java:66](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/BatchingTester.java#L66) executed in 0.00 seconds: 
```java
    return test(reference, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (120#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (120#)}
```



Code from [SingleDerivativeTester.java:77](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/SingleDerivativeTester.java#L77) executed in 0.00 seconds: 
```java
    return test(component, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ 0.5 ], [ 1.484 ], [ -0.772 ] ],
    	[ [ 1.664 ], [ -0.656 ], [ -0.528 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.08021461842017652, negative=3, min=-0.528, max=-0.528, mean=0.282, count=6.0, positive=3, stdDev=1.0040843258080137, zeros=0}
    Output: [
    	[ [ 0.5 ], [ 1.484 ], [ -0.772 ] ],
    	[ [ 1.664 ], [ -0.656 ], [ -0.528 ] ]
    ]
    Outputs Statistics: {meanExponent=-0.08021461842017652, negative=3, min=-0.528, max=-0.528, mean=0.282, count=6.0, positive=3, stdDev=1.0040843258080137, zeros=0}
    Feedback for input 0
    Inputs Values: [
    	[ [ 0.5 ], [ 1.484 ], [ -0.772 ] ],
    	[ [ 1.664 ], [ -0.656 ], [ -0.528 ] ]
    ]
    Value Statistics: {meanExponent=-0.08021461842017652, negative=3, min=-0.528, max=-0.528, mean=0.282, count=6.0, positive=3, stdDev=1.0040843258080137, zeros=0}
    Implemented Feedback: [ [ 1.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 1.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 1.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 1.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 1.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 1.0 ] ]
    Implement
```
...[skipping 1739 bytes](etc/266.txt)...
```
    .9999999999998899, 0.9999999999998899, 0.9999999999998899, 0.9999999999998899, 0.9999999999998899 ] ]
    Measured Statistics: {meanExponent=-0.04010730921015158, negative=3, min=0.9999999999998899, max=0.9999999999998899, mean=0.6409999999999286, count=12.0, positive=9, stdDev=0.7955964219794183, zeros=0}
    Gradient Error: [ [ -5.5067062021407764E-14, -1.000310945187266E-12, 4.849454171562684E-13, 1.0103029524088925E-14, 5.10702591327572E-15, 3.601563491884008E-13 ], [ -1.1013412404281553E-13, -1.1013412404281553E-13, -1.1013412404281553E-13, -1.1013412404281553E-13, -1.1013412404281553E-13, -1.1013412404281553E-13 ] ]
    Error Statistics: {meanExponent=-13.004386503554938, negative=8, min=-1.1013412404281553E-13, max=-1.1013412404281553E-13, mean=-7.132257747362776E-14, count=12.0, positive=4, stdDev=3.3916581561288336E-13, zeros=0}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 6.7444e-14 +- 1.6431e-13 [0.0000e+00 - 1.0003e-12] (48#)
    relativeTol: 8.5106e-14 +- 8.8969e-14 [3.3077e-15 - 3.4106e-13] (18#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=6.7444e-14 +- 1.6431e-13 [0.0000e+00 - 1.0003e-12] (48#), relativeTol=8.5106e-14 +- 8.8969e-14 [3.3077e-15 - 3.4106e-13] (18#)}
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
      "class": "com.simiacryptus.mindseye.layers.java.LinearActivationLayer",
      "id": "51f2eb47-c2a0-4ea1-9685-3460f2a545d2",
      "isFrozen": false,
      "name": "LinearActivationLayer/51f2eb47-c2a0-4ea1-9685-3460f2a545d2",
      "weights": [
        1.0,
        0.0
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
    	[ [ 0.272 ], [ 0.744 ], [ -1.288 ] ],
    	[ [ -0.488 ], [ -0.652 ], [ -1.776 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 0.272 ], [ 0.744 ], [ -1.288 ] ],
    	[ [ -0.488 ], [ -0.652 ], [ -1.776 ] ]
    ]
    --------------------
    Derivative: 
    [
    	[ [ 1.0 ], [ 1.0 ], [ 1.0 ] ],
    	[ [ 1.0 ], [ 1.0 ], [ 1.0 ] ]
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
    	[ [ -1.596 ], [ 1.416 ], [ -0.352 ], [ -1.784 ], [ -0.92 ], [ 1.068 ], [ 0.196 ], [ 0.996 ], ... ],
    	[ [ 0.036 ], [ 0.836 ], [ 1.16 ], [ -0.472 ], [ -1.896 ], [ -0.596 ], [ 0.6 ], [ -1.12 ], ... ],
    	[ [ -1.712 ], [ 1.92 ], [ 0.644 ], [ -1.188 ], [ 0.724 ], [ -1.556 ], [ -1.988 ], [ -1.684 ], ... ],
    	[ [ 1.712 ], [ 1.156 ], [ -1.732 ], [ -0.772 ], [ -1.504 ], [ -1.96 ], [ 1.696 ], [ 1.652 ], ... ],
    	[ [ 1.984 ], [ 0.064 ], [ 1.664 ], [ -1.824 ], [ -0.068 ], [ -0.316 ], [ -1.536 ], [ 0.128 ], ... ],
    	[ [ -1.404 ], [ 1.968 ], [ -1.7 ], [ 1.036 ], [ -1.272 ], [ -1.58 ], [ 0.284 ], [ 1.724 ], ... ],
    	[ [ 0.952 ], [ 0.728 ], [ 0.232 ], [ -1.456 ], [ 0.94 ], [ 0.84 ], [ 1.96 ], [ 0.004 ], ... ],
    	[ [ -0.672 ], [ -0.288 ], [ 0.376 ], [ 1.188 ], [ -0.732 ], [ 1.68 ], [ 0.94 ], [ 0.612 ], ... ],
    	...
    ]
```



First, we use a conjugate gradient descent method, which converges the fastest for purely linear functions.

Code from [LearningTester.java:225](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L225) executed in 0.06 seconds: 
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
    F(0.0) = LineSearchPoint{point=PointSample{avg=2.729096607999998}, derivative=-0.0010916386432}
    New Minimum: 2.729096607999998 > 2.7290966079998835
    F(1.0E-10) = LineSearchPoint{point=PointSample{avg=2.7290966079998835}, derivative=-0.0010916386431999783}, delta = -1.1457501614131615E-13
    New Minimum: 2.7290966079998835 > 2.729096607999234
    F(7.000000000000001E-10) = LineSearchPoint{point=PointSample{avg=2.729096607999234}, derivative=-0.0010916386431998474}, delta = -7.642775301519578E-13
    New Minimum: 2.729096607999234 > 2.7290966079946544
    F(4.900000000000001E-9) = LineSearchPoint{point=PointSample{avg=2.7290966079946544}, derivative=-0.0010916386431989303}, delta = -5.3437254621258035E-12
    New Minimum: 2.7290966079946544 > 2.729096607962559
    F(3.430000000000001E-8) = LineSearchPoint{point=PointSample{avg=2.729096607962559}, derivative=-0.0010916386431925114}, delta = -3.743894083640953E-11
    New Minimum: 2.729096607962559 > 2.729096607737901
    F(2.4010000000000004E-
```
...[skipping 8003 bytes](etc/267.txt)...
```
    hPoint{point=PointSample{avg=3.50298808849682E-299}, derivative=2.831730888388334E-290}, delta = -1.4306892552737812E-274
    3.50298808849682E-299 <= 1.4306892552737812E-274
    Converged to right
    Iteration 12 complete. Error: 3.50298808849682E-299 Total: 239662800068854.2000; Orientation: 0.0003; Line Search: 0.0019
    Zero gradient: 1.1837209280057222E-151
    F(0.0) = LineSearchPoint{point=PointSample{avg=3.50298808849682E-299}, derivative=-1.4011952353987283E-302}
    New Minimum: 3.50298808849682E-299 > 1.0E-323
    F(5000.000000002474) = LineSearchPoint{point=PointSample{avg=1.0E-323}, derivative=6.933675654E-315}, delta = -3.50298808849682E-299
    1.0E-323 <= 3.50298808849682E-299
    Converged to right
    Iteration 13 complete. Error: 1.0E-323 Total: 239662803283125.2000; Orientation: 0.0003; Line Search: 0.0022
    Zero gradient: 0.0
    F(0.0) = LineSearchPoint{point=PointSample{avg=1.0E-323}, derivative=0.0}
    Iteration 14 failed, aborting. Error: 1.0E-323 Total: 239662806086173.2000; Orientation: 0.0005; Line Search: 0.0016
    
```

Returns: 

```
    1.0E-323
```



Training Converged

Next, we run the same optimization using L-BFGS, which is nearly ideal for purely second-order or quadratic functions.

Code from [LearningTester.java:249](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L249) executed in 0.11 seconds: 
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
    th(0)=2.729096607999998;dx=-0.0010916386432
    New Minimum: 2.729096607999998 > 2.726745250531886
    WOLFE (weak): th(2.154434690031884)=2.726745250531886; dx=-0.0010911682703676222 delta=0.002351357468111992
    New Minimum: 2.726745250531886 > 2.7243949064513253
    WOLFE (weak): th(4.308869380063768)=2.7243949064513253; dx=-0.0010906978975352443 delta=0.0047017015486727765
    New Minimum: 2.7243949064513253 > 2.7150036640045125
    WOLFE (weak): th(12.926608140191302)=2.7150036640045125; dx=-0.0010888164062057327 delta=0.014092943995485552
    New Minimum: 2.7150036640045125 > 2.6729437237283
    WOLFE (weak): th(51.70643256076521)=2.6729437237283; dx=-0.0010803496952229309 delta=0.05615288427169807
    New Minimum: 2.6729437237283 > 2.454169298913998
    WOLFE (weak): th(258.53216280382605)=2.454169298913998; dx=-0.001035193903314654 delta=0.27492730908600027
    New Minimum: 2.454169298913998 > 1.29842446370586
    END: th(1551.1929768229563)=1.29842446370586
```
...[skipping 2523 bytes](etc/268.txt)...
```
    02618462465434E-10 delta=1.5103980694602287E-5
    Iteration 6 complete. Error: 1.525654615616326E-7 Total: 239662909190214.1000; Orientation: 0.0006; Line Search: 0.0056
    LBFGS Accumulation History: 1 points
    th(0)=1.525654615616326E-7;dx=-6.102618462465338E-11
    New Minimum: 1.525654615616326E-7 > 1.3451765601259862E-7
    WOLF (strong): th(9694.956105143481)=1.3451765601259862E-7; dx=5.7303051615425997E-11 delta=1.8047805549033983E-8
    New Minimum: 1.3451765601259862E-7 > 1.4196487427545115E-10
    END: th(4847.478052571741)=1.4196487427545115E-10; dx=-1.861566504613721E-12 delta=1.5242349668735714E-7
    Iteration 7 complete. Error: 1.4196487427545115E-10 Total: 239662916555481.1000; Orientation: 0.0006; Line Search: 0.0052
    LBFGS Accumulation History: 1 points
    th(0)=1.4196487427545115E-10;dx=-5.6785949710180336E-14
    MAX ALPHA: th(0)=1.4196487427545115E-10;th'(0)=-5.6785949710180336E-14;
    Iteration 8 failed, aborting. Error: 1.4196487427545115E-10 Total: 239662920489038.1000; Orientation: 0.0006; Line Search: 0.0026
    
```

Returns: 

```
    1.4196487427545115E-10
```



Training Converged

Code from [LearningTester.java:95](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L95) executed in 0.00 seconds: 
```java
    return TestUtil.compare(runs);
```

Returns: 

![Result](etc/test.172.png)



Code from [LearningTester.java:98](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L98) executed in 0.00 seconds: 
```java
    return TestUtil.compareTime(runs);
```

Returns: 

![Result](etc/test.173.png)



### Model Learning
In this test, attempt to train a network to emulate a randomized network given an example input/output. The target state is:

Code from [LearningTester.java:176](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L176) executed in 0.00 seconds: 
```java
    return network_target.state().stream().map(Arrays::toString).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [0.0, 1.0]
```



First, we use a conjugate gradient descent method, which converges the fastest for purely linear functions.

Code from [LearningTester.java:225](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L225) executed in 0.11 seconds: 
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
    F(0.0) = LineSearchPoint{point=PointSample{avg=2.342018923200004}, derivative=-11.21979867050784}
    New Minimum: 2.342018923200004 > 2.3420189220780516
    F(1.0E-10) = LineSearchPoint{point=PointSample{avg=2.3420189220780516}, derivative=-11.219798667762628}, delta = -1.1219523088357164E-9
    New Minimum: 2.3420189220780516 > 2.342018915346165
    F(7.000000000000001E-10) = LineSearchPoint{point=PointSample{avg=2.342018915346165}, derivative=-11.219798651291258}, delta = -7.853838912552646E-9
    New Minimum: 2.342018915346165 > 2.342018868222992
    F(4.900000000000001E-9) = LineSearchPoint{point=PointSample{avg=2.342018868222992}, derivative=-11.219798535992084}, delta = -5.497701183188042E-8
    New Minimum: 2.342018868222992 > 2.342018538360919
    F(3.430000000000001E-8) = LineSearchPoint{point=PointSample{avg=2.342018538360919}, derivative=-11.219797728897298}, delta = -3.84839085043609E-7
    New Minimum: 2.342018538360919 > 2.342016229327151
    F(2.4010000000000004E-7) = LineSearchPoin
```
...[skipping 23052 bytes](etc/269.txt)...
```
    LineSearchPoint{point=PointSample{avg=1.997543723439331E-32}, derivative=7.481070469852958E-32}, delta = -1.4646016218236594E-30
    Right bracket at 0.4031029149041485
    New Minimum: 1.997543723439331E-32 > 1.924081051640624E-32
    F(0.39892567704368925) = LineSearchPoint{point=PointSample{avg=1.924081051640624E-32}, derivative=5.215901144629712E-32}, delta = -1.4653362485416465E-30
    Right bracket at 0.39892567704368925
    Converged to right
    Iteration 19 complete. Error: 1.924081051640624E-32 Total: 239663153325216.8400; Orientation: 0.0001; Line Search: 0.0044
    Zero gradient: 2.8692332605132344E-16
    F(0.0) = LineSearchPoint{point=PointSample{avg=1.924081051640624E-32}, derivative=-8.232499503235405E-32}
    New Minimum: 1.924081051640624E-32 > 0.0
    F(0.39892567704368925) = LineSearchPoint{point=PointSample{avg=0.0}, derivative=0.0}, delta = -1.924081051640624E-32
    0.0 <= 1.924081051640624E-32
    Converged to right
    Iteration 20 complete. Error: 0.0 Total: 239663155212628.8400; Orientation: 0.0001; Line Search: 0.0010
    
```

Returns: 

```
    0.0
```



Training Converged

Next, we run the same optimization using L-BFGS, which is nearly ideal for purely second-order or quadratic functions.

Code from [LearningTester.java:249](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L249) executed in 0.04 seconds: 
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
    th(0)=2.342018923200004;dx=-11.21979867050784
    Armijo: th(2.154434690031884)=41.88062182791107; dx=47.924186218339344 delta=-39.53860290471107
    Armijo: th(1.077217345015942)=6.18358878164892; dx=18.352193773915708 delta=-3.841569858448916
    New Minimum: 2.342018923200004 > 0.08304629959256603
    END: th(0.3590724483386473)=0.08304629959256603; dx=-1.3624678557000012 delta=2.258972623607438
    Iteration 1 complete. Error: 0.08304629959256603 Total: 239663163639439.8400; Orientation: 0.0001; Line Search: 0.0022
    LBFGS Accumulation History: 1 points
    th(0)=0.08304629959256603;dx=-0.3336120040222748
    New Minimum: 0.08304629959256603 > 0.025778089016491625
    WOLF (strong): th(0.7735981389354633)=0.025778089016491625; dx=0.18555526062090075 delta=0.0572682105760744
    New Minimum: 0.025778089016491625 > 0.004208840589006729
    END: th(0.3867990694677316)=0.004208840589006729; dx=-0.07402837170068721 delta=0.0788374590035593
    Iteration 2 complete. 
```
...[skipping 9164 bytes](etc/270.txt)...
```
    196996444.8000; Orientation: 0.0000; Line Search: 0.0012
    LBFGS Accumulation History: 1 points
    th(0)=2.3865888445170225E-29;dx=-9.545141711173765E-29
    New Minimum: 2.3865888445170225E-29 > 1.2621774483536189E-29
    WOLF (strong): th(0.8743830237895417)=1.2621774483536189E-29; dx=6.940627101538148E-29 delta=1.1244113961634036E-29
    New Minimum: 1.2621774483536189E-29 > 4.437342591868191E-31
    END: th(0.43719151189477085)=4.437342591868191E-31; dx=-1.3013675815387334E-29 delta=2.3422154185983406E-29
    Iteration 20 complete. Error: 4.437342591868191E-31 Total: 239663198795228.8000; Orientation: 0.0001; Line Search: 0.0014
    LBFGS Accumulation History: 1 points
    th(0)=4.437342591868191E-31;dx=-1.774993873812591E-30
    Armijo: th(0.9419005594135813)=4.437342591868191E-31; dx=1.774993873812591E-30 delta=0.0
    New Minimum: 4.437342591868191E-31 > 0.0
    END: th(0.47095027970679065)=0.0; dx=0.0 delta=4.437342591868191E-31
    Iteration 21 complete. Error: 0.0 Total: 239663200615670.8000; Orientation: 0.0000; Line Search: 0.0012
    
```

Returns: 

```
    0.0
```



Training Converged

Code from [LearningTester.java:95](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L95) executed in 0.00 seconds: 
```java
    return TestUtil.compare(runs);
```

Returns: 

![Result](etc/test.174.png)



Code from [LearningTester.java:98](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L98) executed in 0.00 seconds: 
```java
    return TestUtil.compareTime(runs);
```

Returns: 

![Result](etc/test.175.png)



### Performance
Now we execute larger-scale runs to benchmark performance:

Code from [PerformanceTester.java:66](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/PerformanceTester.java#L66) executed in 0.25 seconds: 
```java
    test(component, inputPrototype);
```
Logging: 
```
    100 batches
    Input Dimensions:
    	[100, 100, 1]
    Performance:
    	Evaluation performance: 0.012935s +- 0.001431s [0.011360s - 0.015200s]
    	Learning performance: 0.020812s +- 0.000914s [0.019927s - 0.022304s]
    
```

### Function Plots
Code from [ActivationLayerTestBase.java:110](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/java/ActivationLayerTestBase.java#L110) executed in 0.00 seconds: 
```java
    return plot("Value Plot", plotData, x -> new double[]{x[0], x[1]});
```

Returns: 

![Result](etc/test.176.png)



Code from [ActivationLayerTestBase.java:114](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/java/ActivationLayerTestBase.java#L114) executed in 0.00 seconds: 
```java
    return plot("Derivative Plot", plotData, x -> new double[]{x[0], x[2]});
```

Returns: 

![Result](etc/test.177.png)



