# CrossDifferenceLayer
## CrossDifferenceLayerTest
### Batch Execution
Code from [BatchingTester.java:66](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/BatchingTester.java#L66) executed in 0.01 seconds: 
```java
    return test(reference, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (100#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (100#)}
```



Code from [SingleDerivativeTester.java:77](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/SingleDerivativeTester.java#L77) executed in 0.00 seconds: 
```java
    return test(component, inputPrototype);
```
Logging: 
```
    Inputs: [ -1.344, -1.308, 0.864, 1.608 ]
    Inputs Statistics: {meanExponent=0.09695169989934518, negative=2, min=1.608, max=1.608, mean=-0.04500000000000004, count=4.0, positive=2, stdDev=1.3077901207762659, zeros=0}
    Output: [ -0.03600000000000003, -2.208, -2.952, -2.172, -2.9160000000000004, -0.7440000000000001 ]
    Outputs Statistics: {meanExponent=0.007273033180679692, negative=6, min=-0.7440000000000001, max=-0.7440000000000001, mean=-1.838, count=6.0, positive=0, stdDev=1.0874722984977592, zeros=0}
    Feedback for input 0
    Inputs Values: [ -1.344, -1.308, 0.864, 1.608 ]
    Value Statistics: {meanExponent=0.09695169989934518, negative=2, min=1.608, max=1.608, mean=-0.04500000000000004, count=4.0, positive=2, stdDev=1.3077901207762659, zeros=0}
    Implemented Feedback: [ [ 1.0, 1.0, 1.0, 0.0, 0.0, 0.0 ], [ -1.0, 0.0, 0.0, 1.0, 1.0, 0.0 ], [ 0.0, -1.0, 0.0, -1.0, 0.0, 1.0 ], [ 0.0, 0.0, -1.0, 0.0, -1.0, -1.0 ] ]
    Implemented Statistics: {meanExponent=0.0, negative=6, min=-1.0, max=-1.0, mean=0.0, count=24.0, positi
```
...[skipping 362 bytes](etc/209.txt)...
```
    , -0.9999999999998899 ] ]
    Measured Statistics: {meanExponent=-4.7830642341759385E-14, negative=6, min=-0.9999999999998899, max=-0.9999999999998899, mean=3.7007434154171886E-13, count=24.0, positive=6, stdDev=0.7071067811864696, zeros=12}
    Feedback Error: [ [ -1.1013412404281553E-13, 2.1103119252074976E-12, -2.3305801732931286E-12, 0.0, 0.0, 0.0 ], [ 1.1013412404281553E-13, 0.0, 0.0, 2.1103119252074976E-12, 2.1103119252074976E-12, 0.0 ], [ 0.0, 2.3305801732931286E-12, 0.0, 2.3305801732931286E-12, 0.0, -1.1013412404281553E-13 ], [ 0.0, 0.0, -2.1103119252074976E-12, 0.0, 2.3305801732931286E-12, 1.1013412404281553E-13 ] ]
    Error Statistics: {meanExponent=-12.08875579914072, negative=4, min=1.1013412404281553E-13, max=1.1013412404281553E-13, mean=3.7007434154171886E-13, count=24.0, positive=8, stdDev=1.229865831537019E-12, zeros=12}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 7.5850e-13 +- 1.0364e-12 [0.0000e+00 - 2.3306e-12] (24#)
    relativeTol: 7.5850e-13 +- 4.9943e-13 [5.5067e-14 - 1.1653e-12] (12#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=7.5850e-13 +- 1.0364e-12 [0.0000e+00 - 2.3306e-12] (24#), relativeTol=7.5850e-13 +- 4.9943e-13 [5.5067e-14 - 1.1653e-12] (12#)}
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
      "class": "com.simiacryptus.mindseye.layers.java.CrossDifferenceLayer",
      "id": "467a7d3a-1a8d-481d-9f0e-6ef0d36515bc",
      "isFrozen": false,
      "name": "CrossDifferenceLayer/467a7d3a-1a8d-481d-9f0e-6ef0d36515bc"
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
    [[ -0.312, 0.332, -0.36, -1.816 ]]
    --------------------
    Output: 
    [ -0.644, 0.04799999999999999, 1.504, 0.692, 2.148, 1.456 ]
    --------------------
    Derivative: 
    [ 3.0, 1.0, -1.0, -3.0 ]
```



### Input Learning
In this test, we use a network to learn this target input, given it's pre-evaluated output:

Code from [LearningTester.java:127](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L127) executed in 0.00 seconds: 
```java
    return Arrays.stream(input_target).map(x -> x.prettyPrint()).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [ -1.892, 0.768, 1.648, 0.688 ]
```



First, we use a conjugate gradient descent method, which converges the fastest for purely linear functions.

Code from [LearningTester.java:225](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L225) executed in 0.01 seconds: 
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
    F(0.0) = LineSearchPoint{point=PointSample{avg=1.032533333333333}, derivative=-2.753422222222221}
    New Minimum: 1.032533333333333 > 1.0325333330579909
    F(1.0E-10) = LineSearchPoint{point=PointSample{avg=1.0325333330579909}, derivative=-2.7534222218550983}, delta = -2.753421934897915E-10
    New Minimum: 1.0325333330579909 > 1.0325333314059377
    F(7.000000000000001E-10) = LineSearchPoint{point=PointSample{avg=1.0325333314059377}, derivative=-2.753422219652361}, delta = -1.9273953544285405E-9
    New Minimum: 1.0325333314059377 > 1.0325333198415645
    F(4.900000000000001E-9) = LineSearchPoint{point=PointSample{avg=1.0325333198415645}, derivative=-2.7534222042331966}, delta = -1.3491768591222808E-8
    New Minimum: 1.0325333198415645 > 1.0325332388909532
    F(3.430000000000001E-8) = LineSearchPoint{point=PointSample{avg=1.0325332388909532}, derivative=-2.753422096299045}, delta = -9.444237991651505E-8
    New Minimum: 1.0325332388909532 > 1.0325326722367636
    F(2.4010000000000004E-7) = Lin
```
...[skipping 3610 bytes](etc/210.txt)...
```
    .0) = LineSearchPoint{point=PointSample{avg=4.108650548026103E-33}, derivative=-8.217301096052206E-33}
    F(0.5964174942007456) = LineSearchPoint{point=PointSample{avg=4.108650548026103E-33}, derivative=-8.217301096052206E-33}, delta = 0.0
    F(4.174922459405219) = LineSearchPoint{point=PointSample{avg=8.628166150854817E-32}, derivative=3.971695529758566E-32}, delta = 8.217301096052207E-32
    F(0.3211478814927092) = LineSearchPoint{point=PointSample{avg=4.108650548026103E-33}, derivative=-8.217301096052206E-33}, delta = 0.0
    F(2.248035170448964) = LineSearchPoint{point=PointSample{avg=4.108650548026103E-33}, derivative=4.108650548026103E-33}, delta = 0.0
    4.108650548026103E-33 <= 4.108650548026103E-33
    New Minimum: 4.108650548026103E-33 > 0.0
    F(1.498690113632643) = LineSearchPoint{point=PointSample{avg=0.0}, derivative=0.0}, delta = -4.108650548026103E-33
    Right bracket at 1.498690113632643
    Converged to right
    Iteration 3 complete. Error: 0.0 Total: 239636189696180.8000; Orientation: 0.0000; Line Search: 0.0007
    
```

Returns: 

```
    0.0
```



Training Converged

Next, we run the same optimization using L-BFGS, which is nearly ideal for purely second-order or quadratic functions.

Code from [LearningTester.java:249](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L249) executed in 0.01 seconds: 
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
    th(0)=1.032533333333333;dx=-2.753422222222221
    Armijo: th(2.154434690031884)=3.620634208731733; dx=5.156002246924754 delta=-2.5881008753984
    New Minimum: 1.032533333333333 > 0.1965414642178748
    WOLF (strong): th(1.077217345015942)=0.1965414642178748; dx=1.2012900123512666 delta=0.8359918691154583
    END: th(0.3590724483386473)=0.28052664211381195; dx=-1.4351848106977254 delta=0.7520066912195211
    Iteration 1 complete. Error: 0.1965414642178748 Total: 239636195056908.8000; Orientation: 0.0001; Line Search: 0.0006
    LBFGS Accumulation History: 1 points
    th(0)=0.28052664211381195;dx=-0.7480710456368318
    New Minimum: 0.28052664211381195 > 2.7771995995197186E-4
    WOLF (strong): th(0.7735981389354633)=2.7771995995197186E-4; dx=0.023537445958046826 delta=0.28024892215386
    END: th(0.3867990694677316)=0.06578781940130717; dx=-0.36226679983939236 delta=0.21473882271250477
    Iteration 2 complete. Error: 2.7771995995197186E-4 Total: 239636195699819.
```
...[skipping 9591 bytes](etc/211.txt)...
```
     Orientation: 0.0000; Line Search: 0.0003
    LBFGS Accumulation History: 1 points
    th(0)=1.6385347689334675E-26;dx=-4.369422763568806E-26
    New Minimum: 1.6385347689334675E-26 > 1.279576350829345E-26
    WOLF (strong): th(1.4128508391203713)=1.279576350829345E-26; dx=3.8612613029584635E-26 delta=3.589584181041225E-27
    New Minimum: 1.279576350829345E-26 > 5.434922944928929E-29
    END: th(0.7064254195601857)=5.434922944928929E-29; dx=-2.5161923776184915E-27 delta=1.6330998459885386E-26
    Iteration 21 complete. Error: 5.434922944928929E-29 Total: 239636206206688.7800; Orientation: 0.0000; Line Search: 0.0004
    LBFGS Accumulation History: 1 points
    th(0)=5.434922944928929E-29;dx=-1.4489840932705387E-28
    Armijo: th(1.521947429820792)=5.602145022233592E-29; dx=1.4710338512116121E-28 delta=-1.6722207730466277E-30
    New Minimum: 5.434922944928929E-29 > 0.0
    END: th(0.760973714910396)=0.0; dx=0.0 delta=5.434922944928929E-29
    Iteration 22 complete. Error: 0.0 Total: 239636206640710.7800; Orientation: 0.0000; Line Search: 0.0003
    
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

![Result](etc/test.124.png)



Code from [LearningTester.java:98](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L98) executed in 0.00 seconds: 
```java
    return TestUtil.compareTime(runs);
```

Returns: 

![Result](etc/test.125.png)



### Performance
Now we execute larger-scale runs to benchmark performance:

Code from [PerformanceTester.java:66](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/PerformanceTester.java#L66) executed in 0.00 seconds: 
```java
    test(component, inputPrototype);
```
Logging: 
```
    100 batches
    Input Dimensions:
    	[4]
    Performance:
    	Evaluation performance: 0.000232s +- 0.000058s [0.000174s - 0.000341s]
    	Learning performance: 0.000040s +- 0.000002s [0.000037s - 0.000043s]
    
```

