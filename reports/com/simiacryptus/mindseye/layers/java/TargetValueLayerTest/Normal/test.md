# TargetValueLayer
## Normal
### Network Diagram
This is a network with the following layout:

Code from [StandardLayerTests.java:72](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/StandardLayerTests.java#L72) executed in 0.14 seconds: 
```java
    return Graphviz.fromGraph(TestUtil.toGraph((DAGNetwork) layer))
      .height(400).width(600).render(Format.PNG).toImage();
```

Returns: 

![Result](etc/test.253.png)



### Batch Execution
Code from [BatchingTester.java:66](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/BatchingTester.java#L66) executed in 0.00 seconds: 
```java
    return test(reference, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (40#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (40#)}
```



Code from [SingleDerivativeTester.java:77](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/SingleDerivativeTester.java#L77) executed in 0.00 seconds: 
```java
    return test(component, inputPrototype);
```
Logging: 
```
    Inputs: [ 0.516, 1.768, 0.12 ]
    Inputs Statistics: {meanExponent=-0.32022893054936985, negative=0, min=0.12, max=0.12, mean=0.8013333333333333, count=3.0, positive=3, stdDev=0.7023946342493103, zeros=0}
    Output: [ 1.0182933333333333 ]
    Outputs Statistics: {meanExponent=0.007872900493477837, negative=0, min=1.0182933333333333, max=1.0182933333333333, mean=1.0182933333333333, count=1.0, positive=1, stdDev=0.0, zeros=0}
    Feedback for input 0
    Inputs Values: [ 0.516, 1.768, 0.12 ]
    Value Statistics: {meanExponent=-0.32022893054936985, negative=0, min=0.12, max=0.12, mean=0.8013333333333333, count=3.0, positive=3, stdDev=0.7023946342493103, zeros=0}
    Implemented Feedback: [ [ 0.344 ], [ 1.1119999999999999 ], [ -0.053333333333333344 ] ]
    Implemented Statistics: {meanExponent=-0.5634460140820563, negative=1, min=-0.053333333333333344, max=-0.053333333333333344, mean=0.46755555555555556, count=3.0, positive=2, stdDev=0.48370095853114714, zeros=0}
    Measured Feedback: [ [ 0.34403333333488106 ], [ 1.1120333333325405 ], 
```
...[skipping 663 bytes](etc/353.txt)...
```
    ], [ 0.053333333333333344 ] ]
    Implemented Statistics: {meanExponent=-0.5634460140820563, negative=2, min=0.053333333333333344, max=0.053333333333333344, mean=-0.46755555555555556, count=3.0, positive=1, stdDev=0.48370095853114714, zeros=0}
    Measured Gradient: [ [ -0.34396666666713926 ], [ -1.1119666666670192 ], [ 0.053366666665777274 ] ]
    Measured Statistics: {meanExponent=-0.5633739321472679, negative=2, min=0.053366666665777274, max=0.053366666665777274, mean=-0.4675222222227937, count=3.0, positive=1, stdDev=0.48370095853094425, zeros=0}
    Gradient Error: [ [ 3.333333286070772E-5 ], [ 3.333333298072283E-5 ], [ 3.333333244393E-5 ] ]
    Error Statistics: {meanExponent=-4.477121262166247, negative=0, min=3.333333244393E-5, max=3.333333244393E-5, mean=3.333333276178685E-5, count=3.0, positive=3, stdDev=4.547473508864641E-13, zeros=0}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 3.3333e-05 +- 9.0949e-13 [3.3333e-05 - 3.3333e-05] (6#)
    relativeTol: 1.2531e-04 +- 1.3306e-04 [1.4988e-05 - 3.1260e-04] (6#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=3.3333e-05 +- 9.0949e-13 [3.3333e-05 - 3.3333e-05] (6#), relativeTol=1.2531e-04 +- 1.3306e-04 [1.4988e-05 - 3.1260e-04] (6#)}
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
      "class": "com.simiacryptus.mindseye.layers.java.TargetValueLayer",
      "id": "0b7d2a78-388e-4527-b01f-34dd924b5739",
      "isFrozen": false,
      "name": "TargetValueLayer/0b7d2a78-388e-4527-b01f-34dd924b5739",
      "inputs": [
        "4eedd389-33f7-41f0-8449-c58d2f912c34"
      ],
      "nodes": {
        "263fd15f-2002-4082-8d10-bbd953488615": "3240283e-a03c-4c8c-8b90-ec13914d41e6",
        "a340ef6e-04e1-4cf3-a97f-2746bd403a18": "330954b6-be7d-4a78-bcbf-ead3fa27aacd"
      },
      "layers": {
        "3240283e-a03c-4c8c-8b90-ec13914d41e6": {
          "class": "com.simiacryptus.mindseye.layers.java.ConstNNLayer",
          "id": "3240283e-a03c-4c8c-8b90-ec13914d41e6",
          "isFrozen": true,
          "name": "ConstNNLayer/3240283e-a03c-4c8c-8b90-ec13914d41e6",
          "value": [
            0.0,
            0.1,
            0.2
          ]
        },
        "330954b6-be7d-4a78-bcbf-ead3fa27aacd": {
          "class": "com.simiacryptus.mindseye.layers.java.MeanSqLossLayer",
          "id": "330954b6-be7d-4a78-bcbf-ead3fa27aacd",
          "isFrozen": false,
          "name": "MeanSqLossLayer/330954b6-be7d-4a78-bcbf-ead3fa27aacd"
        }
      },
      "links": {
        "263fd15f-2002-4082-8d10-bbd953488615": [],
        "a340ef6e-04e1-4cf3-a97f-2746bd403a18": [
          "4eedd389-33f7-41f0-8449-c58d2f912c34",
          "263fd15f-2002-4082-8d10-bbd953488615"
        ]
      },
      "labels": {},
      "head": "a340ef6e-04e1-4cf3-a97f-2746bd403a18",
      "target": "263fd15f-2002-4082-8d10-bbd953488615"
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
    [[ -1.82, -0.384, 0.384 ]]
    --------------------
    Output: 
    [ 1.1935040000000001 ]
    --------------------
    Derivative: 
    [ -1.2133333333333334, -0.32266666666666666, 0.12266666666666666 ]
```



### Input Learning
In this test, we use a network to learn this target input, given it's pre-evaluated output:

Code from [LearningTester.java:127](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L127) executed in 0.00 seconds: 
```java
    return Arrays.stream(input_target).map(x -> x.prettyPrint()).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [ 1.356, -0.472, 0.468 ]
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
Logging: 
```
    Constructing line search parameters: GD
    F(0.0) = LineSearchPoint{point=PointSample{avg=0.032785137777777716}, derivative=-0.1097234922892956}
    New Minimum: 0.032785137777777716 > 0.0327851377668054
    F(1.0E-10) = LineSearchPoint{point=PointSample{avg=0.0327851377668054}, derivative=-0.1097234922735838}, delta = -1.097231333568871E-11
    New Minimum: 0.0327851377668054 > 0.03278513770097128
    F(7.000000000000001E-10) = LineSearchPoint{point=PointSample{avg=0.03278513770097128}, derivative=-0.1097234921793126}, delta = -7.680643621110761E-11
    New Minimum: 0.03278513770097128 > 0.032785137240132614
    F(4.900000000000001E-9) = LineSearchPoint{point=PointSample{avg=0.032785137240132614}, derivative=-0.10972349151941459}, delta = -5.376451020500106E-10
    New Minimum: 0.032785137240132614 > 0.032785134014262016
    F(3.430000000000001E-8) = LineSearchPoint{point=PointSample{avg=0.032785134014262016}, derivative=-0.10972348690012824}, delta = -3.763515700472286E-9
    New Minimum: 0.032785134014262016 > 0.032785111433171754
    F(
```
...[skipping 4751 bytes](etc/354.txt)...
```
    mple{avg=2.6081713678869703E-29}, derivative=6.216480994360519E-26}, delta = -7.96721978097249E-24
    2.6081713678869703E-29 <= 7.96724586268617E-24
    Converged to right
    Iteration 5 complete. Error: 2.6081713678869703E-29 Total: 239731891313166.1000; Orientation: 0.0000; Line Search: 0.0001
    Zero gradient: 1.0605454400971564E-14
    F(0.0) = LineSearchPoint{point=PointSample{avg=2.6081713678869703E-29}, derivative=-1.124756630510871E-28}
    New Minimum: 2.6081713678869703E-29 > 4.930380657631324E-32
    F(0.46457978874479444) = LineSearchPoint{point=PointSample{avg=4.930380657631324E-32}, derivative=4.8902462196124675E-30}, delta = -2.603240987229339E-29
    4.930380657631324E-32 <= 2.6081713678869703E-29
    New Minimum: 4.930380657631324E-32 > 0.0
    F(0.4452222975470947) = LineSearchPoint{point=PointSample{avg=0.0}, derivative=0.0}, delta = -2.6081713678869703E-29
    Right bracket at 0.4452222975470947
    Converged to right
    Iteration 6 complete. Error: 0.0 Total: 239731891579621.1000; Orientation: 0.0000; Line Search: 0.0002
    
```

Returns: 

```
    0.0
```



Training Converged

Next, we run the same optimization using L-BFGS, which is nearly ideal for purely second-order or quadratic functions.

Code from [LearningTester.java:249](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L249) executed in 0.01 seconds: 
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
    th(0)=0.032785137777777716;dx=-0.1097234922892956
    Armijo: th(2.154434690031884)=0.4114882826037252; dx=0.5909083509818571 delta=-0.3787031448259475
    Armijo: th(1.077217345015942)=0.03525475046861505; dx=0.14337142856858381 delta=-0.0024696126908373348
    New Minimum: 0.032785137777777716 > 0.00456366957112873
    END: th(0.3590724483386473)=0.00456366957112873; dx=-0.04448597465349082 delta=0.028221468206648986
    Iteration 1 complete. Error: 0.00456366957112873 Total: 239731894979413.1000; Orientation: 0.0001; Line Search: 0.0004
    LBFGS Accumulation History: 1 points
    th(0)=0.00456366957112873;dx=-0.018036264610072895
    New Minimum: 0.00456366957112873 > 0.0015455275855682943
    WOLF (strong): th(0.7735981389354633)=0.0015455275855682943; dx=0.011227471488066484 delta=0.0030181419855604356
    New Minimum: 0.0015455275855682943 > 2.256111870546486E-4
    END: th(0.3867990694677316)=2.256111870546486E-4; dx=-0.004149952533318196 delta=0.004338058
```
...[skipping 8647 bytes](etc/355.txt)...
```
    901071111.1000; Orientation: 0.0000; Line Search: 0.0002
    LBFGS Accumulation History: 1 points
    th(0)=5.0487097934144756E-29;dx=-2.1772226647317858E-28
    New Minimum: 5.0487097934144756E-29 > 2.8398992587956425E-29
    WOLF (strong): th(0.8117052959044226)=2.8398992587956425E-29; dx=1.6329169985488519E-28 delta=2.208810534618833E-29
    New Minimum: 2.8398992587956425E-29 > 7.888609052210118E-31
    END: th(0.4058526479522113)=7.888609052210118E-31; dx=-2.7215283309147423E-29 delta=4.9698237028923744E-29
    Iteration 19 complete. Error: 7.888609052210118E-31 Total: 239731901281425.1000; Orientation: 0.0000; Line Search: 0.0002
    LBFGS Accumulation History: 1 points
    th(0)=7.888609052210118E-31;dx=-3.401910413643442E-30
    Armijo: th(0.8743830237895417)=7.888609052210118E-31; dx=3.401910413643445E-30 delta=0.0
    New Minimum: 7.888609052210118E-31 > 0.0
    END: th(0.43719151189477085)=0.0; dx=0.0 delta=7.888609052210118E-31
    Iteration 20 complete. Error: 0.0 Total: 239731901556429.1000; Orientation: 0.0000; Line Search: 0.0002
    
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

![Result](etc/test.254.png)



Code from [LearningTester.java:98](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L98) executed in 0.00 seconds: 
```java
    return TestUtil.compareTime(runs);
```

Returns: 

![Result](etc/test.255.png)



### Model Learning
In this test, attempt to train a network to emulate a randomized network given an example input/output. The target state is:

Code from [LearningTester.java:176](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L176) executed in 0.00 seconds: 
```java
    return network_target.state().stream().map(Arrays::toString).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [0.0, 0.2, 0.1]
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
    java.lang.AssertionError: Nothing to optimize
    	at com.simiacryptus.mindseye.opt.IterativeTrainer.run(IterativeTrainer.java:104)
    	at com.simiacryptus.mindseye.test.unit.LearningTester.lambda$trainCjGD$32(LearningTester.java:233)
    	at com.simiacryptus.util.io.MarkdownNotebookOutput.lambda$null$1(MarkdownNotebookOutput.java:138)
    	at com.simiacryptus.util.lang.TimedResult.time(TimedResult.java:59)
    	at com.simiacryptus.util.io.MarkdownNotebookOutput.lambda$code$2(MarkdownNotebookOutput.java:138)
    	at com.simiacryptus.util.test.SysOutInterceptor.withOutput(SysOutInterceptor.java:72)
    	at com.simiacryptus.util.io.MarkdownNotebookOutput.code(MarkdownNotebookOutput.java:136)
    	at com.simiacryptus.util.io.NotebookOutput.code(NotebookOutput.java:133)
    	at com.simiacryptus.mindseye.test.unit.LearningTester.trainCjGD(LearningTester.java:225)
    	at com.simiacryptus.mindseye.test.unit.LearningTester.testModelLearning(LearningTester.java:186)
    	at com.simiacryptus.mindseye.test.unit.LearningTester.test(LearningTester.java
```
...[skipping 2135 bytes](etc/356.txt)...
```
    unner.java:268)
    	at org.junit.runners.ParentRunner.run(ParentRunner.java:363)
    	at org.junit.runners.Suite.runChild(Suite.java:128)
    	at org.junit.runners.Suite.runChild(Suite.java:27)
    	at org.junit.runners.ParentRunner$3.run(ParentRunner.java:290)
    	at org.junit.runners.ParentRunner$1.schedule(ParentRunner.java:71)
    	at org.junit.runners.ParentRunner.runChildren(ParentRunner.java:288)
    	at org.junit.runners.ParentRunner.access$000(ParentRunner.java:58)
    	at org.junit.runners.ParentRunner$2.evaluate(ParentRunner.java:268)
    	at org.junit.runners.ParentRunner.run(ParentRunner.java:363)
    	at org.junit.runner.JUnitCore.run(JUnitCore.java:137)
    	at com.intellij.junit4.JUnit4IdeaTestRunner.startRunnerWithArgs(JUnit4IdeaTestRunner.java:68)
    	at com.intellij.rt.execution.junit.IdeaTestRunner$Repeater.startRunnerWithArgs(IdeaTestRunner.java:47)
    	at com.intellij.rt.execution.junit.JUnitStarter.prepareStreamsAndStart(JUnitStarter.java:242)
    	at com.intellij.rt.execution.junit.JUnitStarter.main(JUnitStarter.java:70)
    
```



