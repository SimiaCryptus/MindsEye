# BinaryEntropyActivationLayer
## BinaryEntropyActivationLayerTest
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
      "class": "com.simiacryptus.mindseye.layers.java.BinaryEntropyActivationLayer",
      "id": "70635025-b26c-4400-bf3b-b5bb1ebc1177",
      "isFrozen": true,
      "name": "BinaryEntropyActivationLayer/70635025-b26c-4400-bf3b-b5bb1ebc1177"
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
    	[ [ 0.221289882811041 ], [ 0.2696049204709506 ], [ 0.1843678968284379 ] ],
    	[ [ 0.28393728600448565 ], [ 0.27997748635271347 ], [ 0.2666203426945267 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ -0.5285356816157358 ], [ -0.5828654890271993 ], [ -0.47795250988822513 ] ],
    	[ [ -0.5966335941257166 ], [ -0.5929320529148128 ], [ -0.5798683045587465 ] ]
    ]
    --------------------
    Derivative: 
    [
    	[ [ -1.2581653259718657 ], [ -0.9966279627016195 ], [ -1.487030198666802 ] ],
    	[ [ -0.9250143626559848 ], [ -0.944573286422445 ], [ -1.0118378087220221 ] ]
    ]
```



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
    	[ [ 0.21720468213507718 ], [ 0.17419249319272234 ], [ 0.15972678355411063 ] ],
    	[ [ 0.2980438291121994 ], [ 0.12934669772457444 ], [ 0.1278139789734684 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.7543516334686219, negative=0, min=0.1278139789734684, max=0.1278139789734684, mean=0.18438807744869204, count=6.0, positive=6, stdDev=0.0590718078336777, zeros=0}
    Output: [
    	[ [ -0.523347184396289 ], [ -0.4624720608686776 ], [ -0.43921654638624846 ] ],
    	[ [ -0.609197720319052 ], [ -0.3851429160661527 ], [ -0.3822099429262478 ] ]
    ]
    Outputs Statistics: {meanExponent=-0.3367939582908001, negative=6, min=-0.3822099429262478, max=-0.3822099429262478, mean=-0.46693106182711125, count=6.0, positive=0, stdDev=0.07966338225590446, zeros=0}
    Feedback for input 0
    Inputs Values: [
    	[ [ 0.21720468213507718 ], [ 0.17419249319272234 ], [ 0.15972678355411063 ] ],
    	[ [ 0.2980438291121994 ], [ 0.12934669772457444 ], [ 0.1278139789734684 ] ]
    ]
    Value Statistics: {meanExponent=-0.7543516334686219, negative=0, min=0.12781397
```
...[skipping 874 bytes](etc/250.txt)...
```
    [ 0.0, 0.0, 0.0, 0.0, -1.659889867747233, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, -1.9199783894752187 ] ]
    Measured Statistics: {meanExponent=0.16933459320417987, negative=6, min=-1.9199783894752187, max=-1.9199783894752187, mean=-0.25500426534913156, count=36.0, positive=0, stdDev=0.589997689048096, zeros=30}
    Feedback Error: [ [ 2.9403867416633034E-4, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 2.3897470352729488E-4, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 3.475331060156517E-4, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 4.438887541058367E-4, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 3.724760474801414E-4, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 4.484209473769596E-4 ] ]
    Error Statistics: {meanExponent=-3.4570315077509197, negative=0, min=4.484209473769596E-4, max=4.484209473769596E-4, mean=5.959256201867263E-5, count=36.0, positive=6, stdDev=1.367639942561914E-4, zeros=30}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 5.9593e-05 +- 1.3676e-04 [0.0000e+00 - 4.4842e-04] (36#)
    relativeTol: 1.1854e-04 +- 9.5703e-06 [1.1167e-04 - 1.3950e-04] (6#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=5.9593e-05 +- 1.3676e-04 [0.0000e+00 - 4.4842e-04] (36#), relativeTol=1.1854e-04 +- 9.5703e-06 [1.1167e-04 - 1.3950e-04] (6#)}
```



### Performance
Now we execute larger-scale runs to benchmark performance:

Code from [PerformanceTester.java:66](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/PerformanceTester.java#L66) executed in 0.24 seconds: 
```java
    test(component, inputPrototype);
```
Logging: 
```
    100 batches
    Input Dimensions:
    	[100, 100, 1]
    Performance:
    	Evaluation performance: 0.011298s +- 0.006261s [0.007423s - 0.023790s]
    	Learning performance: 0.015268s +- 0.003330s [0.012438s - 0.021746s]
    
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
    	[ [ 0.1577027198850143 ], [ 0.23990005531473577 ], [ 0.1216396907899196 ], [ 0.1799416890066623 ], [ 0.1725300061057638 ], [ 0.2789367486331922 ], [ 0.13049330220500902 ], [ 0.12753906085628786 ], ... ],
    	[ [ 0.252025460764174 ], [ 0.14865203448677172 ], [ 0.2279976333039946 ], [ 0.2174041793565553 ], [ 0.19249679811398127 ], [ 0.11424654684474772 ], [ 0.2772873515310942 ], [ 0.21007645296601474 ], ... ],
    	[ [ 0.23019002758494503 ], [ 0.237280074715603 ], [ 0.21008827808379227 ], [ 0.29168208308954857 ], [ 0.23874893276953416 ], [ 0.21323053072100204 ], [ 0.20590710995469919 ], [ 0.23449327262746802 ], ... ],
    	[ [ 0.1973875573234155 ], [ 0.28165800041173306 ], [ 0.19636323648005866 ], [ 0.24424023271350823 ], [ 0.12711310837583734 ], [ 0.12727730239261315 ], [ 0.11449253515460607 ], [ 0.2532218607893735 ], ... ],
    	[ [ 0.2805626339456765 ], [ 0.1993592582374114 ], [ 0.29587253589957696 ], [ 0.19781686553151065 ], [ 0.1923487476269403 ], [ 0.22927541261573503 ], [ 0.10628030230929629 ], [ 0.23520541048583007 ], ... ],
    	[ [ 0.10128806023096414 ], [ 0.2041485294188471 ], [ 0.27773243702186823 ], [ 0.135213740837823 ], [ 0.28537224727953453 ], [ 0.14927885581400385 ], [ 0.27987616413659855 ], [ 0.1944020621662954 ], ... ],
    	[ [ 0.2155148632545997 ], [ 0.21567505464945816 ], [ 0.2972964526427032 ], [ 0.25547644438828754 ], [ 0.16991078134698961 ], [ 0.2611021474597673 ], [ 0.19914159811433346 ], [ 0.21812323974531 ], ... ],
    	[ [ 0.2047775876081429 ], [ 0.2090498248385434 ], [ 0.14681825507476856 ], [ 0.19924199266225973 ], [ 0.1245625505540273 ], [ 0.2620192186184521 ], [ 0.10627897617503078 ], [ 0.2917081495400171 ], ... ],
    	...
    ]
```



First, we use a conjugate gradient descent method, which converges the fastest for purely linear functions.

Code from [LearningTester.java:300](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L300) executed in 0.00 seconds: 
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
    F(0.0) = LineSearchPoint{point=PointSample{avg=0.013391464527328856}, derivative=-1.2845469595211285E-5}
    New Minimum: 0.013391464527328856 > 0.013391464527327566
    F(1.0E-10) = LineSearchPoint{point=PointSample{avg=0.013391464527327566}, derivative=-1.2845469595210194E-5}, delta = -1.2906342661267445E-15
    New Minimum: 0.013391464527327566 > 0.01339146452731988
    F(7.000000000000001E-10) = LineSearchPoint{point=PointSample{avg=0.01339146452731988}, derivative=-1.2845469595203651E-5}, delta = -8.977193988179977E-15
    New Minimum: 0.01339146452731988 > 0.01339146452726592
    F(4.900000000000001E-9) = LineSearchPoint{point=PointSample{avg=0.01339146452726592}, derivative=-1.2845469595157856E-5}, delta = -6.293576770843856E-14
    New Minimum: 0.01339146452726592 > 0.013391464526888289
    F(3.430000000000001E-8) = LineSearchPoint{point=PointSample{avg=0.013391464526888289}, derivative=-1.2845469594837276E-5}, delta = -4.405677211938297E-13
    New Minimum: 0.013391464526888289 > 0.0133
```
...[skipping 4295 bytes](etc/251.txt)...
```
    130283577120407E-4
    1.1477435857791132E-4 <= 3.27802716289952E-4
    New Minimum: 1.1477435857791132E-4 > 1.1003927885190995E-4
    F(1666.3064641391923) = LineSearchPoint{point=PointSample{avg=1.1003927885190995E-4}, derivative=-5.616242174136333E-9}, delta = -2.1776343743804206E-4
    Left bracket at 1666.3064641391923
    New Minimum: 1.1003927885190995E-4 > 1.0994330360618657E-4
    F(1699.7257211560582) = LineSearchPoint{point=PointSample{avg=1.0994330360618657E-4}, derivative=-1.2202557285904844E-10}, delta = -2.1785941268376546E-4
    Left bracket at 1699.7257211560582
    Converged to left
    Iteration 3 complete. Error: 1.0994330360618657E-4 Total: 249769328803092.7000; Orientation: 0.0003; Line Search: 0.0088
    F(0.0) = LineSearchPoint{point=PointSample{avg=1.0994330360618657E-4}, derivative=-5.889876213615334E-8}
    New Minimum: 1.0994330360618657E-4 > 5.183552567655458E-5
    F(1699.7257211560582) = LineSearchPoint{point=PointSample{avg=5.183552567655458E-5}, derivative=-9.415202819335162E-9}, delta = -5.8107777929631985E-5
    
```

Returns: 

```
    java.lang.RuntimeException: java.lang.RuntimeException: java.lang.RuntimeException: com.simiacryptus.mindseye.lang.GpuError: java.util.concurrent.ExecutionException: com.simiacryptus.mindseye.lang.GpuError: Failed executing 1 items
    	at com.simiacryptus.util.lang.TimedResult.time(TimedResult.java:61)
    	at com.simiacryptus.util.io.MarkdownNotebookOutput.lambda$code$2(MarkdownNotebookOutput.java:138)
    	at com.simiacryptus.util.test.SysOutInterceptor.withOutput(SysOutInterceptor.java:72)
    	at com.simiacryptus.util.io.MarkdownNotebookOutput.code(MarkdownNotebookOutput.java:136)
    	at com.simiacryptus.util.io.NotebookOutput.code(NotebookOutput.java:133)
    	at com.simiacryptus.mindseye.test.unit.LearningTester.trainCjGD(LearningTester.java:300)
    	at com.simiacryptus.mindseye.test.unit.LearningTester.testInputLearning(LearningTester.java:141)
    	at com.simiacryptus.mindseye.test.unit.LearningTester.test(LearningTester.java:406)
    	at com.simiacryptus.mindseye.test.unit.LearningTester.test(LearningTester.java:58)
    	at co
```
...[skipping 11172 bytes](etc/252.txt)...
```
    tractPipeline.java:471)
    	at java.util.stream.Nodes$SizedCollectorTask.compute(Nodes.java:1878)
    	at java.util.concurrent.CountedCompleter.exec(CountedCompleter.java:731)
    	at java.util.concurrent.ForkJoinTask.doExec(ForkJoinTask.java:289)
    	at java.util.concurrent.ForkJoinTask.doInvoke(ForkJoinTask.java:401)
    	at java.util.concurrent.ForkJoinTask.invoke(ForkJoinTask.java:734)
    	at java.util.stream.Nodes.collect(Nodes.java:325)
    	at java.util.stream.ReferencePipeline.evaluateToNode(ReferencePipeline.java:109)
    	at java.util.stream.AbstractPipeline.evaluate(AbstractPipeline.java:540)
    	at java.util.stream.AbstractPipeline.evaluateToArrayNode(AbstractPipeline.java:260)
    	at java.util.stream.ReferencePipeline.toArray(ReferencePipeline.java:438)
    	at com.simiacryptus.mindseye.layers.java.SimpleActivationLayer.eval(SimpleActivationLayer.java:83)
    	at com.simiacryptus.mindseye.network.InnerNode.eval(InnerNode.java:83)
    	at com.simiacryptus.mindseye.network.LazyResult.lambda$get$0(LazyResult.java:75)
    	... 35 more
    
```



