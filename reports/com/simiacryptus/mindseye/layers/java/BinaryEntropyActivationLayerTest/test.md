# BinaryEntropyActivationLayer
## BinaryEntropyActivationLayerTest
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
    	[ [ 0.25864900115887335 ], [ 0.2607404720329617 ], [ 0.24526004699643508 ] ],
    	[ [ 0.2717944176187357 ], [ 0.18283998368082793 ], [ 0.10568305544864326 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.6768561454317402, negative=0, min=0.10568305544864326, max=0.10568305544864326, mean=0.22082782948941282, count=6.0, positive=6, stdDev=0.05904713066298703, zeros=0}
    Output: [
    	[ [ -0.5716390662422919 ], [ -0.5738300035768087 ], [ -0.5570676063385148 ] ],
    	[ [ -0.5850354514815974 ], [ -0.47567267797556256 ], [ -0.3373934312070483 ] ]
    ]
    Outputs Statistics: {meanExponent=-0.29425998801860437, negative=6, min=-0.3373934312070483, max=-0.3373934312070483, mean=-0.516773039470304, count=6.0, positive=0, stdDev=0.08795403215616808, zeros=0}
    Feedback for input 0
    Inputs Values: [
    	[ [ 0.25864900115887335 ], [ 0.2607404720329617 ], [ 0.24526004699643508 ] ],
    	[ [ 0.2717944176187357 ], [ 0.18283998368082793 ], [ 0.10568305544864326 ] ]
    ]
    Value Statistics: {meanExponent=-0.6768561454317402, negative=0, min=0.1056
```
...[skipping 881 bytes](etc/205.txt)...
```
     [ 0.0, 0.0, 0.0, 0.0, -1.1237841044031693, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, -2.1350867901243076 ] ]
    Measured Statistics: {meanExponent=0.09816377326103902, negative=6, min=-2.1350867901243076, max=-2.1350867901243076, mean=-0.2176569539032632, count=36.0, positive=0, stdDev=0.5142743235744811, zeros=30}
    Feedback Error: [ [ 2.607347313983155E-4, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 2.526050861186846E-4, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 2.593753462676318E-4, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 3.346033430244866E-4, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 2.700884545923099E-4, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 5.288742614060205E-4 ] ]
    Error Statistics: {meanExponent=-3.5146736004100454, negative=0, min=5.288742614060205E-4, max=5.288742614060205E-4, mean=5.29522561890958E-5, count=36.0, positive=6, stdDev=1.2502674859452397E-4, zeros=30}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 5.2952e-05 +- 1.2503e-04 [0.0000e+00 - 5.2887e-04] (36#)
    relativeTol: 1.2203e-04 +- 5.1516e-06 [1.1175e-04 - 1.2817e-04] (6#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=5.2952e-05 +- 1.2503e-04 [0.0000e+00 - 5.2887e-04] (36#), relativeTol=1.2203e-04 +- 5.1516e-06 [1.1175e-04 - 1.2817e-04] (6#)}
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
      "class": "com.simiacryptus.mindseye.layers.java.BinaryEntropyActivationLayer",
      "id": "f0e45f4c-b4c9-41d9-90b4-4813e43321a6",
      "isFrozen": true,
      "name": "BinaryEntropyActivationLayer/f0e45f4c-b4c9-41d9-90b4-4813e43321a6"
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
    	[ [ 0.2099032376612878 ], [ 0.2430472272455965 ], [ 0.20301085587103648 ] ],
    	[ [ 0.263257785550171 ], [ 0.2624580592747366 ], [ 0.1536539188184498 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ -0.5138284395136365 ], [ -0.5545670237406403 ], [ -0.5045481328945611 ] ],
    	[ [ -0.5764369515504076 ], [ -0.5756122999492101 ], [ -0.4289951676473458 ] ]
    ]
    --------------------
    Derivative: 
    [
    	[ [ -1.3255087704461077 ], [ -1.136045088850879 ], [ -1.3675816029710572 ] ],
    	[ [ -1.0291043288209851 ], [ -1.0332316603020826 ], [ -1.7062255622770253 ] ]
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
    	[ [ 0.17517342429022675 ], [ 0.28494805557835134 ], [ 0.23906672111255664 ], [ 0.28543833470317626 ], [ 0.25296398376916907 ], [ 0.15045187419265768 ], [ 0.25948228452729827 ], [ 0.23823583362673806 ], ... ],
    	[ [ 0.28266868199651707 ], [ 0.2109197992306595 ], [ 0.24297254908298507 ], [ 0.19747625051574547 ], [ 0.10263813209252473 ], [ 0.213938112545551 ], [ 0.16105289122109795 ], [ 0.23745765065365004 ], ... ],
    	[ [ 0.28614525675886815 ], [ 0.12682609843770298 ], [ 0.13388710694758887 ], [ 0.2012823813057554 ], [ 0.14686525794359562 ], [ 0.12439806716549265 ], [ 0.25175595618829216 ], [ 0.20120185236255209 ], ... ],
    	[ [ 0.1793275095308376 ], [ 0.2518820674272969 ], [ 0.19536125770721605 ], [ 0.22321680934756039 ], [ 0.14176280517888812 ], [ 0.2131895823740649 ], [ 0.2749700885400183 ], [ 0.2055466722844974 ], ... ],
    	[ [ 0.16246410025454294 ], [ 0.16606687052361485 ], [ 0.1013986608333616 ], [ 0.25322197441911776 ], [ 0.1877921583466439 ], [ 0.1402959840380258 ], [ 0.2857963657113087 ], [ 0.24966786418837916 ], ... ],
    	[ [ 0.2490145013203014 ], [ 0.12830516899347255 ], [ 0.21411016449915649 ], [ 0.2378991940418833 ], [ 0.2672072348160057 ], [ 0.2543674363072586 ], [ 0.2661135797563058 ], [ 0.11831878410071046 ], ... ],
    	[ [ 0.21557732046997083 ], [ 0.2772308268299364 ], [ 0.207900553522301 ], [ 0.26264085556779654 ], [ 0.11558588189489337 ], [ 0.130517634805654 ], [ 0.13039932157462647 ], [ 0.25911175623176075 ], ... ],
    	[ [ 0.2850390168142406 ], [ 0.25815362755190746 ], [ 0.25669737902541523 ], [ 0.22840361400407988 ], [ 0.11194509214813385 ], [ 0.21119149246294808 ], [ 0.20530591401896844 ], [ 0.1392743698874258 ], ... ],
    	...
    ]
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
Logging: 
```
    Constructing line search parameters: GD
    F(0.0) = LineSearchPoint{point=PointSample{avg=0.013471569201249142}, derivative=-1.299382245797244E-5}
    New Minimum: 0.013471569201249142 > 0.013471569201247856
    F(1.0E-10) = LineSearchPoint{point=PointSample{avg=0.013471569201247856}, derivative=-1.2993822457971334E-5}, delta = -1.285430095698814E-15
    New Minimum: 0.013471569201247856 > 0.013471569201240024
    F(7.000000000000001E-10) = LineSearchPoint{point=PointSample{avg=0.013471569201240024}, derivative=-1.2993822457964685E-5}, delta = -9.117706589734098E-15
    New Minimum: 0.013471569201240024 > 0.013471569201185463
    F(4.900000000000001E-9) = LineSearchPoint{point=PointSample{avg=0.013471569201185463}, derivative=-1.299382245791815E-5}, delta = -6.367822935615663E-14
    New Minimum: 0.013471569201185463 > 0.013471569200803437
    F(3.430000000000001E-8) = LineSearchPoint{point=PointSample{avg=0.013471569200803437}, derivative=-1.2993822457592412E-5}, delta = -4.45704237406197E-13
    New Minimum: 0.013471569200803437 > 0.01
```
...[skipping 4296 bytes](etc/206.txt)...
```
    2.13983355013059E-4
    1.154474347779134E-4 <= 3.294307897909724E-4
    New Minimum: 1.154474347779134E-4 > 1.1036864404454523E-4
    F(1668.2217365772306) = LineSearchPoint{point=PointSample{avg=1.1036864404454523E-4}, derivative=-5.513067526427444E-9}, delta = -2.1906214574642715E-4
    Left bracket at 1668.2217365772306
    New Minimum: 1.1036864404454523E-4 > 1.1027618940687607E-4
    F(1701.0295433837684) = LineSearchPoint{point=PointSample{avg=1.1027618940687607E-4}, derivative=-1.1807562823078218E-10}, delta = -2.1915460038409633E-4
    Left bracket at 1701.0295433837684
    Converged to left
    Iteration 3 complete. Error: 1.1027618940687607E-4 Total: 239635552225170.4700; Orientation: 0.0009; Line Search: 0.0103
    F(0.0) = LineSearchPoint{point=PointSample{avg=1.1027618940687607E-4}, derivative=-6.078122316814394E-8}
    New Minimum: 1.1027618940687607E-4 > 5.065869660240513E-5
    F(1701.0295433837684) = LineSearchPoint{point=PointSample{avg=5.065869660240513E-5}, derivative=-9.258581704945138E-9}, delta = -5.9617492804470945E-5
    
```

Returns: 

```
    java.lang.RuntimeException: java.lang.RuntimeException: java.lang.RuntimeException: com.simiacryptus.mindseye.lang.GpuError: java.util.concurrent.ExecutionException: com.simiacryptus.mindseye.lang.GpuError: Failed executing 1 items
    	at com.simiacryptus.util.lang.TimedResult.time(TimedResult.java:61)
    	at com.simiacryptus.util.io.MarkdownNotebookOutput.lambda$code$2(MarkdownNotebookOutput.java:138)
    	at com.simiacryptus.util.test.SysOutInterceptor.withOutput(SysOutInterceptor.java:72)
    	at com.simiacryptus.util.io.MarkdownNotebookOutput.code(MarkdownNotebookOutput.java:136)
    	at com.simiacryptus.util.io.NotebookOutput.code(NotebookOutput.java:133)
    	at com.simiacryptus.mindseye.test.unit.LearningTester.trainCjGD(LearningTester.java:225)
    	at com.simiacryptus.mindseye.test.unit.LearningTester.testInputLearning(LearningTester.java:141)
    	at com.simiacryptus.mindseye.test.unit.LearningTester.test(LearningTester.java:328)
    	at com.simiacryptus.mindseye.test.unit.StandardLayerTests.lambda$test$4(StandardLayerTests
```
...[skipping 11087 bytes](etc/207.txt)...
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



