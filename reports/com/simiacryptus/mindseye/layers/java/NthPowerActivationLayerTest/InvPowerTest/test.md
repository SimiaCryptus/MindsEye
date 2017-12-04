# NthPowerActivationLayer
## InvPowerTest
### Json Serialization
Code from [LayerTestBase.java:83](../../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L83) executed in 0.00 seconds: 
```java
    JsonObject json = layer.getJson();
    NNLayer echo = NNLayer.fromJson(json);
    assert (echo != null) : "Failed to deserialize";
    assert (layer != echo) : "Serialization did not copy";
    Assert.assertEquals("Serialization not equal", layer, echo);
    return new GsonBuilder().setPrettyPrinting().create().toJson(json);
```

Returns: 

```
    {
      "class": "com.simiacryptus.mindseye.layers.java.NthPowerActivationLayer",
      "id": "a864e734-2f23-44db-97c1-504000002c44",
      "isFrozen": false,
      "name": "NthPowerActivationLayer/a864e734-2f23-44db-97c1-504000002c44",
      "power": -1.0
    }
```



### Example Input/Output Pair
Code from [LayerTestBase.java:120](../../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L120) executed in 0.00 seconds: 
```java
    SimpleEval eval = SimpleEval.run(layer, inputPrototype);
    return String.format("--------------------\nInput: \n[%s]\n--------------------\nOutput: \n%s",
      Arrays.stream(inputPrototype).map(t->t.prettyPrint()).reduce((a,b)->a+",\n"+b).get(),
      eval.getOutput().prettyPrint());
```

Returns: 

```
    --------------------
    Input: 
    [[
    	[ [ 1.42 ], [ -1.952 ], [ -0.028 ] ],
    	[ [ 0.572 ], [ 1.36 ], [ -1.676 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 0.7042253521126761 ], [ -0.5122950819672131 ], [ -35.714285714285715 ] ],
    	[ [ 1.7482517482517483 ], [ 0.7352941176470588 ], [ -0.5966587112171838 ] ]
    ]
```



### Batch Execution
Code from [LayerTestBase.java:138](../../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L138) executed in 0.00 seconds: 
```java
    BatchingTester batchingTester = getBatchingTester();
    return batchingTester==null?null:batchingTester.test(layer, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (120#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (120#)}
```



### Differential Validation
Code from [LayerTestBase.java:144](../../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.00 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ 1.42 ], [ -1.952 ], [ -0.028 ] ],
    	[ [ 0.572 ], [ 1.36 ], [ -1.676 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.16581080991442532, negative=3, min=-1.676, max=-1.676, mean=-0.05066666666666664, count=6.0, positive=3, stdDev=1.34150694701477, zeros=0}
    Output: [
    	[ [ 0.7042253521126761 ], [ -0.5122950819672131 ], [ -35.714285714285715 ] ],
    	[ [ 1.7482517482517483 ], [ 0.7352941176470588 ], [ -0.5966587112171838 ] ]
    ]
    Outputs Statistics: {meanExponent=0.16581080991442532, negative=3, min=-0.5966587112171838, max=-0.5966587112171838, mean=-5.605911381576438, count=6.0, positive=3, stdDev=13.488662970577998, zeros=0}
    Feedback for input 0
    Inputs Values: [
    	[ [ 1.42 ], [ -1.952 ], [ -0.028 ] ],
    	[ [ 0.572 ], [ 1.36 ], [ -1.676 ] ]
    ]
    Value Statistics: {meanExponent=-0.16581080991442532, negative=3, min=-1.676, max=-1.676, mean=-0.05066666666666664, count=6.0, positive=3, stdDev=1.34150694701477, zeros=0}
    Implemented Feedback: [ [ -0.49593334655822263, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, -3.0563841752652947, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, -0.2624462510077936, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, -0.5406574394463667, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, -1275.5102040816325, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, -0.35600161767135075 ] ]
    Implemented Statistics: {meanExponent=0.33162161982885063, negative=6, min=-0.35600161767135075, max=-0.35600161767135075, mean=-35.561711858655045, count=36.0, positive=0, stdDev=209.59017440555826, zeros=30}
    Measured: [ [ -0.4958984241343334, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, -3.0558499357646873, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, -0.26245969668936553, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, -0.5406176881461722, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, -1280.081925243195, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, -0.35602286008518114 ] ]
    Measured Statistics: {meanExponent=0.33186554589645517, negative=6, min=-0.35602286008518114, max=-0.35602286008518114, mean=-35.688688162444855, count=36.0, positive=0, stdDev=210.3414709673846, zeros=30}
    Feedback Error: [ [ 3.492242388924982E-5, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 5.342395006073808E-4, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, -1.3445681571910839E-5, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 3.975130019451267E-5, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, -4.5717211615624365, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, -2.124241383039127E-5 ] ]
    Error Statistics: {meanExponent=-3.502323644236906, negative=3, min=-2.124241383039127E-5, max=-2.124241383039127E-5, mean=-0.12697630378980965, count=36.0, positive=3, stdDev=0.7512990107113925, zeros=30}
    
```

Returns: 

```
    java.lang.AssertionError: ToleranceStatistics{absoluteTol=1.2701e-01 +- 7.5129e-01 [0.0000e+00 - 4.5717e+00] (36#), relativeTol=3.3396e-04 +- 6.5100e-04 [2.5615e-05 - 1.7889e-03] (6#)}
    	at com.simiacryptus.mindseye.layers.DerivativeTester.lambda$test$6(DerivativeTester.java:90)
    	at java.util.stream.IntPipeline$4$1.accept(IntPipeline.java:250)
    	at java.util.stream.Streams$RangeIntSpliterator.forEachRemaining(Streams.java:110)
    	at java.util.Spliterator$OfInt.forEachRemaining(Spliterator.java:693)
    	at java.util.stream.AbstractPipeline.copyInto(AbstractPipeline.java:481)
    	at java.util.stream.AbstractPipeline.wrapAndCopyInto(AbstractPipeline.java:471)
    	at java.util.stream.ReduceOps$ReduceOp.evaluateSequential(ReduceOps.java:708)
    	at java.util.stream.AbstractPipeline.evaluate(AbstractPipeline.java:234)
    	at java.util.stream.ReferencePipeline.reduce(ReferencePipeline.java:479)
    	at com.simiacryptus.mindseye.layers.DerivativeTester.test(DerivativeTester.java:121)
    	at com.simiacryptus.mindseye.layers.LayerTestBase.lambda$test$16(LayerTestBase.java:145)
    	at com.simiacryptus.util.io.MarkdownNotebookOutput.lambda$null$1(MarkdownNotebookOutput.java:136)
    	at com.simiacryptus.util.lang.TimedResult.time(TimedResult.java:59)
    	at com.simiacryptus.util.io.MarkdownNotebookOutput.lambda$code$2(MarkdownNotebookOutput.java:136)
    	at com.simiacryptus.util.test.SysOutInterceptor.withOutput(SysOutInterceptor.java:83)
    	at com.simiacryptus.util.io.MarkdownNotebookOutput.code(MarkdownNotebookOutput.java:134)
    	at com.simiacryptus.util.io.NotebookOutput.code(NotebookOutput.java:133)
    	at com.simiacryptus.mindseye.layers.LayerTestBase.test(LayerTestBase.java:144)
    	at com.simiacryptus.mindseye.layers.java.ActivationLayerTestBase.test(ActivationLayerTestBase.java:65)
    	at com.simiacryptus.mindseye.layers.LayerTestBase.test(LayerTestBase.java:68)
    	at sun.reflect.GeneratedMethodAccessor1.invoke(Unknown Source)
    	at sun.reflect.DelegatingMethodAccessorImpl.invoke(DelegatingMethodAccessorImpl.java:43)
    	at java.lang.reflect.Method.invoke(Method.java:498)
    	at org.junit.runners.model.FrameworkMethod$1.runReflectiveCall(FrameworkMethod.java:50)
    	at org.junit.internal.runners.model.ReflectiveCallable.run(ReflectiveCallable.java:12)
    	at org.junit.runners.model.FrameworkMethod.invokeExplosively(FrameworkMethod.java:47)
    	at org.junit.internal.runners.statements.InvokeMethod.evaluate(InvokeMethod.java:17)
    	at org.junit.runners.ParentRunner.runLeaf(ParentRunner.java:325)
    	at org.junit.runners.BlockJUnit4ClassRunner.runChild(BlockJUnit4ClassRunner.java:78)
    	at org.junit.runners.BlockJUnit4ClassRunner.runChild(BlockJUnit4ClassRunner.java:57)
    	at org.junit.runners.ParentRunner$3.run(ParentRunner.java:290)
    	at org.junit.runners.ParentRunner$1.schedule(ParentRunner.java:71)
    	at org.junit.runners.ParentRunner.runChildren(ParentRunner.java:288)
    	at org.junit.runners.ParentRunner.access$000(ParentRunner.java:58)
    	at org.junit.runners.ParentRunner$2.evaluate(ParentRunner.java:268)
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



