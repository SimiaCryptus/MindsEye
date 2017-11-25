### Json Serialization
Code from [LayerTestBase.java:57](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L57) executed in 0.00 seconds: 
```java
    NNLayer layer = getLayer();
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
      "class": "com.simiacryptus.mindseye.layers.cudnn.f64.ActivationLayer",
      "id": "9d13704a-9a5a-4ecb-a687-5c7c00016395",
      "isFrozen": false,
      "name": "ActivationLayer/9d13704a-9a5a-4ecb-a687-5c7c00016395",
      "mode": 0
    }
```



### Differential Validation
Code from [LayerTestBase.java:74](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L74) executed in 0.00 seconds: 
```java
    getDerivativeTester().test(getLayer(), outputPrototype, inputPrototype);
```
Logging: 
```
    Component: ActivationLayer/9d13704a-9a5a-4ecb-a687-5c7c00016397
    Inputs: [[ [ [ 0.8565618788771715,0.3147381403455458 ],[ 0.710375664209023,0.6438531925398906 ],[ 0.6165235509397496,0.802915425070658 ] ],[ [ 0.1854414466213834,0.7477544087972795 ],[ 0.9471133956662442,0.9317680521676784 ],[ 0.8358761084187464,0.326766388775764 ] ],[ [ 0.8013652566210963,0.8171361946905552 ],[ 0.8168990885485823,0.49870011338013953 ],[ 0.9092838895463307,0.6754690571485723 ] ] ]]
    output=[ [ [ 0.7019418321614374,0.5780413641205897 ],[ 0.6704841625522763,0.6556239598937811 ],[ 0.649427472185735,0.6905977731639599 ] ],[ [ 0.5462279615500981,0.6786891993437985 ],[ 0.720534288210851,0.7174338472376235 ],[ 0.6975959665669805,0.5809723820410759 ] ],[ [ 0.6902664469863441,0.6936280958048269 ],[ 0.6935777065015918,0.6221538044290235 ],[ 0.7128536020548057,0.6627266888367438 ] ] ]
    measured/actual: [ [ 0.20921948618024544,0.0,0.0,0.0,0.0,0.0,0.0,0.0,... ],[ 0.0,0.24786297458945228,0.0,0.0,0.0,0.0,0.0,0.0,... ],[ 0.0,0.0,0.21379868986315387,0.0,0.0,0.0,0.0,0.0,... ],[ 0.0,0.0,0.0,0.22093514795429314,0.0,0.0,0.0,0.0,... ],[ 0.0,0.0,0.0,0.0,0.20136463607656196,0.0,0.0,0.0,... ],[ 0.0,0.0,0.0,0.0,0.0,0.21252767323787225,0.0,0.0,... ],[ 0.0,0.0,0.0,0.0,0.0,0.0,0.22767142615620628,0.0,... ],[ 0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.21095584168406845,... ],... ]
    implemented/expected: [ [ 0.20921949642328186,0.0,0.0,0.0,0.0,0.0,0.0,0.0,... ],[ 0.0,0.24786297557092266,0.0,0.0,0.0,0.0,0.0,0.0,... ],[ 0.0,0.0,0.21379867915119272,0.0,0.0,0.0,0.0,0.0,... ],[ 0.0,0.0,0.0,0.22093515031884903,0.0,0.0,0.0,0.0,... ],[ 0.0,0.0,0.0,0.0,0.20136462772333333,0.0,0.0,0.0,... ],[ 0.0,0.0,0.0,0.0,0.0,0.21252767154558358,0.0,0.0,... ],[ 0.0,0.0,0.0,0.0,0.0,0.0,0.2276714305561814,0.0,... ],[ 0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.2109558339964607,... ],... ]
    error: [ [ -1.0243036419055329E-8,0.0,0.0,0.0,0.0,0.0,0.0,0.0,... ],[ 0.0,-9.814703771926503E-10,0.0,0.0,0.0,0.0,0.0,0.0,... ],[ 0.0,0.0,1.0711961151432803E-8,0.0,0.0,0.0,0.0,0.0,... ],[ 0.0,0.0,0.0,-2.3645558844265224E-9,0.0,0.0,0.0,0.0,... ],[ 0.0,0.0,0.0,0.0,8.35322863657062E-9,0.0,0.0,0.0,... ],[ 0.0,0.0,0.0,0.0,0.0,1.6922886703074624E-9,0.0,0.0,... ],[ 0.0,0.0,0.0,0.0,0.0,0.0,-4.399975106483822E-9,0.0,... ],[ 0.0,0.0,0.0,0.0,0.0,0.0,0.0,7.687607744832903E-9,... ],... ]
    
```

Returns: 

```
    java.lang.AssertionError
    	at com.simiacryptus.mindseye.layers.DerivativeTester.testFeedback(DerivativeTester.java:219)
    	at com.simiacryptus.mindseye.layers.DerivativeTester.lambda$test$0(DerivativeTester.java:69)
    	at java.util.stream.IntPipeline$4$1.accept(IntPipeline.java:250)
    	at java.util.stream.Streams$RangeIntSpliterator.forEachRemaining(Streams.java:110)
    	at java.util.Spliterator$OfInt.forEachRemaining(Spliterator.java:693)
    	at java.util.stream.AbstractPipeline.copyInto(AbstractPipeline.java:481)
    	at java.util.stream.AbstractPipeline.wrapAndCopyInto(AbstractPipeline.java:471)
    	at java.util.stream.ReduceOps$ReduceOp.evaluateSequential(ReduceOps.java:708)
    	at java.util.stream.AbstractPipeline.evaluate(AbstractPipeline.java:234)
    	at java.util.stream.ReferencePipeline.reduce(ReferencePipeline.java:479)
    	at com.simiacryptus.mindseye.layers.DerivativeTester.test(DerivativeTester.java:70)
    	at com.simiacryptus.mindseye.layers.LayerTestBase.lambda$test$6(LayerTestBase.java:75)
    	at com.simiacryptus.util.io.NotebookOutput.lambda$code$1(NotebookOutput.java:142)
    	at com.simiacryptus.util.io.MarkdownNotebookOutput.lambda$null$1(MarkdownNotebookOutput.java:136)
    	at com.simiacryptus.util.lang.TimedResult.time(TimedResult.java:59)
    	at com.simiacryptus.util.io.MarkdownNotebookOutput.lambda$code$2(MarkdownNotebookOutput.java:136)
    	at com.simiacryptus.util.test.SysOutInterceptor.withOutput(SysOutInterceptor.java:77)
    	at com.simiacryptus.util.io.MarkdownNotebookOutput.code(MarkdownNotebookOutput.java:134)
    	at com.simiacryptus.util.io.NotebookOutput.code(NotebookOutput.java:141)
    	at com.simiacryptus.mindseye.layers.LayerTestBase.test(LayerTestBase.java:74)
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



