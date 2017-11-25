### Json Serialization
Code from [LayerTestBase.java:57](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L57) executed in 0.00 seconds: 
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
      "class": "com.simiacryptus.mindseye.layers.java.AvgImageBandLayer",
      "id": "9d13704a-9a5a-4ecb-a687-5c7c0002dcf1",
      "isFrozen": false,
      "name": "AvgImageBandLayer/9d13704a-9a5a-4ecb-a687-5c7c0002dcf1"
    }
```



### Differential Validation
Code from [LayerTestBase.java:74](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L74) executed in 0.00 seconds: 
```java
    getDerivativeTester().test(getLayer(), outputPrototype, inputPrototype);
```
Logging: 
```
    Component: AvgImageBandLayer/9d13704a-9a5a-4ecb-a687-5c7c0002dcf3
    Inputs: [[ [ [ 0.2575762943735136,0.18497138271834201,0.17521485991357455 ],[ 0.8297749607385472,0.2698773614849065,0.6211516492744192 ] ],[ [ 0.16738016947622625,0.09228516676935072,0.2813823696558503 ],[ 0.74702786898521,0.36551363529680925,0.9418886214166055 ] ] ]]
    output=[ [ [ 0.22816188656735212,0.5004398233933742,0.5049093750651124 ] ] ]
    measured/actual: [ [ 0.0,0.24999999848063226,0.0 ],[ 0.0,446955.1671738126,-446954.9171738141 ],[ 0.0,446955.1671738126,-446954.9171738141 ],[ 2.7674748849776022E7,-2.7227793682602208E7,-446954.9171738141 ],[ 0.2500000012561898,446955.1671738126,-446955.1671738126 ],[ 0.2500000012561898,0.0,0.0 ],[ 0.2500000012561898,0.0,0.0 ],[ 2.7674748849776022E7,-2.7227793432602208E7,-446955.1671738126 ],... ]
    implemented/expected: [ [ 0.25,0.0,0.0 ],[ 0.25,0.0,0.0 ],[ 0.25,0.0,0.0 ],[ 0.25,0.0,0.0 ],[ 0.0,0.25,0.0 ],[ 0.0,0.25,0.0 ],[ 0.0,0.25,0.0 ],[ 0.0,0.25,0.0 ],... ]
    error: [ [ -0.25,0.24999999848063226,0.0 ],[ -0.25,446955.1671738126,-446954.9171738141 ],[ -0.25,446955.1671738126,-446954.9171738141 ],[ 2.7674748599776022E7,-2.7227793682602208E7,-446954.9171738141 ],[ 0.2500000012561898,446954.9171738126,-446955.1671738126 ],[ 0.2500000012561898,-0.25,0.0 ],[ 0.2500000012561898,-0.25,0.0 ],[ 2.7674748849776022E7,-2.7227793682602208E7,-446955.1671738126 ],... ]
    
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



