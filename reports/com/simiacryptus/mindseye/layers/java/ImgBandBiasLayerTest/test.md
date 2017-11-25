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
      "class": "com.simiacryptus.mindseye.layers.java.ImgBandBiasLayer",
      "id": "9d13704a-9a5a-4ecb-a687-5c7c0002dd1d",
      "isFrozen": false,
      "name": "ImgBandBiasLayer/9d13704a-9a5a-4ecb-a687-5c7c0002dd1d",
      "bias": [
        0.0,
        0.0,
        0.0
      ]
    }
```



### Differential Validation
Code from [LayerTestBase.java:74](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L74) executed in 0.00 seconds: 
```java
    getDerivativeTester().test(getLayer(), outputPrototype, inputPrototype);
```
Logging: 
```
    Component: ImgBandBiasLayer/9d13704a-9a5a-4ecb-a687-5c7c0002dd1f
    Inputs: [[ [ [ 0.07265346564431341,0.7686966809255052,0.7433401227243309 ],[ 0.643578215612138,0.5266087673993083,0.73521464575074 ] ],[ [ 0.7855794602963456,0.9992856649112627,0.37033042847025566 ],[ 0.13174930601946,0.1999350495416543,0.12680201063266805 ] ] ]]
    output=[ [ [ 0.07265346564431341,0.7686966809255052,0.7433401227243309 ],[ 0.643578215612138,0.5266087673993083,0.73521464575074 ] ],[ [ 0.7855794602963456,0.9992856649112627,0.37033042847025566 ],[ 0.13174930601946,0.1999350495416543,0.12680201063266805 ] ] ]
    measured/actual: [ [ 0.9999999994736442,0.0,0.0,0.0,0.0,0.0,0.0,0.0,... ],[ 0.0,1.0000000050247593,0.0,0.0,0.0,0.0,0.0,0.0,... ],[ 0.0,0.0,1.0000000050247593,0.0,0.0,0.0,0.0,0.0,... ],[ 0.0,0.0,0.0,0.9999999994736442,0.0,0.0,0.0,0.0,... ],[ 0.0,0.0,0.0,0.0,1.0000000050247593,0.0,0.0,0.0,... ],[ 0.0,0.0,0.0,0.0,0.0,1.0000000050247593,0.0,0.0,... ],[ 0.0,0.0,0.0,0.0,0.0,0.0,1.0000000050247593,0.0,... ],[ 0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.9999999994736442,... ],... ]
    implemented/expected: [ [ 1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,... ],[ 0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,... ],[ 0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,... ],[ 0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,... ],[ 0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,... ],[ 0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,... ],[ 0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,... ],[ 0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,... ],... ]
    error: [ [ -5.26355847796367E-10,0.0,0.0,0.0,0.0,0.0,0.0,0.0,... ],[ 0.0,5.024759275329416E-9,0.0,0.0,0.0,0.0,0.0,0.0,... ],[ 0.0,0.0,5.024759275329416E-9,0.0,0.0,0.0,0.0,0.0,... ],[ 0.0,0.0,0.0,-5.26355847796367E-10,0.0,0.0,0.0,0.0,... ],[ 0.0,0.0,0.0,0.0,5.024759275329416E-9,0.0,0.0,0.0,... ],[ 0.0,0.0,0.0,0.0,0.0,5.024759275329416E-9,0.0,0.0,... ],[ 0.0,0.0,0.0,0.0,0.0,0.0,5.024759275329416E-9,0.0,... ],[ 0.0,0.0,0.0,0.0,0.0,0.0,0.0,-5.26355847796367E-10,... ],... ]
    
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



