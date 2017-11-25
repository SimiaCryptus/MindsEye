### Json Serialization
Code from [LayerTestBase.java:57](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L57) executed in 0.01 seconds: 
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
      "class": "com.simiacryptus.mindseye.layers.cudnn.f64.ImgBandBiasLayer",
      "id": "9d13704a-9a5a-4ecb-a687-5c7c0002dce4",
      "isFrozen": false,
      "name": "ImgBandBiasLayer/9d13704a-9a5a-4ecb-a687-5c7c0002dce4",
      "bias": [
        0.0,
        0.0
      ]
    }
```



### Differential Validation
Code from [LayerTestBase.java:74](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L74) executed in 0.00 seconds: 
```java
    getDerivativeTester().test(getLayer(), outputPrototype, inputPrototype);
```
Logging: 
```
    Component: ImgBandBiasLayer/9d13704a-9a5a-4ecb-a687-5c7c0002dce6
    Inputs: [[ [ [ 0.6301724213382108,0.2244901924342254 ],[ 0.7054273097109587,0.4285411777196908 ],[ 0.0577548982845415,0.5027673018446146 ] ],[ [ 0.3046723930587746,0.11723339371277763 ],[ 0.9963422987038463,0.6043017711542129 ],[ 0.8872217471260075,0.25291019776641643 ] ],[ [ 0.740355832325488,0.7122801464497925 ],[ 0.8375426418715255,0.7371056481150072 ],[ 0.8953592440037063,0.693583020154881 ] ] ]]
    output=[ [ [ 0.6301724213382108,0.2244901924342254 ],[ 0.7054273097109587,0.4285411777196908 ],[ 0.0577548982845415,0.5027673018446146 ] ],[ [ 0.3046723930587746,0.11723339371277763 ],[ 0.9963422987038463,0.6043017711542129 ],[ 0.8872217471260075,0.25291019776641643 ] ],[ [ 0.740355832325488,0.7122801464497925 ],[ 0.8375426418715255,0.7371056481150072 ],[ 0.8953592440037063,0.693583020154881 ] ] ]
    measured/actual: [ [ 1.0000000050247593,0.0,0.0,0.0,0.0,0.0,0.0,0.0,... ],[ 0.0,0.9999999994736442,0.0,0.0,0.0,0.0,0.0,0.0,... ],[ 0.0,0.0,1.0000000050247593,0.0,0.0,0.0,0.0,0.0,... ],[ 0.0,0.0,0.0,1.0000000050247593,0.0,0.0,0.0,0.0,... ],[ 0.0,0.0,0.0,0.0,1.0000000050247593,0.0,0.0,0.0,... ],[ 0.0,0.0,0.0,0.0,0.0,1.0000000050247593,0.0,0.0,... ],[ 0.0,0.0,0.0,0.0,0.0,0.0,1.0000000001675335,0.0,... ],[ 0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0000000050247593,... ],... ]
    implemented/expected: [ [ 1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,... ],[ 0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,... ],[ 0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,... ],[ 0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,... ],[ 0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,... ],[ 0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,... ],[ 0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,... ],[ 0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,... ],... ]
    error: [ [ 5.024759275329416E-9,0.0,0.0,0.0,0.0,0.0,0.0,0.0,... ],[ 0.0,-5.26355847796367E-10,0.0,0.0,0.0,0.0,0.0,0.0,... ],[ 0.0,0.0,5.024759275329416E-9,0.0,0.0,0.0,0.0,0.0,... ],[ 0.0,0.0,0.0,5.024759275329416E-9,0.0,0.0,0.0,0.0,... ],[ 0.0,0.0,0.0,0.0,5.024759275329416E-9,0.0,0.0,0.0,... ],[ 0.0,0.0,0.0,0.0,0.0,5.024759275329416E-9,0.0,0.0,... ],[ 0.0,0.0,0.0,0.0,0.0,0.0,1.6753354259435582E-10,0.0,... ],[ 0.0,0.0,0.0,0.0,0.0,0.0,0.0,5.024759275329416E-9,... ],... ]
    
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



