### Json Serialization
Code from [LayerTestBase.java:74](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L74) executed in 0.00 seconds: 
```java
  
```

Returns: 

```
    {
      "class": "com.simiacryptus.mindseye.layers.java.AvgImageBandLayer",
      "id": "bdd6bbba-380b-47fe-a761-c2410002dcaa",
      "isFrozen": false,
      "name": "AvgImageBandLayer/bdd6bbba-380b-47fe-a761-c2410002dcaa"
    }
```



### Differential Validation
Code from [LayerTestBase.java:98](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L98) executed in 0.00 seconds: 
```java
  
```
Logging: 
```
    Component: AvgImageBandLayer/bdd6bbba-380b-47fe-a761-c2410002dcaa
    Inputs: [[ [ [ 0.22316564719896226,0.8543596010621872,-1.8244291261538663 ],[ 1.3606016496923141,-1.8265263899384458,-0.37738212009576655 ] ],[ [ 1.9896425461238985,1.0250421518150765,-1.8263434104467993 ],[ 0.7024109186753664,0.8038549350904218,-1.8216450180699457 ] ] ]]
    output=[ [ [ 0.21418257450730993,-1.4624499186915945,1.0689551904226353 ] ] ]
    measured/actual: [ [ 0.0,0.0,0.2500000000349445 ],[ 0.0,2531405.3591142297,-2531405.10911423 ],[ 854772.8659153255,1676632.4931989044,-2531405.10911423 ],[ 0.0,2531405.3591142297,-2531405.10911423 ],[ 0.2500000000349445,0.0,0.0 ],[ 854772.6159153254,1676632.7431989042,-2531405.10911423 ],[ 0.24999999997943334,0.0,0.0 ],[ -1676632.4931989044,2531405.10911423,-854772.3659153254 ],... ]
    implemented/expected: [ [ 0.25,0.0,0.0 ],[ 0.25,0.0,0.0 ],[ 0.25,0.0,0.0 ],[ 0.25,0.0,0.0 ],[ 0.0,0.25,0.0 ],[ 0.0,0.25,0.0 ],[ 0.0,0.25,0.0 ],[ 0.0,0.25,0.0 ],... ]
    error: [ [ -0.25,0.0,0.2500000000349445 ],[ -0.25,2531405.3591142297,-2531405.10911423 ],[ 854772.6159153255,1676632.4931989044,-2531405.10911423 ],[ -0.25,2531405.3591142297,-2531405.10911423 ],[ 0.2500000000349445,-0.25,0.0 ],[ 854772.6159153254,1676632.4931989042,-2531405.10911423 ],[ 0.24999999997943334,-0.25,0.0 ],[ -1676632.4931989044,2531404.85911423,-854772.3659153254 ],... ]
    
```

Returns: 

```
    java.lang.AssertionError
    	at com.simiacryptus.mindseye.layers.DerivativeTester.testFeedback(DerivativeTester.java:220)
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
    	at com.simiacryptus.mindseye.layers.LayerTestBase.lambda$test$7(LayerTestBase.java:99)
    	at com.simiacryptus.util.io.NotebookOutput.lambda$code$1(NotebookOutput.java:142)
    	at com.simiacryptus.util.io.MarkdownNotebookOutput.lambda$null$1(MarkdownNotebookOutput.java:136)
    	at com.simiacryptus.util.lang.TimedResult.time(TimedResult.java:59)
    	at com.simiacryptus.util.io.MarkdownNotebookOutput.lambda$code$2(MarkdownNotebookOutput.java:136)
    	at com.simiacryptus.util.test.SysOutInterceptor.withOutput(SysOutInterceptor.java:77)
    	at com.simiacryptus.util.io.MarkdownNotebookOutput.code(MarkdownNotebookOutput.java:134)
    	at com.simiacryptus.util.io.NotebookOutput.code(NotebookOutput.java:141)
    	at com.simiacryptus.mindseye.layers.LayerTestBase.test(LayerTestBase.java:98)
    	at com.simiacryptus.mindseye.layers.LayerTestBase.test(LayerTestBase.java:66)
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



