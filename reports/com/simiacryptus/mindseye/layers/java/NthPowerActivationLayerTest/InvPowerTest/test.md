# NthPowerActivationLayer
## InvPowerTest
### Json Serialization
Code from [LayerTestBase.java:83](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L83) executed in 0.00 seconds: 
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
      "id": "370a9587-74a1-4959-b406-fa4500002c44",
      "isFrozen": false,
      "name": "NthPowerActivationLayer/370a9587-74a1-4959-b406-fa4500002c44",
      "power": -1.0
    }
```



### Example Input/Output Pair
Code from [LayerTestBase.java:120](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L120) executed in 0.00 seconds: 
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
    	[ [ -1.564 ], [ -0.508 ], [ -1.512 ] ],
    	[ [ 0.584 ], [ -1.428 ], [ 0.084 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ -0.639386189258312 ], [ -1.968503937007874 ], [ -0.6613756613756614 ] ],
    	[ [ 1.7123287671232879 ], [ -0.700280112044818 ], [ 11.904761904761903 ] ]
    ]
```



### Batch Execution
Code from [LayerTestBase.java:138](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L138) executed in 0.00 seconds: 
```java
    BatchingTester batchingTester = getBatchingTester();
    return batchingTester==null?null:batchingTester.test(layer, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (120#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (120#)}
```



### Differential Validation
Code from [LayerTestBase.java:144](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.00 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ -1.564 ], [ -0.508 ], [ -1.512 ] ],
    	[ [ 0.584 ], [ -1.428 ], [ 0.084 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.17915456786877118, negative=4, min=0.084, max=0.084, mean=-0.7240000000000001, count=6.0, positive=2, stdDev=0.8398952315616515, zeros=0}
    Output: [
    	[ [ -0.639386189258312 ], [ -1.968503937007874 ], [ -0.6613756613756614 ] ],
    	[ [ 1.7123287671232879 ], [ -0.700280112044818 ], [ 11.904761904761903 ] ]
    ]
    Outputs Statistics: {meanExponent=0.17915456786877118, negative=4, min=11.904761904761903, max=11.904761904761903, mean=1.6079241286997543, count=6.0, positive=2, stdDev=4.732063296344672, zeros=0}
    Feedback for input 0
    Inputs Values: [
    	[ [ -1.564 ], [ -0.508 ], [ -1.512 ] ],
    	[ [ 0.584 ], [ -1.428 ], [ 0.084 ] ]
    ]
    Value Statistics: {meanExponent=-0.17915456786877118, negative=4, min=0.084, max=0.084, mean=-0.7240000000000001, count=6.0, positive=2, stdDev=0.8398952315616515, zeros=0}
    Implemented Feedback: [ [ -0.408814699014266, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, -2.932069806717959, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, -3.8750077500155, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, -0.4903922353255028, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, -0.4374177654600935, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, -141.72335600907027 ] ]
    Implemented Statistics: {meanExponent=0.3583091357375425, negative=6, min=-141.72335600907027, max=-141.72335600907027, mean=-4.162973840711211, count=36.0, positive=0, stdDev=23.265274947660078, zeros=30}
    Measured: [ [ -0.40884083973291574, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, -2.9315678259256295, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, -3.8757706970016237, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, -0.4904265789229534, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, -0.4374466971202029, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, -141.55483834437987 ] ]
    Measured Statistics: {meanExponent=0.3582393586779457, negative=6, min=-141.55483834437987, max=-141.55483834437987, mean=-4.158302527307867, count=36.0, positive=0, stdDev=23.237597468017146, zeros=30}
    Feedback Error: [ [ -2.6140718649758643E-5, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 5.019807923294373E-4, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, -7.629469861236693E-4, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, -3.434359745058124E-5, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, -2.8931660109421387E-5, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.1685176646903983 ] ]
    Error Statistics: {meanExponent=-3.4626060740366635, negative=4, min=0.1685176646903983, max=0.1685176646903983, mean=0.004671313403344287, count=36.0, positive=2, stdDev=0.027695506055708512, zeros=30}
    
```

Returns: 

```
    java.lang.AssertionError: ToleranceStatistics{absoluteTol=4.7187e-03 +- 2.7687e-02 [0.0000e+00 - 1.6852e-01] (36#), relativeTol=1.4650e-04 +- 2.0227e-04 [3.1970e-05 - 5.9488e-04] (6#)}
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



