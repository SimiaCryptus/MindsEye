# EntropyLossLayer
## EntropyLossLayerTest
### Json Serialization
Code from [LayerTestBase.java:83](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L83) executed in 0.00 seconds: 
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
      "class": "com.simiacryptus.mindseye.layers.java.EntropyLossLayer",
      "id": "370a9587-74a1-4959-b406-fa4500002bad",
      "isFrozen": false,
      "name": "EntropyLossLayer/370a9587-74a1-4959-b406-fa4500002bad"
    }
```



### Example Input/Output Pair
Code from [LayerTestBase.java:120](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L120) executed in 0.00 seconds: 
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
    [[ 0.740892618921717, 0.8724377512963908, 0.40516602604812435, 0.5988824823115886 ],
    [ 0.054199933214984175, 0.4494344258926718, 0.4846226823998425, 0.6291579492519833 ]]
    --------------------
    Output: 
    [ 0.8379854754911811 ]
```



### Batch Execution
Code from [LayerTestBase.java:138](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L138) executed in 0.00 seconds: 
```java
    BatchingTester batchingTester = getBatchingTester();
    return batchingTester==null?null:batchingTester.test(layer, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (90#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (67#)}
```



### Differential Validation
Code from [LayerTestBase.java:144](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.00 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [ 0.740892618921717, 0.8724377512963908, 0.40516602604812435, 0.5988824823115886 ],
    [ 0.054199933214984175, 0.4494344258926718, 0.4846226823998425, 0.6291579492519833 ]
    Inputs Statistics: {meanExponent=-0.20113391261407923, negative=0, min=0.5988824823115886, max=0.5988824823115886, mean=0.6543447196444552, count=4.0, positive=4, stdDev=0.17336463771252522, zeros=0},
    {meanExponent=-0.5322928719405488, negative=0, min=0.6291579492519833, max=0.6291579492519833, mean=0.4043537476898704, count=4.0, positive=4, stdDev=0.21308448574332584, zeros=0}
    Output: [ 0.8379854754911811 ]
    Outputs Statistics: {meanExponent=-0.07676350877867162, negative=0, min=0.8379854754911811, max=0.8379854754911811, mean=0.8379854754911811, count=1.0, positive=1, stdDev=0.0, zeros=0}
    Feedback for input 0
    Inputs Values: [ 0.740892618921717, 0.8724377512963908, 0.40516602604812435, 0.5988824823115886 ]
    Value Statistics: {meanExponent=-0.20113391261407923, negative=0, min=0.5988824823115886, max=0.5988824823115886, mean=0.6543447196444552, count=4.0, positive=4, stdDev=0.17336463771252522, zeros=0}
    Implemented Feedback: [ [ -0.07315491048333815 ], [ -0.5151478431840425 ], [ -1.1961088818988999 ], [ -1.0505532685202885 ] ]
    Implemented Statistics: {meanExponent=-0.33115895932646955, negative=4, min=-1.0505532685202885, max=-1.0505532685202885, mean=-0.7087412260216422, count=4.0, positive=0, stdDev=0.44604121636078975, zeros=0}
    Measured Feedback: [ [ -0.07315490435644278 ], [ -0.5151478355003292 ], [ -1.196108856404976 ], [ -1.0505532666016393 ] ]
    Measured Statistics: {meanExponent=-0.33115897255163457, negative=4, min=-1.0505532666016393, max=-1.0505532666016393, mean=-0.7087412157158468, count=4.0, positive=0, stdDev=0.4460412120455807, zeros=0}
    Feedback Error: [ [ 6.1268953660542635E-9 ], [ 7.683713332262698E-9 ], [ 2.5493923860864243E-8 ], [ 1.9186492394851484E-9 ] ]
    Error Statistics: {meanExponent=-8.159439028126947, negative=0, min=1.9186492394851484E-9, max=1.9186492394851484E-9, mean=1.0305795449666588E-8, count=4.0, positive=4, stdDev=9.018896644340907E-9, zeros=0}
    Feedback for input 1
    Inputs Values: [ 0.054199933214984175, 0.4494344258926718, 0.4846226823998425, 0.6291579492519833 ]
    Value Statistics: {meanExponent=-0.5322928719405488, negative=0, min=0.6291579492519833, max=0.6291579492519833, mean=0.4043537476898704, count=4.0, positive=4, stdDev=0.21308448574332584, zeros=0}
    Implemented Feedback: [ [ -0.740892618921717 ], [ -0.8724377512963908 ], [ -0.40516602604812435 ], [ -0.5988824823115886 ] ]
    Implemented Statistics: {meanExponent=-0.20113391261407923, negative=4, min=-0.5988824823115886, max=-0.5988824823115886, mean=-0.6543447196444552, count=4.0, positive=0, stdDev=0.17336463771252522, zeros=0}
    Measured: [ [ 0.2998995940473037 ], [ 0.1364639734546813 ], [ 0.9034583525746598 ], [ 0.512689890541651 ] ]
    Measured Statistics: {meanExponent=-0.43056080456935986, negative=0, min=0.512689890541651, max=0.512689890541651, mean=0.46312795265457396, count=4.0, positive=4, stdDev=0.2870975875644189, zeros=0}
    Feedback Error: [ [ 1.0407922129690208 ], [ 1.008901724751072 ], [ 1.3086243786227842 ], [ 1.1115723728532396 ] ]
    Error Statistics: {meanExponent=0.04599141238854975, negative=0, min=1.1115723728532396, max=1.1115723728532396, mean=1.1174726722990291, count=4.0, positive=4, stdDev=0.11644884269342906, zeros=0}
    
```

Returns: 

```
    java.lang.AssertionError: ToleranceStatistics{absoluteTol=1.1175e+00 +- 1.1645e-01 [1.0089e+00 - 1.3086e+00] (4#), relativeTol=1.0000e+00 +- 0.0000e+00 [1.0000e+00 - 1.0000e+00] (4#)}
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



