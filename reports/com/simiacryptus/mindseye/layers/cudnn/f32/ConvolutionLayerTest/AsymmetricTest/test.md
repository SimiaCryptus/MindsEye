# ConvolutionLayer
## AsymmetricTest
### Json Serialization
Code from [LayerTestBase.java:83](../../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L83) executed in 0.00 seconds: 
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
      "class": "com.simiacryptus.mindseye.layers.cudnn.f32.ConvolutionLayer",
      "id": "370a9587-74a1-4959-b406-fa45000002f3",
      "isFrozen": false,
      "name": "ConvolutionLayer/370a9587-74a1-4959-b406-fa45000002f3",
      "filter": {
        "dimensions": [
          1,
          1,
          8
        ],
        "data": [
          -0.516,
          -1.984,
          -1.52,
          -0.644,
          1.772,
          -0.58,
          0.476,
          0.648
        ]
      },
      "strideX": 1,
      "strideY": 1
    }
```



### Example Input/Output Pair
Code from [LayerTestBase.java:120](../../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L120) executed in 0.00 seconds: 
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
    	[ [ 1.056, 1.264 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 1.694912075996399, -2.828223943710327, -1.0034558773040771, 0.1390080749988556 ] ]
    ]
```



### Reference Implementation
Code from [LayerTestBase.java:131](../../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L131) executed in 0.02 seconds: 
```java
    System.out.println(new GsonBuilder().setPrettyPrinting().create().toJson(referenceLayer.getJson()));
    getEquivalencyTester().test(referenceLayer, layer, inputPrototype);
```
Logging: 
```
    {
      "class": "com.simiacryptus.mindseye.layers.aparapi.ConvolutionLayer",
      "id": "370a9587-74a1-4959-b406-fa45000002fc",
      "isFrozen": false,
      "name": "ConvolutionLayer/370a9587-74a1-4959-b406-fa45000002fc",
      "filter": {
        "dimensions": [
          1,
          1,
          8
        ],
        "data": [
          -0.516,
          -1.984,
          -1.52,
          -0.644,
          1.772,
          -0.58,
          0.476,
          0.648
        ]
      },
      "skip": {
        "dimensions": [
          1,
          1
        ]
      },
      "simple": true
    }
    Inputs: [
    	[ [ 1.056, 1.264 ] ]
    ]
    Error: [
    	[ [ 7.599639895161658E-8, 5.6289672922815726E-8, 1.2269592297720067E-7, 7.49988555703851E-8 ] ]
    ]
    Accuracy:
    absoluteTol: 8.2495e-08 +- 2.4501e-08 [5.6290e-08 - 1.2270e-07] (4#)
    relativeTol: 9.0818e-08 +- 1.0502e-07 [9.9514e-09 - 2.6976e-07] (4#)
    
```

### Batch Execution
Code from [LayerTestBase.java:138](../../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L138) executed in 0.04 seconds: 
```java
    BatchingTester batchingTester = getBatchingTester();
    return batchingTester==null?null:batchingTester.test(layer, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (60#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (60#)}
```



### Differential Validation
Code from [LayerTestBase.java:144](../../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.00 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ 1.056, 1.264 ] ]
    ]
    Inputs Statistics: {meanExponent=0.06270549607207984, negative=0, min=1.264, max=1.264, mean=1.1600000000000001, count=2.0, positive=2, stdDev=0.10399999999999864, zeros=0}
    Output: [
    	[ [ 1.694912075996399, -2.828223943710327, -1.0034558773040771, 0.1390080749988556 ] ]
    ]
    Outputs Statistics: {meanExponent=-0.043700180281069895, negative=2, min=0.1390080749988556, max=0.1390080749988556, mean=-0.49943991750478745, count=4.0, positive=2, stdDev=1.6507622208359571, zeros=0}
    Feedback for input 0
    Inputs Values: [
    	[ [ 1.056, 1.264 ] ]
    ]
    Value Statistics: {meanExponent=0.06270549607207984, negative=0, min=1.264, max=1.264, mean=1.1600000000000001, count=2.0, positive=2, stdDev=0.10399999999999864, zeros=0}
    Implemented Feedback: [ [ 0.0, 0.0, -1.5199999809265137, -0.6439999938011169 ], [ 0.0, 0.0, 0.47600001096725464, 0.6480000019073486 ] ]
    Implemented Statistics: {meanExponent=-0.1300221461124223, negative=2, min=0.6480000019073486, max=0.6480000019073486, mean=-0.12999999523162842, count=8.0, positive=2, stdDev=0.6360440193948257, zeros=4}
    Measured: [ [ -0.5161762237548828, -1.983642578125, -1.5211105346679688, -0.6443262100219727 ], [ 1.7702579498291016, -0.5793571472167969, 0.4756450653076172, 0.6473064422607422 ] ]
    Measured Statistics: {meanExponent=-0.06238702037485962, negative=5, min=0.6473064422607422, max=0.6473064422607422, mean=-0.293925404548645, count=8.0, positive=3, stdDev=1.137639217805605, zeros=0}
    Feedback Error: [ [ -0.5161762237548828, -1.983642578125, -0.0011105537414550781, -3.262162208557129E-4 ], [ 1.7702579498291016, -0.5793571472167969, -3.5494565963745117E-4, -6.935596466064453E-4 ] ]
    Error Statistics: {meanExponent=-1.6285581051927402, negative=7, min=-6.935596466064453E-4, max=-6.935596466064453E-4, mean=-0.1639254093170166, count=8.0, positive=1, stdDev=0.9653867950777939, zeros=0}
    
```

Returns: 

```
    java.lang.AssertionError: ToleranceStatistics{absoluteTol=6.0649e-01 +- 7.6877e-01 [3.2622e-04 - 1.9836e+00] (8#), relativeTol=5.0019e-01 +- 4.9981e-01 [2.5321e-04 - 1.0000e+00] (8#)}
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
    	at sun.reflect.NativeMethodAccessorImpl.invoke0(Native Method)
    	at sun.reflect.NativeMethodAccessorImpl.invoke(NativeMethodAccessorImpl.java:62)
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



