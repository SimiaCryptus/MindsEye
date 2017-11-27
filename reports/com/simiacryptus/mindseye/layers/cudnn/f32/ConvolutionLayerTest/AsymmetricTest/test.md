# ConvolutionLayer
## AsymmetricTest
### Json Serialization
Code from [LayerTestBase.java:84](../../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L84) executed in 0.04 seconds: 
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
      "id": "79cb4e2c-4a9a-4706-8f49-c2d500000002",
      "isFrozen": false,
      "name": "ConvolutionLayer/79cb4e2c-4a9a-4706-8f49-c2d500000002",
      "filter": {
        "dimensions": [
          1,
          1,
          8
        ],
        "data": [
          -1.416,
          1.456,
          -0.524,
          0.008,
          0.748,
          0.076,
          -0.548,
          -0.396
        ]
      },
      "strideX": 1,
      "strideY": 1
    }
```



### Example Input/Output Pair
Code from [LayerTestBase.java:121](../../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L121) executed in 0.02 seconds: 
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
    	[ [ -0.464, 1.58 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 1.8388640880584717, -0.5555039644241333, -0.622704029083252, -0.6293920278549194 ] ]
    ]
```



### Reference Implementation
Code from [LayerTestBase.java:132](../../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L132) executed in 0.73 seconds: 
```java
    System.out.println(new GsonBuilder().setPrettyPrinting().create().toJson(referenceLayer.getJson()));
    getEquivalencyTester().test(referenceLayer, layer, inputPrototype);
```
Logging: 
```
    {
      "class": "com.simiacryptus.mindseye.layers.aparapi.ConvolutionLayer",
      "id": "79cb4e2c-4a9a-4706-8f49-c2d50000000c",
      "isFrozen": false,
      "name": "ConvolutionLayer/79cb4e2c-4a9a-4706-8f49-c2d50000000c",
      "filter": {
        "dimensions": [
          1,
          1,
          8
        ],
        "data": [
          -1.416,
          1.456,
          -0.524,
          0.008,
          0.748,
          0.076,
          -0.548,
          -0.396
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
    	[ [ -0.464, 1.58 ] ]
    ]
    Error: [
    	[ [ 8.805847162562941E-8, 3.5575866697001857E-8, -2.9083251917860764E-8, -2.7854919371073095E-8 ] ]
    ]
    Accuracy:
    absoluteTol: 4.5143e-08 +- 2.4950e-08 [2.7855e-08 - 8.8058e-08] (4#)
    relativeTol: 2.5361e-08 +- 3.9004e-09 [2.2128e-08 - 3.2021e-08] (4#)
    
```

### Differential Validation
Code from [LayerTestBase.java:139](../../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L139) executed in 0.00 seconds: 
```java
    getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Component: ConvolutionLayer/79cb4e2c-4a9a-4706-8f49-c2d500000002
    Inputs: [
    	[ [ -0.464, 1.58 ] ]
    ]
    output=[
    	[ [ 1.8388640880584717, -0.5555039644241333, -0.622704029083252, -0.6293920278549194 ] ]
    ]
    measured/actual: [ [ -1.4159679412841797, 1.4559626579284668, -0.523984432220459, 0.007987022399902344 ], [ 0.7479190826416016, 0.07599592208862305, -0.547945499420166, -0.3959536552429199 ] ]
    implemented/expected: [ [ 0.0, 0.0, -0.5239999890327454, 0.00800000037997961 ], [ 0.0, 0.0, -0.5479999780654907, -0.3959999978542328 ] ]
    error: [ [ -1.4159679412841797, 1.4559626579284668, 1.5556812286376953E-5, -1.2977980077266693E-5 ], [ 0.7479190826416016, 0.07599592208862305, 5.447864532470703E-5, 4.634261131286621E-5 ] ]
    
```

Returns: 

```
    java.lang.AssertionError: ToleranceStatistics{absoluteTol=4.6200e-01 +- 6.1058e-01 [1.2978e-05 - 1.4560e+00] (8#), relativeTol=5.0012e-01 +- 4.9988e-01 [1.4845e-05 - 1.0000e+00] (8#)}
    	at com.simiacryptus.mindseye.layers.DerivativeTester.testFeedback(DerivativeTester.java:224)
    	at com.simiacryptus.mindseye.layers.DerivativeTester.lambda$test$0(DerivativeTester.java:71)
    	at java.util.stream.IntPipeline$4$1.accept(IntPipeline.java:250)
    	at java.util.stream.Streams$RangeIntSpliterator.forEachRemaining(Streams.java:110)
    	at java.util.Spliterator$OfInt.forEachRemaining(Spliterator.java:693)
    	at java.util.stream.AbstractPipeline.copyInto(AbstractPipeline.java:481)
    	at java.util.stream.AbstractPipeline.wrapAndCopyInto(AbstractPipeline.java:471)
    	at java.util.stream.ReduceOps$ReduceOp.evaluateSequential(ReduceOps.java:708)
    	at java.util.stream.AbstractPipeline.evaluate(AbstractPipeline.java:234)
    	at java.util.stream.ReferencePipeline.reduce(ReferencePipeline.java:479)
    	at com.simiacryptus.mindseye.layers.DerivativeTester.test(DerivativeTester.java:72)
    	at com.simiacryptus.mindseye.layers.LayerTestBase.lambda$test$15(LayerTestBase.java:140)
    	at com.simiacryptus.util.io.NotebookOutput.lambda$code$1(NotebookOutput.java:142)
    	at com.simiacryptus.util.io.MarkdownNotebookOutput.lambda$null$1(MarkdownNotebookOutput.java:136)
    	at com.simiacryptus.util.lang.TimedResult.time(TimedResult.java:59)
    	at com.simiacryptus.util.io.MarkdownNotebookOutput.lambda$code$2(MarkdownNotebookOutput.java:136)
    	at com.simiacryptus.util.test.SysOutInterceptor.withOutput(SysOutInterceptor.java:77)
    	at com.simiacryptus.util.io.MarkdownNotebookOutput.code(MarkdownNotebookOutput.java:134)
    	at com.simiacryptus.util.io.NotebookOutput.code(NotebookOutput.java:141)
    	at com.simiacryptus.mindseye.layers.LayerTestBase.test(LayerTestBase.java:139)
    	at com.simiacryptus.mindseye.layers.LayerTestBase.test(LayerTestBase.java:69)
    	at sun.reflect.NativeMethodAccessorImpl.invoke0(Native Method)
    	at sun.reflect.NativeMethodAccessorImpl.invoke(Nat
```
...[skipping 75 bytes](etc/1.txt)...
```
    l.invoke(DelegatingMethodAccessorImpl.java:43)
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



