# ConvolutionLayer
## IrregularTest
### Json Serialization
Code from [LayerTestBase.java:84](../../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L84) executed in 0.00 seconds: 
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
      "id": "79cb4e2c-4a9a-4706-8f49-c2d500000035",
      "isFrozen": false,
      "name": "ConvolutionLayer/79cb4e2c-4a9a-4706-8f49-c2d500000035",
      "filter": {
        "dimensions": [
          1,
          1,
          6
        ],
        "data": [
          1.572,
          0.036,
          1.848,
          0.256,
          0.552,
          -1.0
        ]
      },
      "strideX": 1,
      "strideY": 1
    }
```



### Example Input/Output Pair
Code from [LayerTestBase.java:121](../../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L121) executed in 0.01 seconds: 
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
    	[ [ 1.356, 0.812 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 2.339503765106201, 0.4970399737358093, 1.6938879489898682 ] ]
    ]
```



### Reference Implementation
Code from [LayerTestBase.java:132](../../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L132) executed in 0.02 seconds: 
```java
    System.out.println(new GsonBuilder().setPrettyPrinting().create().toJson(referenceLayer.getJson()));
    getEquivalencyTester().test(referenceLayer, layer, inputPrototype);
```
Logging: 
```
    {
      "class": "com.simiacryptus.mindseye.layers.aparapi.ConvolutionLayer",
      "id": "79cb4e2c-4a9a-4706-8f49-c2d50000003f",
      "isFrozen": false,
      "name": "ConvolutionLayer/79cb4e2c-4a9a-4706-8f49-c2d50000003f",
      "filter": {
        "dimensions": [
          1,
          1,
          6
        ],
        "data": [
          1.572,
          0.036,
          1.848,
          0.256,
          0.552,
          -1.0
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
    	[ [ 1.356, 0.812 ] ]
    ]
    Error: [
    	[ [ -2.3489379907815078E-7, -2.6264190766678297E-8, -5.1010131896234157E-8 ] ]
    ]
    Accuracy:
    absoluteTol: 1.0406e-07 +- 9.3066e-08 [2.6264e-08 - 2.3489e-07] (3#)
    relativeTol: 3.0560e-08 +- 1.4643e-08 [1.5057e-08 - 5.0202e-08] (3#)
    
```

### Differential Validation
Code from [LayerTestBase.java:139](../../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L139) executed in 0.00 seconds: 
```java
    getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Component: ConvolutionLayer/79cb4e2c-4a9a-4706-8f49-c2d500000035
    Inputs: [
    	[ [ 1.356, 0.812 ] ]
    ]
    output=[
    	[ [ 2.339503765106201, 0.4970399737358093, 1.6938879489898682 ] ]
    ]
    measured/actual: [ [ 1.5721321105957031, 0.03600120544433594, 1.847982406616211 ], [ 0.2560615539550781, 0.5520284175872803, -0.9999275207519531 ] ]
    implemented/expected: [ [ 0.0, 0.0, 1.8480000495910645 ], [ 0.0, 0.0, -1.0 ] ]
    error: [ [ 1.5721321105957031, 0.03600120544433594, -1.7642974853515625E-5 ], [ 0.2560615539550781, 0.5520284175872803, 7.2479248046875E-5 ] ]
    
```

Returns: 

```
    java.lang.AssertionError: ToleranceStatistics{absoluteTol=4.0272e-01 +- 5.5829e-01 [1.7643e-05 - 1.5721e+00] (6#), relativeTol=6.6667e-01 +- 4.7139e-01 [4.7736e-06 - 1.0000e+00] (6#)}
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



