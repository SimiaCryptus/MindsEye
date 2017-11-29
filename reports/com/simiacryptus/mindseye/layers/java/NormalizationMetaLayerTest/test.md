# NormalizationMetaLayer
## NormalizationMetaLayerTest
### Json Serialization
Code from [LayerTestBase.java:84](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L84) executed in 0.00 seconds: 
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
      "class": "com.simiacryptus.mindseye.layers.java.NormalizationMetaLayer",
      "id": "c88cbdf1-1c2a-4a5e-b964-890900000f87",
      "isFrozen": false,
      "name": "NormalizationMetaLayer/c88cbdf1-1c2a-4a5e-b964-890900000f87",
      "inputs": [
        "5006d633-b15c-4dd0-bb7a-ca4555370a90"
      ],
      "nodes": {
        "49d7914a-5104-485e-8e58-469d9982ca0c": "c88cbdf1-1c2a-4a5e-b964-890900000f8c",
        "bab5e4f5-4cd4-49a1-844d-9ef8e67f492c": "c88cbdf1-1c2a-4a5e-b964-890900000f8b",
        "563f2fa4-1678-42e0-b0c8-1d812b9a75d2": "c88cbdf1-1c2a-4a5e-b964-890900000f8a",
        "67609569-5763-412d-bd44-a48635cc82b5": "c88cbdf1-1c2a-4a5e-b964-890900000f89",
        "14f0d16c-eddf-48f2-b268-4db9a8920daa": "c88cbdf1-1c2a-4a5e-b964-890900000f88"
      },
      "layers": {
        "c88cbdf1-1c2a-4a5e-b964-890900000f8c": {
          "class": "com.simiacryptus.mindseye.layers.java.SqActivationLayer",
          "id": "c88cbdf1-1c2a-4a5e-b964-890900000f8c",
          "isFrozen": true,
          "name": "SqActivationLayer/c88cbdf1-1c2a-4a5e-b964-890900000f8c"
        },
        "c88cbdf1-1c2a-4a5e-b964-890900000f8b": {
          "class": "com.simiacryptus.mindseye.layers.java.AvgReducerLayer",
          "id": "c88cbdf1-1c2a-4a5e-b964-890900000f8b",
          "isFrozen": false,
          "name": "AvgReducerLayer/c88cbdf1-1c2a-4a5e-b964-890900000f8b"
        },
        "c88cbdf1-1c2a-4a5e-b964-890900000f8a": {
          "class": "com.simiacryptus.mindseye.layers.java.AvgMetaLayer",
          "id": "c88cbdf1-1c2a-4a5e-b964-890900000f8a",
          "isFrozen": false,
          "name": "AvgMetaLayer/c88cbdf1-1c2a-4a5e-b964-890900000f8a"
        },
        "c88cbdf1-1c2a-4a5e-b964-890900000f89": {
          "class": "com.simiacryptus.mindseye.layers.java.NthPowerActivationLayer",
          "id": "c88cbdf1-1c2a-4a5e-b964-890900000f89",
          "isFrozen": false,
          "name": "NthPowerActivationLayer/c88cbdf1-1c2a-4a5e-b964-890900000f89",
          "power": -0.5
        },
        "c88cbdf1-1c2a-4a5e-b964-890900000f88": {
          "class": "com.simiacryptus.mindseye.layers.java.ProductInputsLayer",
          "id": "c88cbdf1-1c2a-4a5e-b964-890900000f88",
          "isFrozen": false,
          "name": "ProductInputsLayer/c88cbdf1-1c2a-4a5e-b964-890900000f88"
        }
      },
      "links": {
        "49d7914a-5104-485e-8e58-469d9982ca0c": [
          "5006d633-b15c-4dd0-bb7a-ca4555370a90"
        ],
        "bab5e4f5-4cd4-49a1-844d-9ef8e67f492c": [
          "49d7914a-5104-485e-8e58-469d9982ca0c"
        ],
        "563f2fa4-1678-42e0-b0c8-1d812b9a75d2": [
          "bab5e4f5-4cd4-49a1-844d-9ef8e67f492c"
        ],
        "67609569-5763-412d-bd44-a48635cc82b5": [
          "563f2fa4-1678-42e0-b0c8-1d812b9a75d2"
        ],
        "14f0d16c-eddf-48f2-b268-4db9a8920daa": [
          "5006d633-b15c-4dd0-bb7a-ca4555370a90",
          "67609569-5763-412d-bd44-a48635cc82b5"
        ]
      },
      "labels": {},
      "head": "14f0d16c-eddf-48f2-b268-4db9a8920daa"
    }
```



### Network Diagram
Code from [LayerTestBase.java:95](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L95) executed in 0.16 seconds: 
```java
    return Graphviz.fromGraph(toGraph((DAGNetwork) layer))
      .height(400).width(600).render(Format.PNG).toImage();
```

Returns: 

![Result](etc/test.1.png)



### Example Input/Output Pair
Code from [LayerTestBase.java:121](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L121) executed in 0.00 seconds: 
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
    [[ -0.136, -0.54, -0.64 ]]
    --------------------
    Output: 
    [ -0.27766746239076623, -1.1025031594927481, -1.3066704112506644 ]
```



### Differential Validation
Code from [LayerTestBase.java:139](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L139) executed in 0.00 seconds: 
```java
    getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Component: NormalizationMetaLayer/c88cbdf1-1c2a-4a5e-b964-890900000f87
    Inputs: [ -0.136, -0.54, -0.64 ]
    output=[ -0.27766746239076623, -1.1025031594927481, -1.3066704112506644 ]
    measured/actual: [ [ 2.0416725175792516, 0.0, 0.0 ], [ 0.0, 2.0416725175786965, 0.0 ], [ 0.0, 0.0, 2.0416725175786965 ] ]
    implemented/expected: [ [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ] ]
    error: [ [ 2.0416725175792516, 0.0, 0.0 ], [ 0.0, 2.0416725175786965, 0.0 ], [ 0.0, 0.0, 2.0416725175786965 ] ]
    
```

Returns: 

```
    java.lang.AssertionError: ToleranceStatistics{absoluteTol=6.8056e-01 +- 9.6245e-01 [0.0000e+00 - 2.0417e+00] (9#), relativeTol=1.0000e+00 +- 0.0000e+00 [1.0000e+00 - 1.0000e+00] (3#)}
    	at com.simiacryptus.mindseye.layers.DerivativeTester.testFeedback(DerivativeTester.java:266)
    	at com.simiacryptus.mindseye.layers.DerivativeTester.lambda$test$0(DerivativeTester.java:74)
    	at java.util.stream.IntPipeline$4$1.accept(IntPipeline.java:250)
    	at java.util.stream.Streams$RangeIntSpliterator.forEachRemaining(Streams.java:110)
    	at java.util.Spliterator$OfInt.forEachRemaining(Spliterator.java:693)
    	at java.util.stream.AbstractPipeline.copyInto(AbstractPipeline.java:481)
    	at java.util.stream.AbstractPipeline.wrapAndCopyInto(AbstractPipeline.java:471)
    	at java.util.stream.ReduceOps$ReduceOp.evaluateSequential(ReduceOps.java:708)
    	at java.util.stream.AbstractPipeline.evaluate(AbstractPipeline.java:234)
    	at java.util.stream.ReferencePipeline.reduce(ReferencePipeline.java:479)
    	at com.simiacryptus.mindseye.layers.DerivativeTester.test(DerivativeTester.java:75)
    	at com.simiacryptus.mindseye.layers.LayerTestBase.lambda$test$15(LayerTestBase.java:140)
    	at com.simiacryptus.util.io.NotebookOutput.lambda$code$1(NotebookOutput.java:157)
    	at com.simiacryptus.util.io.MarkdownNotebookOutput.lambda$null$1(MarkdownNotebookOutput.java:136)
    	at com.simiacryptus.util.lang.TimedResult.time(TimedResult.java:59)
    	at com.simiacryptus.util.io.MarkdownNotebookOutput.lambda$code$2(MarkdownNotebookOutput.java:136)
    	at com.simiacryptus.util.test.SysOutInterceptor.withOutput(SysOutInterceptor.java:82)
    	at com.simiacryptus.util.io.MarkdownNotebookOutput.code(MarkdownNotebookOutput.java:134)
    	at com.simiacryptus.util.io.NotebookOutput.code(NotebookOutput.java:156)
    	at com.simiacryptus.mindseye.layers.LayerTestBase.test(LayerTestBase.java:139)
    	at com.simiacryptus.mindseye.layers.LayerTestBase.test(LayerTestBase.java:69)
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



