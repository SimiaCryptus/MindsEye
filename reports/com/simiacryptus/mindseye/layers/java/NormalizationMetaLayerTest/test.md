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
      "id": "f4569375-56fe-4e46-925c-95f400000a35",
      "isFrozen": false,
      "name": "NormalizationMetaLayer/f4569375-56fe-4e46-925c-95f400000a35",
      "inputs": [
        "bc2ee76a-1939-4c6c-a4d1-fa59a0574fe3"
      ],
      "nodes": {
        "a49f056f-8743-4e0f-8d8f-813ae66e2451": "f4569375-56fe-4e46-925c-95f400000a3a",
        "7907cd41-806c-4cc9-b675-2a9d65ff69af": "f4569375-56fe-4e46-925c-95f400000a39",
        "2b446efd-66b3-4222-a3e8-38c4e3b85db2": "f4569375-56fe-4e46-925c-95f400000a38",
        "a40af5de-332b-42d7-bae9-71d2cb5d7fe1": "f4569375-56fe-4e46-925c-95f400000a37",
        "8a481977-ec04-43d2-bcb6-fb92acc81441": "f4569375-56fe-4e46-925c-95f400000a36"
      },
      "layers": {
        "f4569375-56fe-4e46-925c-95f400000a3a": {
          "class": "com.simiacryptus.mindseye.layers.java.SqActivationLayer",
          "id": "f4569375-56fe-4e46-925c-95f400000a3a",
          "isFrozen": true,
          "name": "SqActivationLayer/f4569375-56fe-4e46-925c-95f400000a3a"
        },
        "f4569375-56fe-4e46-925c-95f400000a39": {
          "class": "com.simiacryptus.mindseye.layers.java.AvgReducerLayer",
          "id": "f4569375-56fe-4e46-925c-95f400000a39",
          "isFrozen": false,
          "name": "AvgReducerLayer/f4569375-56fe-4e46-925c-95f400000a39"
        },
        "f4569375-56fe-4e46-925c-95f400000a38": {
          "class": "com.simiacryptus.mindseye.layers.java.AvgMetaLayer",
          "id": "f4569375-56fe-4e46-925c-95f400000a38",
          "isFrozen": false,
          "name": "AvgMetaLayer/f4569375-56fe-4e46-925c-95f400000a38"
        },
        "f4569375-56fe-4e46-925c-95f400000a37": {
          "class": "com.simiacryptus.mindseye.layers.java.NthPowerActivationLayer",
          "id": "f4569375-56fe-4e46-925c-95f400000a37",
          "isFrozen": false,
          "name": "NthPowerActivationLayer/f4569375-56fe-4e46-925c-95f400000a37",
          "power": -0.5
        },
        "f4569375-56fe-4e46-925c-95f400000a36": {
          "class": "com.simiacryptus.mindseye.layers.java.ProductInputsLayer",
          "id": "f4569375-56fe-4e46-925c-95f400000a36",
          "isFrozen": false,
          "name": "ProductInputsLayer/f4569375-56fe-4e46-925c-95f400000a36"
        }
      },
      "links": {
        "a49f056f-8743-4e0f-8d8f-813ae66e2451": [
          "bc2ee76a-1939-4c6c-a4d1-fa59a0574fe3"
        ],
        "7907cd41-806c-4cc9-b675-2a9d65ff69af": [
          "a49f056f-8743-4e0f-8d8f-813ae66e2451"
        ],
        "2b446efd-66b3-4222-a3e8-38c4e3b85db2": [
          "7907cd41-806c-4cc9-b675-2a9d65ff69af"
        ],
        "a40af5de-332b-42d7-bae9-71d2cb5d7fe1": [
          "2b446efd-66b3-4222-a3e8-38c4e3b85db2"
        ],
        "8a481977-ec04-43d2-bcb6-fb92acc81441": [
          "bc2ee76a-1939-4c6c-a4d1-fa59a0574fe3",
          "a40af5de-332b-42d7-bae9-71d2cb5d7fe1"
        ]
      },
      "labels": {},
      "head": "8a481977-ec04-43d2-bcb6-fb92acc81441"
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
    [[ 0.932, 1.244, -0.72 ]]
    --------------------
    Output: 
    [ 0.9423343947630005, 1.257793977559198, -0.7279836526066098 ]
```



### Differential Validation
Code from [LayerTestBase.java:139](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L139) executed in 0.00 seconds: 
```java
    getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Feedback for input 0
    Inputs: [ 0.932, 1.244, -0.72 ]
    Output: [ 0.9423343947630005, 1.257793977559198, -0.7279836526066098 ]
    Measured: [ [ 1.0110884063985193, 0.0, 0.0 ], [ 0.0, 1.0110884063996295, 0.0 ], [ 0.0, 0.0, 1.0110884063985193 ] ]
    Implemented: [ [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ] ]
    Error: [ [ 1.0110884063985193, 0.0, 0.0 ], [ 0.0, 1.0110884063996295, 0.0 ], [ 0.0, 0.0, 1.0110884063985193 ] ]
    
```

Returns: 

```
    java.lang.AssertionError: ToleranceStatistics{absoluteTol=3.3703e-01 +- 4.7663e-01 [0.0000e+00 - 1.0111e+00] (9#), relativeTol=1.0000e+00 +- 0.0000e+00 [1.0000e+00 - 1.0000e+00] (3#)}
    	at com.simiacryptus.mindseye.layers.DerivativeTester.testFeedback(DerivativeTester.java:283)
    	at com.simiacryptus.mindseye.layers.DerivativeTester.lambda$test$0(DerivativeTester.java:77)
    	at java.util.stream.IntPipeline$4$1.accept(IntPipeline.java:250)
    	at java.util.stream.Streams$RangeIntSpliterator.forEachRemaining(Streams.java:110)
    	at java.util.Spliterator$OfInt.forEachRemaining(Spliterator.java:693)
    	at java.util.stream.AbstractPipeline.copyInto(AbstractPipeline.java:481)
    	at java.util.stream.AbstractPipeline.wrapAndCopyInto(AbstractPipeline.java:471)
    	at java.util.stream.ReduceOps$ReduceOp.evaluateSequential(ReduceOps.java:708)
    	at java.util.stream.AbstractPipeline.evaluate(AbstractPipeline.java:234)
    	at java.util.stream.ReferencePipeline.reduce(ReferencePipeline.java:479)
    	at com.simiacryptus.mindseye.layers.DerivativeTester.test(DerivativeTester.java:78)
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



