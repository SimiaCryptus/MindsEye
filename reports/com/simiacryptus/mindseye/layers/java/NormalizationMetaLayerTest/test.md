# NormalizationMetaLayer
## NormalizationMetaLayerTest
### Json Serialization
Code from [LayerTestBase.java:75](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L75) executed in 0.00 seconds: 
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
      "id": "055e4cd6-0193-4699-9154-1c170000ebd9",
      "isFrozen": false,
      "name": "NormalizationMetaLayer/055e4cd6-0193-4699-9154-1c170000ebd9",
      "inputs": [
        "3b924dbc-89d1-403e-813e-387f6993f91a"
      ],
      "nodes": {
        "fde17e33-9beb-4c48-b58c-defc79a46d76": "055e4cd6-0193-4699-9154-1c170000ebde",
        "6d1252f9-b771-4893-bf00-6a2a827e3c33": "055e4cd6-0193-4699-9154-1c170000ebdd",
        "25b6c9b0-cace-442c-887f-ce9c6ce1530e": "055e4cd6-0193-4699-9154-1c170000ebdc",
        "46c34bce-31f5-4c46-a2db-264fe78d5c35": "055e4cd6-0193-4699-9154-1c170000ebdb",
        "bd2e28a7-5ebb-4471-b9f8-24a24f1747d0": "055e4cd6-0193-4699-9154-1c170000ebda"
      },
      "layers": {
        "055e4cd6-0193-4699-9154-1c170000ebde": {
          "class": "com.simiacryptus.mindseye.layers.java.SqActivationLayer",
          "id": "055e4cd6-0193-4699-9154-1c170000ebde",
          "isFrozen": true,
          "name": "SqActivationLayer/055e4cd6-0193-4699-9154-1c170000ebde"
        },
        "055e4cd6-0193-4699-9154-1c170000ebdd": {
          "class": "com.simiacryptus.mindseye.layers.java.AvgReducerLayer",
          "id": "055e4cd6-0193-4699-9154-1c170000ebdd",
          "isFrozen": false,
          "name": "AvgReducerLayer/055e4cd6-0193-4699-9154-1c170000ebdd"
        },
        "055e4cd6-0193-4699-9154-1c170000ebdc": {
          "class": "com.simiacryptus.mindseye.layers.java.AvgMetaLayer",
          "id": "055e4cd6-0193-4699-9154-1c170000ebdc",
          "isFrozen": false,
          "name": "AvgMetaLayer/055e4cd6-0193-4699-9154-1c170000ebdc"
        },
        "055e4cd6-0193-4699-9154-1c170000ebdb": {
          "class": "com.simiacryptus.mindseye.layers.java.NthPowerActivationLayer",
          "id": "055e4cd6-0193-4699-9154-1c170000ebdb",
          "isFrozen": false,
          "name": "NthPowerActivationLayer/055e4cd6-0193-4699-9154-1c170000ebdb",
          "power": -0.5
        },
        "055e4cd6-0193-4699-9154-1c170000ebda": {
          "class": "com.simiacryptus.mindseye.layers.java.ProductInputsLayer",
          "id": "055e4cd6-0193-4699-9154-1c170000ebda",
          "isFrozen": false,
          "name": "ProductInputsLayer/055e4cd6-0193-4699-9154-1c170000ebda"
        }
      },
      "links": {
        "fde17e33-9beb-4c48-b58c-defc79a46d76": [
          "3b924dbc-89d1-403e-813e-387f6993f91a"
        ],
        "6d1252f9-b771-4893-bf00-6a2a827e3c33": [
          "fde17e33-9beb-4c48-b58c-defc79a46d76"
        ],
        "25b6c9b0-cace-442c-887f-ce9c6ce1530e": [
          "6d1252f9-b771-4893-bf00-6a2a827e3c33"
        ],
        "46c34bce-31f5-4c46-a2db-264fe78d5c35": [
          "25b6c9b0-cace-442c-887f-ce9c6ce1530e"
        ],
        "bd2e28a7-5ebb-4471-b9f8-24a24f1747d0": [
          "3b924dbc-89d1-403e-813e-387f6993f91a",
          "46c34bce-31f5-4c46-a2db-264fe78d5c35"
        ]
      },
      "labels": {},
      "head": "bd2e28a7-5ebb-4471-b9f8-24a24f1747d0"
    }
```



### Network Diagram
Code from [LayerTestBase.java:86](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L86) executed in 2.01 seconds: 
```java
    return Graphviz.fromGraph(toGraph((DAGNetwork) layer))
      .height(400).width(600).render(Format.PNG).toImage();
```

Returns: 

![Result](etc/test.1.png)



### Example Input/Output Pair
Code from [LayerTestBase.java:112](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L112) executed in 0.00 seconds: 
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
    [[ -1.868, -0.68, -0.976 ]]
    --------------------
    Output: 
    [ -1.4609809642934755, -0.5318346122695735, -0.7633390905516231 ]
```



### Differential Validation
Code from [LayerTestBase.java:130](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L130) executed in 0.00 seconds: 
```java
    getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Component: NormalizationMetaLayer/055e4cd6-0193-4699-9154-1c170000ebd9
    Inputs: [ -1.868, -0.68, -0.976 ]
    output=[ -1.4609809642934755, -0.5318346122695735, -0.7633390905516231 ]
    measured/actual: [ [ 0.782109723873603, 0.0, 0.0 ], [ 0.0, 0.7821097239846253, 0.0 ], [ 0.0, 0.0, 0.7821097239846253 ] ]
    implemented/expected: [ [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ] ]
    error: [ [ 0.782109723873603, 0.0, 0.0 ], [ 0.0, 0.7821097239846253, 0.0 ], [ 0.0, 0.0, 0.7821097239846253 ] ]
    
```

Returns: 

```
    java.lang.AssertionError: ToleranceStatistics{absoluteTol=2.6070e-01 +- 3.6869e-01 [0.0000e+00 - 7.8211e-01] (9#), relativeTol=1.0000e+00 +- 0.0000e+00 [1.0000e+00 - 1.0000e+00] (3#)}
    	at com.simiacryptus.mindseye.layers.DerivativeTester.testFeedback(DerivativeTester.java:221)
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
    	at com.simiacryptus.mindseye.layers.LayerTestBase.lambda$test$15(LayerTestBase.java:131)
    	at com.simiacryptus.util.io.NotebookOutput.lambda$code$1(NotebookOutput.java:142)
    	at com.simiacryptus.util.io.MarkdownNotebookOutput.lambda$null$1(MarkdownNotebookOutput.java:136)
    	at com.simiacryptus.util.lang.TimedResult.time(TimedResult.java:59)
    	at com.simiacryptus.util.io.MarkdownNotebookOutput.lambda$code$2(MarkdownNotebookOutput.java:136)
    	at com.simiacryptus.util.test.SysOutInterceptor.withOutput(SysOutInterceptor.java:77)
    	at com.simiacryptus.util.io.MarkdownNotebookOutput.code(MarkdownNotebookOutput.java:134)
    	at com.simiacryptus.util.io.NotebookOutput.code(NotebookOutput.java:141)
    	at com.simiacryptus.mindseye.layers.LayerTestBase.test(LayerTestBase.java:130)
    	at com.simiacryptus.mindseye.layers.LayerTestBase.test(LayerTestBase.java:65)
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



