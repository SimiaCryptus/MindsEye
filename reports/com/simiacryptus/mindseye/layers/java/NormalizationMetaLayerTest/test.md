# NormalizationMetaLayer
## NormalizationMetaLayerTest
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
      "class": "com.simiacryptus.mindseye.layers.java.NormalizationMetaLayer",
      "id": "370a9587-74a1-4959-b406-fa4500002c3b",
      "isFrozen": false,
      "name": "NormalizationMetaLayer/370a9587-74a1-4959-b406-fa4500002c3b",
      "inputs": [
        "cc18c7d2-9920-4c90-adf1-52af0f9b6e59"
      ],
      "nodes": {
        "8e6c9675-20c2-423a-8bf8-98c2d320391c": "370a9587-74a1-4959-b406-fa4500002c40",
        "7ce47916-12e3-416d-98bd-80d4cdf159ca": "370a9587-74a1-4959-b406-fa4500002c3f",
        "fb8e59b7-6135-470d-b206-abb0ab766c6e": "370a9587-74a1-4959-b406-fa4500002c3e",
        "8d770705-53e8-491c-ace5-a0a97d90e996": "370a9587-74a1-4959-b406-fa4500002c3d",
        "16c7b6d6-d2fa-4e84-93a9-0c6cad7b5401": "370a9587-74a1-4959-b406-fa4500002c3c"
      },
      "layers": {
        "370a9587-74a1-4959-b406-fa4500002c40": {
          "class": "com.simiacryptus.mindseye.layers.java.SqActivationLayer",
          "id": "370a9587-74a1-4959-b406-fa4500002c40",
          "isFrozen": true,
          "name": "SqActivationLayer/370a9587-74a1-4959-b406-fa4500002c40"
        },
        "370a9587-74a1-4959-b406-fa4500002c3f": {
          "class": "com.simiacryptus.mindseye.layers.java.AvgReducerLayer",
          "id": "370a9587-74a1-4959-b406-fa4500002c3f",
          "isFrozen": false,
          "name": "AvgReducerLayer/370a9587-74a1-4959-b406-fa4500002c3f"
        },
        "370a9587-74a1-4959-b406-fa4500002c3e": {
          "class": "com.simiacryptus.mindseye.layers.java.AvgMetaLayer",
          "id": "370a9587-74a1-4959-b406-fa4500002c3e",
          "isFrozen": false,
          "name": "AvgMetaLayer/370a9587-74a1-4959-b406-fa4500002c3e"
        },
        "370a9587-74a1-4959-b406-fa4500002c3d": {
          "class": "com.simiacryptus.mindseye.layers.java.NthPowerActivationLayer",
          "id": "370a9587-74a1-4959-b406-fa4500002c3d",
          "isFrozen": false,
          "name": "NthPowerActivationLayer/370a9587-74a1-4959-b406-fa4500002c3d",
          "power": -0.5
        },
        "370a9587-74a1-4959-b406-fa4500002c3c": {
          "class": "com.simiacryptus.mindseye.layers.java.ProductInputsLayer",
          "id": "370a9587-74a1-4959-b406-fa4500002c3c",
          "isFrozen": false,
          "name": "ProductInputsLayer/370a9587-74a1-4959-b406-fa4500002c3c"
        }
      },
      "links": {
        "8e6c9675-20c2-423a-8bf8-98c2d320391c": [
          "cc18c7d2-9920-4c90-adf1-52af0f9b6e59"
        ],
        "7ce47916-12e3-416d-98bd-80d4cdf159ca": [
          "8e6c9675-20c2-423a-8bf8-98c2d320391c"
        ],
        "fb8e59b7-6135-470d-b206-abb0ab766c6e": [
          "7ce47916-12e3-416d-98bd-80d4cdf159ca"
        ],
        "8d770705-53e8-491c-ace5-a0a97d90e996": [
          "fb8e59b7-6135-470d-b206-abb0ab766c6e"
        ],
        "16c7b6d6-d2fa-4e84-93a9-0c6cad7b5401": [
          "cc18c7d2-9920-4c90-adf1-52af0f9b6e59",
          "8d770705-53e8-491c-ace5-a0a97d90e996"
        ]
      },
      "labels": {},
      "head": "16c7b6d6-d2fa-4e84-93a9-0c6cad7b5401"
    }
```



### Network Diagram
Code from [LayerTestBase.java:94](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L94) executed in 0.20 seconds: 
```java
    return Graphviz.fromGraph(toGraph((DAGNetwork) layer))
      .height(400).width(600).render(Format.PNG).toImage();
```

Returns: 

![Result](etc/test.1.png)



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
    [[ 0.264, -1.496, -0.692 ]]
    --------------------
    Output: 
    [ 0.27392338389709414, -1.5522325087502, -0.7180112941545042 ]
```



### Batch Execution
Code from [LayerTestBase.java:138](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L138) executed in 0.00 seconds: 
```java
    BatchingTester batchingTester = getBatchingTester();
    return batchingTester==null?null:batchingTester.test(layer, inputPrototype);
```

### Differential Validation
Code from [LayerTestBase.java:144](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.00 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [ 0.264, -1.496, -0.692 ]
    Inputs Statistics: {meanExponent=-0.1877861283816562, negative=2, min=-0.692, max=-0.692, mean=-0.6413333333333333, count=3.0, positive=1, stdDev=0.7194096345075053, zeros=0}
    Output: [ 0.27392338389709414, -1.5522325087502, -0.7180112941545042 ]
    Outputs Statistics: {meanExponent=-0.1717609471888691, negative=2, min=-0.7180112941545042, max=-0.7180112941545042, mean=-0.6654401396692032, count=3.0, positive=1, stdDev=0.7464512177744984, zeros=0}
    Feedback for input 0
    Inputs Values: [ 0.264, -1.496, -0.692 ]
    Value Statistics: {meanExponent=-0.1877861283816562, negative=2, min=-0.692, max=-0.692, mean=-0.6413333333333333, count=3.0, positive=1, stdDev=0.7194096345075053, zeros=0}
    Implemented Feedback: [ [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ] ]
    Implemented Statistics: {meanExponent=NaN, negative=0, min=0.0, max=0.0, mean=0.0, count=9.0, positive=0, stdDev=0.0, zeros=9}
    Measured: [ [ 1.0375885753671765, 0.0, 0.0 ], [ 0.0, 1.0375885753677316, 0.0 ], [ 0.0, 0.0, 1.0375885753677316 ] ]
    Measured Statistics: {meanExponent=0.016025181192689083, negative=0, min=1.0375885753677316, max=1.0375885753677316, mean=0.34586285845584885, count=9.0, positive=3, stdDev=0.4891239451493875, zeros=6}
    Feedback Error: [ [ 1.0375885753671765, 0.0, 0.0 ], [ 0.0, 1.0375885753677316, 0.0 ], [ 0.0, 0.0, 1.0375885753677316 ] ]
    Error Statistics: {meanExponent=0.016025181192689083, negative=0, min=1.0375885753677316, max=1.0375885753677316, mean=0.34586285845584885, count=9.0, positive=3, stdDev=0.4891239451493875, zeros=6}
    
```

Returns: 

```
    java.lang.AssertionError: ToleranceStatistics{absoluteTol=3.4586e-01 +- 4.8912e-01 [0.0000e+00 - 1.0376e+00] (9#), relativeTol=1.0000e+00 +- 0.0000e+00 [1.0000e+00 - 1.0000e+00] (3#)}
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



