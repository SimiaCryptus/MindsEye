### Json Serialization
Code from [LayerTestBase.java:57](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L57) executed in 0.00 seconds: 
```java
    NNLayer layer = getLayer();
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
      "id": "9d13704a-9a5a-4ecb-a687-5c7c0002dd45",
      "isFrozen": false,
      "name": "NormalizationMetaLayer/9d13704a-9a5a-4ecb-a687-5c7c0002dd45",
      "inputs": [
        "0e284b54-a297-47ec-9dd9-a8014f65410f"
      ],
      "nodes": {
        "2ce60ca2-bc41-4b7f-b751-a95b6aaf7e15": "9d13704a-9a5a-4ecb-a687-5c7c0002dd4a",
        "5186c0f6-c5b4-45e8-91c2-efa15fef24bd": "9d13704a-9a5a-4ecb-a687-5c7c0002dd49",
        "b9b57f8e-813b-4402-b73c-7f43f4957c69": "9d13704a-9a5a-4ecb-a687-5c7c0002dd48",
        "0e7fb53e-2db1-4a11-9ada-e1dfc8a1ef7b": "9d13704a-9a5a-4ecb-a687-5c7c0002dd47",
        "e3fcba5e-419c-43f1-8cca-4622b2639a1f": "9d13704a-9a5a-4ecb-a687-5c7c0002dd46"
      },
      "layers": {
        "9d13704a-9a5a-4ecb-a687-5c7c0002dd4a": {
          "class": "com.simiacryptus.mindseye.layers.java.SqActivationLayer",
          "id": "9d13704a-9a5a-4ecb-a687-5c7c0002dd4a",
          "isFrozen": true,
          "name": "SqActivationLayer/9d13704a-9a5a-4ecb-a687-5c7c0002dd4a"
        },
        "9d13704a-9a5a-4ecb-a687-5c7c0002dd49": {
          "class": "com.simiacryptus.mindseye.layers.java.AvgReducerLayer",
          "id": "9d13704a-9a5a-4ecb-a687-5c7c0002dd49",
          "isFrozen": false,
          "name": "AvgReducerLayer/9d13704a-9a5a-4ecb-a687-5c7c0002dd49"
        },
        "9d13704a-9a5a-4ecb-a687-5c7c0002dd48": {
          "class": "com.simiacryptus.mindseye.layers.java.AvgMetaLayer",
          "id": "9d13704a-9a5a-4ecb-a687-5c7c0002dd48",
          "isFrozen": false,
          "name": "AvgMetaLayer/9d13704a-9a5a-4ecb-a687-5c7c0002dd48"
        },
        "9d13704a-9a5a-4ecb-a687-5c7c0002dd47": {
          "class": "com.simiacryptus.mindseye.layers.java.NthPowerActivationLayer",
          "id": "9d13704a-9a5a-4ecb-a687-5c7c0002dd47",
          "isFrozen": false,
          "name": "NthPowerActivationLayer/9d13704a-9a5a-4ecb-a687-5c7c0002dd47",
          "power": -0.5
        },
        "9d13704a-9a5a-4ecb-a687-5c7c0002dd46": {
          "class": "com.simiacryptus.mindseye.layers.java.ProductInputsLayer",
          "id": "9d13704a-9a5a-4ecb-a687-5c7c0002dd46",
          "isFrozen": false,
          "name": "ProductInputsLayer/9d13704a-9a5a-4ecb-a687-5c7c0002dd46"
        }
      },
      "links": {
        "2ce60ca2-bc41-4b7f-b751-a95b6aaf7e15": [
          "0e284b54-a297-47ec-9dd9-a8014f65410f"
        ],
        "5186c0f6-c5b4-45e8-91c2-efa15fef24bd": [
          "2ce60ca2-bc41-4b7f-b751-a95b6aaf7e15"
        ],
        "b9b57f8e-813b-4402-b73c-7f43f4957c69": [
          "5186c0f6-c5b4-45e8-91c2-efa15fef24bd"
        ],
        "0e7fb53e-2db1-4a11-9ada-e1dfc8a1ef7b": [
          "b9b57f8e-813b-4402-b73c-7f43f4957c69"
        ],
        "e3fcba5e-419c-43f1-8cca-4622b2639a1f": [
          "0e284b54-a297-47ec-9dd9-a8014f65410f",
          "0e7fb53e-2db1-4a11-9ada-e1dfc8a1ef7b"
        ]
      },
      "labels": {},
      "head": "e3fcba5e-419c-43f1-8cca-4622b2639a1f"
    }
```



### Differential Validation
Code from [LayerTestBase.java:74](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L74) executed in 0.00 seconds: 
```java
    getDerivativeTester().test(getLayer(), outputPrototype, inputPrototype);
```
Logging: 
```
    Component: NormalizationMetaLayer/9d13704a-9a5a-4ecb-a687-5c7c0002dd51
    Inputs: [[ 0.4913341472607682,0.04028971881659693,0.16934615288818022 ]]
    output=[ 1.6326168133034669,0.13387563781178577,0.5627074324365375 ]
    measured/actual: [ [ 3.3228238383031794,0.0,0.0 ],[ 0.0,3.3228238299765067,0.0 ],[ 0.0,0.0,3.322823827200949 ] ]
    implemented/expected: [ [ 0.0,0.0,0.0 ],[ 0.0,0.0,0.0 ],[ 0.0,0.0,0.0 ] ]
    error: [ [ 3.3228238383031794,0.0,0.0 ],[ 0.0,3.3228238299765067,0.0 ],[ 0.0,0.0,3.322823827200949 ] ]
    
```

Returns: 

```
    java.lang.AssertionError
    	at com.simiacryptus.mindseye.layers.DerivativeTester.testFeedback(DerivativeTester.java:219)
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
    	at com.simiacryptus.mindseye.layers.LayerTestBase.lambda$test$6(LayerTestBase.java:75)
    	at com.simiacryptus.util.io.NotebookOutput.lambda$code$1(NotebookOutput.java:142)
    	at com.simiacryptus.util.io.MarkdownNotebookOutput.lambda$null$1(MarkdownNotebookOutput.java:136)
    	at com.simiacryptus.util.lang.TimedResult.time(TimedResult.java:59)
    	at com.simiacryptus.util.io.MarkdownNotebookOutput.lambda$code$2(MarkdownNotebookOutput.java:136)
    	at com.simiacryptus.util.test.SysOutInterceptor.withOutput(SysOutInterceptor.java:77)
    	at com.simiacryptus.util.io.MarkdownNotebookOutput.code(MarkdownNotebookOutput.java:134)
    	at com.simiacryptus.util.io.NotebookOutput.code(NotebookOutput.java:141)
    	at com.simiacryptus.mindseye.layers.LayerTestBase.test(LayerTestBase.java:74)
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



