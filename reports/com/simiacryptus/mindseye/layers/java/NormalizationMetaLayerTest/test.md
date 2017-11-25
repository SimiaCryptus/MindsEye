### Json Serialization
Code from [LayerTestBase.java:74](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L74) executed in 0.00 seconds: 
```java
  
```

Returns: 

```
    {
      "class": "com.simiacryptus.mindseye.layers.java.NormalizationMetaLayer",
      "id": "bdd6bbba-380b-47fe-a761-c2410002dcd1",
      "isFrozen": false,
      "name": "NormalizationMetaLayer/bdd6bbba-380b-47fe-a761-c2410002dcd1",
      "inputs": [
        "fd76e7be-5e8a-4379-a32e-41363ac32867"
      ],
      "nodes": {
        "99b44962-2990-4dd0-b976-37b402ba7ee3": "bdd6bbba-380b-47fe-a761-c2410002dcd6",
        "2be2f535-25e1-4a2f-99e4-1d94ebb0fe93": "bdd6bbba-380b-47fe-a761-c2410002dcd5",
        "8d8c65e7-1b38-4516-a950-1582d7581182": "bdd6bbba-380b-47fe-a761-c2410002dcd4",
        "5d6fd9b0-8731-43eb-8b78-ff6a932562dd": "bdd6bbba-380b-47fe-a761-c2410002dcd3",
        "fb2d1e84-91cd-4288-ba50-aec967295d4e": "bdd6bbba-380b-47fe-a761-c2410002dcd2"
      },
      "layers": {
        "bdd6bbba-380b-47fe-a761-c2410002dcd6": {
          "class": "com.simiacryptus.mindseye.layers.java.SqActivationLayer",
          "id": "bdd6bbba-380b-47fe-a761-c2410002dcd6",
          "isFrozen": true,
          "name": "SqActivationLayer/bdd6bbba-380b-47fe-a761-c2410002dcd6"
        },
        "bdd6bbba-380b-47fe-a761-c2410002dcd5": {
          "class": "com.simiacryptus.mindseye.layers.java.AvgReducerLayer",
          "id": "bdd6bbba-380b-47fe-a761-c2410002dcd5",
          "isFrozen": false,
          "name": "AvgReducerLayer/bdd6bbba-380b-47fe-a761-c2410002dcd5"
        },
        "bdd6bbba-380b-47fe-a761-c2410002dcd4": {
          "class": "com.simiacryptus.mindseye.layers.java.AvgMetaLayer",
          "id": "bdd6bbba-380b-47fe-a761-c2410002dcd4",
          "isFrozen": false,
          "name": "AvgMetaLayer/bdd6bbba-380b-47fe-a761-c2410002dcd4"
        },
        "bdd6bbba-380b-47fe-a761-c2410002dcd3": {
          "class": "com.simiacryptus.mindseye.layers.java.NthPowerActivationLayer",
          "id": "bdd6bbba-380b-47fe-a761-c2410002dcd3",
          "isFrozen": false,
          "name": "NthPowerActivationLayer/bdd6bbba-380b-47fe-a761-c2410002dcd3",
          "power": -0.5
        },
        "bdd6bbba-380b-47fe-a761-c2410002dcd2": {
          "class": "com.simiacryptus.mindseye.layers.java.ProductInputsLayer",
          "id": "bdd6bbba-380b-47fe-a761-c2410002dcd2",
          "isFrozen": false,
          "name": "ProductInputsLayer/bdd6bbba-380b-47fe-a761-c2410002dcd2"
        }
      },
      "links": {
        "99b44962-2990-4dd0-b976-37b402ba7ee3": [
          "fd76e7be-5e8a-4379-a32e-41363ac32867"
        ],
        "2be2f535-25e1-4a2f-99e4-1d94ebb0fe93": [
          "99b44962-2990-4dd0-b976-37b402ba7ee3"
        ],
        "8d8c65e7-1b38-4516-a950-1582d7581182": [
          "2be2f535-25e1-4a2f-99e4-1d94ebb0fe93"
        ],
        "5d6fd9b0-8731-43eb-8b78-ff6a932562dd": [
          "8d8c65e7-1b38-4516-a950-1582d7581182"
        ],
        "fb2d1e84-91cd-4288-ba50-aec967295d4e": [
          "fd76e7be-5e8a-4379-a32e-41363ac32867",
          "5d6fd9b0-8731-43eb-8b78-ff6a932562dd"
        ]
      },
      "labels": {},
      "head": "fb2d1e84-91cd-4288-ba50-aec967295d4e"
    }
```



### Network Diagram
Code from [LayerTestBase.java:85](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L85) executed in 0.32 seconds: 
```java
    log.h3("Network Diagram");
    log.code(()->{
      return Graphviz.fromGraph(toGraph((DAGNetwork) layer))
        .height(400).width(600).render(Format.PNG).toImage();
    });
```

Returns: 

![Result](etc/test.1.png)



### Differential Validation
Code from [LayerTestBase.java:98](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L98) executed in 0.00 seconds: 
```java
  
```
Logging: 
```
    Component: NormalizationMetaLayer/bdd6bbba-380b-47fe-a761-c2410002dcd1
    Inputs: [[ 0.8366347174703086,-1.2899417325016493,-1.2092419003794062 ]]
    output=[ 0.7408225466765789,-1.1422164289640349,-1.0707584152087137 ]
    measured/actual: [ [ 0.885479088164054,0.0,0.0 ],[ 0.0,0.885479087830987,0.0 ],[ 0.0,0.0,0.8854790880530317 ] ]
    implemented/expected: [ [ 0.0,0.0,0.0 ],[ 0.0,0.0,0.0 ],[ 0.0,0.0,0.0 ] ]
    error: [ [ 0.885479088164054,0.0,0.0 ],[ 0.0,0.885479087830987,0.0 ],[ 0.0,0.0,0.8854790880530317 ] ]
    
```

Returns: 

```
    java.lang.AssertionError
    	at com.simiacryptus.mindseye.layers.DerivativeTester.testFeedback(DerivativeTester.java:220)
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
    	at com.simiacryptus.mindseye.layers.LayerTestBase.lambda$test$7(LayerTestBase.java:99)
    	at com.simiacryptus.util.io.NotebookOutput.lambda$code$1(NotebookOutput.java:142)
    	at com.simiacryptus.util.io.MarkdownNotebookOutput.lambda$null$1(MarkdownNotebookOutput.java:136)
    	at com.simiacryptus.util.lang.TimedResult.time(TimedResult.java:59)
    	at com.simiacryptus.util.io.MarkdownNotebookOutput.lambda$code$2(MarkdownNotebookOutput.java:136)
    	at com.simiacryptus.util.test.SysOutInterceptor.withOutput(SysOutInterceptor.java:77)
    	at com.simiacryptus.util.io.MarkdownNotebookOutput.code(MarkdownNotebookOutput.java:134)
    	at com.simiacryptus.util.io.NotebookOutput.code(NotebookOutput.java:141)
    	at com.simiacryptus.mindseye.layers.LayerTestBase.test(LayerTestBase.java:98)
    	at com.simiacryptus.mindseye.layers.LayerTestBase.test(LayerTestBase.java:66)
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



