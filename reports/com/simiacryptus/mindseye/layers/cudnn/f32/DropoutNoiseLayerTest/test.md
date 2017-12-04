# DropoutNoiseLayer
## DropoutNoiseLayerTest
### Json Serialization
Code from [LayerTestBase.java:83](../../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L83) executed in 0.00 seconds: 
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
      "class": "com.simiacryptus.mindseye.layers.cudnn.f32.DropoutNoiseLayer",
      "id": "a864e734-2f23-44db-97c1-5040000003aa",
      "isFrozen": false,
      "name": "DropoutNoiseLayer/a864e734-2f23-44db-97c1-5040000003aa",
      "inputs": [
        "c46bacf6-be81-4b68-aab4-458602aa9980"
      ],
      "nodes": {
        "bce7d604-b5da-4109-875e-2ae14658b07e": "a864e734-2f23-44db-97c1-5040000003ac",
        "a63a481e-43aa-4a13-8e21-114478b1b61c": "a864e734-2f23-44db-97c1-5040000003ab"
      },
      "layers": {
        "a864e734-2f23-44db-97c1-5040000003ac": {
          "class": "com.simiacryptus.mindseye.layers.java.BinaryNoiseLayer",
          "id": "a864e734-2f23-44db-97c1-5040000003ac",
          "isFrozen": false,
          "name": "mask",
          "value": 0.5
        },
        "a864e734-2f23-44db-97c1-5040000003ab": {
          "class": "com.simiacryptus.mindseye.layers.cudnn.f32.ProductInputsLayer",
          "id": "a864e734-2f23-44db-97c1-5040000003ab",
          "isFrozen": false,
          "name": "ProductInputsLayer/a864e734-2f23-44db-97c1-5040000003ab"
        }
      },
      "links": {
        "bce7d604-b5da-4109-875e-2ae14658b07e": [
          "c46bacf6-be81-4b68-aab4-458602aa9980"
        ],
        "a63a481e-43aa-4a13-8e21-114478b1b61c": [
          "bce7d604-b5da-4109-875e-2ae14658b07e",
          "c46bacf6-be81-4b68-aab4-458602aa9980"
        ]
      },
      "labels": {},
      "head": "a63a481e-43aa-4a13-8e21-114478b1b61c"
    }
```



### Network Diagram
Code from [LayerTestBase.java:94](../../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L94) executed in 0.15 seconds: 
```java
    return Graphviz.fromGraph(toGraph((DAGNetwork) layer))
      .height(400).width(600).render(Format.PNG).toImage();
```

Returns: 

![Result](etc/test.1.png)



### Example Input/Output Pair
Code from [LayerTestBase.java:120](../../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L120) executed in 0.00 seconds: 
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
    	[ [ 1.752, -1.264 ], [ -1.728, -0.548 ], [ -1.404, -0.128 ] ],
    	[ [ -1.812, -0.188 ], [ -0.868, -1.684 ], [ 1.484, 0.504 ] ],
    	[ [ -1.304, -1.952 ], [ -0.212, 0.568 ], [ -0.3, 1.328 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 1.7519999742507935, 0.0 ], [ 0.0, -0.5479999780654907 ], [ 0.0, -0.12800000607967377 ] ],
    	[ [ -1.812000036239624, -0.18799999356269836 ], [ 0.0, -1.684000015258789 ], [ 1.4839999675750732, 0.5040000081062317 ] ],
    	[ [ 0.0, -1.9520000219345093 ], [ -0.21199999749660492, 0.0 ], [ 0.0, 1.3279999494552612 ] ]
    ]
```



### Batch Execution
Code from [LayerTestBase.java:138](../../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L138) executed in 0.00 seconds: 
```java
    BatchingTester batchingTester = getBatchingTester();
    return batchingTester==null?null:batchingTester.test(layer, inputPrototype);
```

Returns: 

```
    java.lang.AssertionError: ToleranceStatistics{absoluteTol=5.8818e-01 +- 8.0659e-01 [0.0000e+00 - 2.4392e+00] (180#), relativeTol=5.8451e-01 +- 4.9281e-01 [0.0000e+00 - 1.0000e+00] (142#)}
    	at com.simiacryptus.mindseye.layers.BatchingTester.test(BatchingTester.java:80)
    	at com.simiacryptus.mindseye.layers.LayerTestBase.lambda$test$15(LayerTestBase.java:140)
    	at com.simiacryptus.util.io.MarkdownNotebookOutput.lambda$null$1(MarkdownNotebookOutput.java:136)
    	at com.simiacryptus.util.lang.TimedResult.time(TimedResult.java:59)
    	at com.simiacryptus.util.io.MarkdownNotebookOutput.lambda$code$2(MarkdownNotebookOutput.java:136)
    	at com.simiacryptus.util.test.SysOutInterceptor.withOutput(SysOutInterceptor.java:83)
    	at com.simiacryptus.util.io.MarkdownNotebookOutput.code(MarkdownNotebookOutput.java:134)
    	at com.simiacryptus.util.io.NotebookOutput.code(NotebookOutput.java:133)
    	at com.simiacryptus.mindseye.layers.LayerTestBase.test(LayerTestBase.java:138)
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



