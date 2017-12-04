# DropoutNoiseLayer
## DropoutNoiseLayerTest
### Json Serialization
Code from [LayerTestBase.java:83](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L83) executed in 0.00 seconds: 
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
      "id": "370a9587-74a1-4959-b406-fa45000003aa",
      "isFrozen": false,
      "name": "DropoutNoiseLayer/370a9587-74a1-4959-b406-fa45000003aa",
      "inputs": [
        "6f4108a9-eba1-45e3-99fa-e681168a0649"
      ],
      "nodes": {
        "9e0da1cf-74b8-4a68-b456-373e48864a7f": "370a9587-74a1-4959-b406-fa45000003ac",
        "e40d9018-35e5-4ca3-886a-38388f9808f6": "370a9587-74a1-4959-b406-fa45000003ab"
      },
      "layers": {
        "370a9587-74a1-4959-b406-fa45000003ac": {
          "class": "com.simiacryptus.mindseye.layers.java.BinaryNoiseLayer",
          "id": "370a9587-74a1-4959-b406-fa45000003ac",
          "isFrozen": false,
          "name": "mask",
          "value": 0.5
        },
        "370a9587-74a1-4959-b406-fa45000003ab": {
          "class": "com.simiacryptus.mindseye.layers.cudnn.f32.ProductInputsLayer",
          "id": "370a9587-74a1-4959-b406-fa45000003ab",
          "isFrozen": false,
          "name": "ProductInputsLayer/370a9587-74a1-4959-b406-fa45000003ab"
        }
      },
      "links": {
        "9e0da1cf-74b8-4a68-b456-373e48864a7f": [
          "6f4108a9-eba1-45e3-99fa-e681168a0649"
        ],
        "e40d9018-35e5-4ca3-886a-38388f9808f6": [
          "9e0da1cf-74b8-4a68-b456-373e48864a7f",
          "6f4108a9-eba1-45e3-99fa-e681168a0649"
        ]
      },
      "labels": {},
      "head": "e40d9018-35e5-4ca3-886a-38388f9808f6"
    }
```



### Network Diagram
Code from [LayerTestBase.java:94](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L94) executed in 0.14 seconds: 
```java
    return Graphviz.fromGraph(toGraph((DAGNetwork) layer))
      .height(400).width(600).render(Format.PNG).toImage();
```

Returns: 

![Result](etc/test.1.png)



### Example Input/Output Pair
Code from [LayerTestBase.java:120](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L120) executed in 0.00 seconds: 
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
    	[ [ -1.08, -1.168 ], [ -1.204, -1.432 ], [ -1.28, -0.976 ] ],
    	[ [ 1.668, 0.88 ], [ 1.316, 0.712 ], [ -0.916, -1.624 ] ],
    	[ [ 1.476, -1.176 ], [ -1.124, -0.232 ], [ 1.128, -0.384 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ -1.0800000429153442, 0.0 ], [ -1.2039999961853027, 0.0 ], [ 0.0, 0.0 ] ],
    	[ [ 0.0, 0.8799999952316284 ], [ 0.0, 0.7120000123977661 ], [ 0.0, -1.6239999532699585 ] ],
    	[ [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 1.128000020980835, 0.0 ] ]
    ]
```



### Batch Execution
Code from [LayerTestBase.java:138](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L138) executed in 0.00 seconds: 
```java
    BatchingTester batchingTester = getBatchingTester();
    return batchingTester==null?null:batchingTester.test(layer, inputPrototype);
```

Returns: 

```
    java.lang.AssertionError: ToleranceStatistics{absoluteTol=5.7928e-01 +- 7.9971e-01 [0.0000e+00 - 2.4866e+00] (180#), relativeTol=7.4545e-01 +- 4.3561e-01 [0.0000e+00 - 1.0000e+00] (110#)}
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



