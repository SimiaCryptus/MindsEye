# StdDevMetaLayer
## StdDevMetaLayerTest
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
      "class": "com.simiacryptus.mindseye.layers.java.StdDevMetaLayer",
      "id": "f4569375-56fe-4e46-925c-95f400000a85",
      "isFrozen": false,
      "name": "StdDevMetaLayer/f4569375-56fe-4e46-925c-95f400000a85",
      "inputs": [
        "7a77610b-bc01-4e2f-9ff7-7251358d29a8"
      ],
      "nodes": {
        "8fc4dd02-0217-4803-b03b-62bd4ec26e7e": "f4569375-56fe-4e46-925c-95f400000a89",
        "cfebd4b1-d149-4015-804a-bffb8eb93152": "f4569375-56fe-4e46-925c-95f400000a88",
        "a7142715-7f58-43b2-a6db-4893a5750c9b": "f4569375-56fe-4e46-925c-95f400000a8c",
        "d9ebc7a9-181a-4718-b046-0fd44d627c0f": "f4569375-56fe-4e46-925c-95f400000a8b",
        "8b2ceec8-20a1-4829-b85f-10c7f760180c": "f4569375-56fe-4e46-925c-95f400000a8a",
        "a67cbea4-15d6-4083-89d3-25adffc935aa": "f4569375-56fe-4e46-925c-95f400000a87",
        "6b40b660-efb8-474a-ba51-86949917ad8b": "f4569375-56fe-4e46-925c-95f400000a86"
      },
      "layers": {
        "f4569375-56fe-4e46-925c-95f400000a89": {
          "class": "com.simiacryptus.mindseye.layers.java.SqActivationLayer",
          "id": "f4569375-56fe-4e46-925c-95f400000a89",
          "isFrozen": true,
          "name": "SqActivationLayer/f4569375-56fe-4e46-925c-95f400000a89"
        },
        "f4569375-56fe-4e46-925c-95f400000a88": {
          "class": "com.simiacryptus.mindseye.layers.java.AvgMetaLayer",
          "id": "f4569375-56fe-4e46-925c-95f400000a88",
          "isFrozen": false,
          "name": "AvgMetaLayer/f4569375-56fe-4e46-925c-95f400000a88"
        },
        "f4569375-56fe-4e46-925c-95f400000a8c": {
          "class": "com.simiacryptus.mindseye.layers.java.AvgMetaLayer",
          "id": "f4569375-56fe-4e46-925c-95f400000a8c",
          "isFrozen": false,
          "name": "AvgMetaLayer/f4569375-56fe-4e46-925c-95f400000a8c"
        },
        "f4569375-56fe-4e46-925c-95f400000a8b": {
          "class": "com.simiacryptus.mindseye.layers.java.SqActivationLayer",
          "id": "f4569375-56fe-4e46-925c-95f400000a8b",
          "isFrozen": true,
          "name": "SqActivationLayer/f4569375-56fe-4e46-925c-95f400000a8b"
        },
        "f4569375-56fe-4e46-925c-95f400000a8a": {
          "class": "com.simiacryptus.mindseye.layers.java.LinearActivationLayer",
          "id": "f4569375-56fe-4e46-925c-95f400000a8a",
          "isFrozen": false,
          "name": "LinearActivationLayer/f4569375-56fe-4e46-925c-95f400000a8a",
          "weights": {
            "dimensions": [
              2
            ],
            "data": [
              -1.0,
              0.0
            ]
          }
        },
        "f4569375-56fe-4e46-925c-95f400000a87": {
          "class": "com.simiacryptus.mindseye.layers.java.SumInputsLayer",
          "id": "f4569375-56fe-4e46-925c-95f400000a87",
          "isFrozen": false,
          "name": "SumInputsLayer/f4569375-56fe-4e46-925c-95f400000a87"
        },
        "f4569375-56fe-4e46-925c-95f400000a86": {
          "class": "com.simiacryptus.mindseye.layers.java.NthPowerActivationLayer",
          "id": "f4569375-56fe-4e46-925c-95f400000a86",
          "isFrozen": false,
          "name": "NthPowerActivationLayer/f4569375-56fe-4e46-925c-95f400000a86",
          "power": 0.5
        }
      },
      "links": {
        "8fc4dd02-0217-4803-b03b-62bd4ec26e7e": [
          "7a77610b-bc01-4e2f-9ff7-7251358d29a8"
        ],
        "cfebd4b1-d149-4015-804a-bffb8eb93152": [
          "8fc4dd02-0217-4803-b03b-62bd4ec26e7e"
        ],
        "a7142715-7f58-43b2-a6db-4893a5750c9b": [
          "7a77610b-bc01-4e2f-9ff7-7251358d29a8"
        ],
        "d9ebc7a9-181a-4718-b046-0fd44d627c0f": [
          "a7142715-7f58-43b2-a6db-4893a5750c9b"
        ],
        "8b2ceec8-20a1-4829-b85f-10c7f760180c": [
          "d9ebc7a9-181a-4718-b046-0fd44d627c0f"
        ],
        "a67cbea4-15d6-4083-89d3-25adffc935aa": [
          "cfebd4b1-d149-4015-804a-bffb8eb93152",
          "8b2ceec8-20a1-4829-b85f-10c7f760180c"
        ],
        "6b40b660-efb8-474a-ba51-86949917ad8b": [
          "a67cbea4-15d6-4083-89d3-25adffc935aa"
        ]
      },
      "labels": {},
      "head": "6b40b660-efb8-474a-ba51-86949917ad8b"
    }
```



### Network Diagram
Code from [LayerTestBase.java:95](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L95) executed in 0.20 seconds: 
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
    [[ 0.268, -0.304, 0.388 ]]
    --------------------
    Output: 
    [ 0.0, 0.0, 0.0 ]
```



### Differential Validation
Code from [LayerTestBase.java:139](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L139) executed in 0.00 seconds: 
```java
    getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Feedback for input 0
    Inputs: [ 0.268, -0.304, 0.388 ]
    Output: [ 0.0, 0.0, 0.0 ]
    Measured: [ [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ] ]
    Implemented: [ [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ] ]
    Error: [ [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ] ]
    Finite-Difference Derivative Accuracy:
    absoluteTol: 0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (9#)
    relativeTol: 0.0000e+00 +- 0.0000e+00 [Infinity - -Infinity] (0#)
    
```

Returns: 

```
    java.lang.RuntimeException: java.lang.RuntimeException: java.util.concurrent.ExecutionException: java.lang.RuntimeException: Frozen component did not pass input backwards
    	at com.simiacryptus.util.lang.TimedResult.time(TimedResult.java:61)
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
    	at org.junit.runners.Sui
```
...[skipping 776 bytes](etc/1.txt)...
```
    .java:242)
    	at com.intellij.rt.execution.junit.JUnitStarter.main(JUnitStarter.java:70)
    Caused by: java.lang.RuntimeException: java.util.concurrent.ExecutionException: java.lang.RuntimeException: Frozen component did not pass input backwards
    	at com.simiacryptus.mindseye.layers.cudnn.GpuController.lambda$run$8(GpuController.java:215)
    	at com.simiacryptus.util.lang.StaticResourcePool.apply(StaticResourcePool.java:88)
    	at com.simiacryptus.mindseye.layers.cudnn.GpuController.run(GpuController.java:211)
    	at com.simiacryptus.mindseye.layers.DerivativeTester.testFrozen(DerivativeTester.java:100)
    	at com.simiacryptus.mindseye.layers.DerivativeTester.test(DerivativeTester.java:91)
    	at com.simiacryptus.mindseye.layers.LayerTestBase.lambda$test$15(LayerTestBase.java:140)
    	at com.simiacryptus.util.io.NotebookOutput.lambda$code$1(NotebookOutput.java:157)
    	at com.simiacryptus.util.io.MarkdownNotebookOutput.lambda$null$1(MarkdownNotebookOutput.java:136)
    	at com.simiacryptus.util.lang.TimedResult.time(TimedResult.java:59)
    	... 35 more
    Caused by: java.util.concurrent.ExecutionException: java.lang.RuntimeException: Frozen component did not pass input backwards
    	at java.util.concurrent.FutureTask.report(FutureTask.java:122)
    	at java.util.concurrent.FutureTask.get(FutureTask.java:192)
    	at com.simiacryptus.mindseye.layers.cudnn.GpuController.lambda$run$8(GpuController.java:213)
    	... 43 more
    Caused by: java.lang.RuntimeException: Frozen component did not pass input backwards
    	at com.simiacryptus.mindseye.layers.DerivativeTester.lambda$testFrozen$11(DerivativeTester.java:118)
    	at com.simiacryptus.mindseye.layers.cudnn.GpuController.lambda$null$7(GpuController.java:213)
    	at java.util.concurrent.Executors$RunnableAdapter.call(Executors.java:511)
    	at java.util.concurrent.FutureTask.run(FutureTask.java:266)
    	at java.util.concurrent.ThreadPoolExecutor.runWorker(ThreadPoolExecutor.java:1149)
    	at java.util.concurrent.ThreadPoolExecutor$Worker.run(ThreadPoolExecutor.java:624)
    	at java.lang.Thread.run(Thread.java:748)
    
```



