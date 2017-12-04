# StdDevMetaLayer
## StdDevMetaLayerTest
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
      "class": "com.simiacryptus.mindseye.layers.java.StdDevMetaLayer",
      "id": "370a9587-74a1-4959-b406-fa4500002ca2",
      "isFrozen": false,
      "name": "StdDevMetaLayer/370a9587-74a1-4959-b406-fa4500002ca2",
      "inputs": [
        "567ce1f0-b842-477a-bf51-c75625f62a13"
      ],
      "nodes": {
        "372ac95d-1b97-4458-bb00-342c51c00807": "370a9587-74a1-4959-b406-fa4500002ca6",
        "760a2a2c-4c44-4889-87bc-ba6bd642b33f": "370a9587-74a1-4959-b406-fa4500002ca5",
        "e325ff4d-2510-4830-83a1-e1379925fa10": "370a9587-74a1-4959-b406-fa4500002ca9",
        "8be5a43c-ed9f-4aa7-9e2c-0f7d605c3996": "370a9587-74a1-4959-b406-fa4500002ca8",
        "07b3a390-d8f7-42ca-b95d-5e630fe5c997": "370a9587-74a1-4959-b406-fa4500002ca7",
        "37e44896-a012-486a-8af2-c16e9a58cee5": "370a9587-74a1-4959-b406-fa4500002ca4",
        "71d93eb3-92a5-449d-ae31-e9e450154828": "370a9587-74a1-4959-b406-fa4500002ca3"
      },
      "layers": {
        "370a9587-74a1-4959-b406-fa4500002ca6": {
          "class": "com.simiacryptus.mindseye.layers.java.SqActivationLayer",
          "id": "370a9587-74a1-4959-b406-fa4500002ca6",
          "isFrozen": true,
          "name": "SqActivationLayer/370a9587-74a1-4959-b406-fa4500002ca6"
        },
        "370a9587-74a1-4959-b406-fa4500002ca5": {
          "class": "com.simiacryptus.mindseye.layers.java.AvgMetaLayer",
          "id": "370a9587-74a1-4959-b406-fa4500002ca5",
          "isFrozen": false,
          "name": "AvgMetaLayer/370a9587-74a1-4959-b406-fa4500002ca5"
        },
        "370a9587-74a1-4959-b406-fa4500002ca9": {
          "class": "com.simiacryptus.mindseye.layers.java.AvgMetaLayer",
          "id": "370a9587-74a1-4959-b406-fa4500002ca9",
          "isFrozen": false,
          "name": "AvgMetaLayer/370a9587-74a1-4959-b406-fa4500002ca9"
        },
        "370a9587-74a1-4959-b406-fa4500002ca8": {
          "class": "com.simiacryptus.mindseye.layers.java.SqActivationLayer",
          "id": "370a9587-74a1-4959-b406-fa4500002ca8",
          "isFrozen": true,
          "name": "SqActivationLayer/370a9587-74a1-4959-b406-fa4500002ca8"
        },
        "370a9587-74a1-4959-b406-fa4500002ca7": {
          "class": "com.simiacryptus.mindseye.layers.java.LinearActivationLayer",
          "id": "370a9587-74a1-4959-b406-fa4500002ca7",
          "isFrozen": false,
          "name": "LinearActivationLayer/370a9587-74a1-4959-b406-fa4500002ca7",
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
        "370a9587-74a1-4959-b406-fa4500002ca4": {
          "class": "com.simiacryptus.mindseye.layers.java.SumInputsLayer",
          "id": "370a9587-74a1-4959-b406-fa4500002ca4",
          "isFrozen": false,
          "name": "SumInputsLayer/370a9587-74a1-4959-b406-fa4500002ca4"
        },
        "370a9587-74a1-4959-b406-fa4500002ca3": {
          "class": "com.simiacryptus.mindseye.layers.java.NthPowerActivationLayer",
          "id": "370a9587-74a1-4959-b406-fa4500002ca3",
          "isFrozen": false,
          "name": "NthPowerActivationLayer/370a9587-74a1-4959-b406-fa4500002ca3",
          "power": 0.5
        }
      },
      "links": {
        "372ac95d-1b97-4458-bb00-342c51c00807": [
          "567ce1f0-b842-477a-bf51-c75625f62a13"
        ],
        "760a2a2c-4c44-4889-87bc-ba6bd642b33f": [
          "372ac95d-1b97-4458-bb00-342c51c00807"
        ],
        "e325ff4d-2510-4830-83a1-e1379925fa10": [
          "567ce1f0-b842-477a-bf51-c75625f62a13"
        ],
        "8be5a43c-ed9f-4aa7-9e2c-0f7d605c3996": [
          "e325ff4d-2510-4830-83a1-e1379925fa10"
        ],
        "07b3a390-d8f7-42ca-b95d-5e630fe5c997": [
          "8be5a43c-ed9f-4aa7-9e2c-0f7d605c3996"
        ],
        "37e44896-a012-486a-8af2-c16e9a58cee5": [
          "760a2a2c-4c44-4889-87bc-ba6bd642b33f",
          "07b3a390-d8f7-42ca-b95d-5e630fe5c997"
        ],
        "71d93eb3-92a5-449d-ae31-e9e450154828": [
          "37e44896-a012-486a-8af2-c16e9a58cee5"
        ]
      },
      "labels": {},
      "head": "71d93eb3-92a5-449d-ae31-e9e450154828"
    }
```



### Network Diagram
Code from [LayerTestBase.java:94](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L94) executed in 0.29 seconds: 
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
    [[ 1.504, 1.292, 1.8 ]]
    --------------------
    Output: 
    [ 0.0, 0.0, 0.0 ]
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
    Inputs: [ 1.504, 1.292, 1.8 ]
    Inputs Statistics: {meanExponent=0.18126095167266496, negative=0, min=1.8, max=1.8, mean=1.532, count=3.0, positive=3, stdDev=0.2083330666664961, zeros=0}
    Output: [ 0.0, 0.0, 0.0 ]
    Outputs Statistics: {meanExponent=NaN, negative=0, min=0.0, max=0.0, mean=0.0, count=3.0, positive=0, stdDev=0.0, zeros=3}
    Feedback for input 0
    Inputs Values: [ 1.504, 1.292, 1.8 ]
    Value Statistics: {meanExponent=0.18126095167266496, negative=0, min=1.8, max=1.8, mean=1.532, count=3.0, positive=3, stdDev=0.2083330666664961, zeros=0}
    Implemented Feedback: [ [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ] ]
    Implemented Statistics: {meanExponent=NaN, negative=0, min=0.0, max=0.0, mean=0.0, count=9.0, positive=0, stdDev=0.0, zeros=9}
    Measured Feedback: [ [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ] ]
    Measured Statistics: {meanExponent=NaN, negative=0, min=0.0, max=0.0, mean=0.0, count=9.0, positive=0, stdDev=0.0, zeros=9}
    Feedback Error: [ [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ] ]
    Error Statistics: {meanExponent=NaN, negative=0, min=0.0, max=0.0, mean=0.0, count=9.0, positive=0, stdDev=0.0, zeros=9}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (9#)
    relativeTol: 0.0000e+00 +- 0.0000e+00 [Infinity - -Infinity] (0#)
    
```

Returns: 

```
    java.lang.RuntimeException: java.lang.RuntimeException: java.util.concurrent.ExecutionException: java.lang.RuntimeException: Frozen component did not pass input backwards
    	at com.simiacryptus.util.lang.TimedResult.time(TimedResult.java:61)
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
    	at org.junit.runners.Sui
```
...[skipping 693 bytes](etc/1.txt)...
```
    at com.intellij.rt.execution.junit.JUnitStarter.prepareStreamsAndStart(JUnitStarter.java:242)
    	at com.intellij.rt.execution.junit.JUnitStarter.main(JUnitStarter.java:70)
    Caused by: java.lang.RuntimeException: java.util.concurrent.ExecutionException: java.lang.RuntimeException: Frozen component did not pass input backwards
    	at com.simiacryptus.mindseye.layers.cudnn.GpuController.lambda$run$8(GpuController.java:215)
    	at com.simiacryptus.util.lang.StaticResourcePool.apply(StaticResourcePool.java:88)
    	at com.simiacryptus.mindseye.layers.cudnn.GpuController.run(GpuController.java:211)
    	at com.simiacryptus.mindseye.layers.DerivativeTester.testFrozen(DerivativeTester.java:181)
    	at com.simiacryptus.mindseye.layers.DerivativeTester.test(DerivativeTester.java:172)
    	at com.simiacryptus.mindseye.layers.LayerTestBase.lambda$test$16(LayerTestBase.java:145)
    	at com.simiacryptus.util.io.MarkdownNotebookOutput.lambda$null$1(MarkdownNotebookOutput.java:136)
    	at com.simiacryptus.util.lang.TimedResult.time(TimedResult.java:59)
    	... 35 more
    Caused by: java.util.concurrent.ExecutionException: java.lang.RuntimeException: Frozen component did not pass input backwards
    	at java.util.concurrent.FutureTask.report(FutureTask.java:122)
    	at java.util.concurrent.FutureTask.get(FutureTask.java:192)
    	at com.simiacryptus.mindseye.layers.cudnn.GpuController.lambda$run$8(GpuController.java:213)
    	... 42 more
    Caused by: java.lang.RuntimeException: Frozen component did not pass input backwards
    	at com.simiacryptus.mindseye.layers.DerivativeTester.lambda$testFrozen$19(DerivativeTester.java:199)
    	at com.simiacryptus.mindseye.layers.cudnn.GpuController.lambda$null$7(GpuController.java:213)
    	at java.util.concurrent.Executors$RunnableAdapter.call(Executors.java:511)
    	at java.util.concurrent.FutureTask.run(FutureTask.java:266)
    	at java.util.concurrent.ThreadPoolExecutor.runWorker(ThreadPoolExecutor.java:1149)
    	at java.util.concurrent.ThreadPoolExecutor$Worker.run(ThreadPoolExecutor.java:624)
    	at java.lang.Thread.run(Thread.java:748)
    
```



