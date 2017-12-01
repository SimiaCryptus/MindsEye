# FullyConnectedLayer
## FullyConnectedLayerTest
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
      "class": "com.simiacryptus.mindseye.layers.java.FullyConnectedLayer",
      "id": "f4569375-56fe-4e46-925c-95f4000009b3",
      "isFrozen": false,
      "name": "FullyConnectedLayer/f4569375-56fe-4e46-925c-95f4000009b3",
      "outputDims": [
        3
      ],
      "inputDims": [
        3
      ],
      "weights": {
        "dimensions": [
          3,
          3
        ],
        "data": [
          0.12712713148425284,
          0.8257901367294009,
          -0.5918167144000283,
          -0.8782129963032719,
          0.12679003816390902,
          0.678869295613072,
          -0.2251931768521838,
          0.10696719269606929,
          0.8340151395808627
        ]
      }
    }
```



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
    [[ -0.512, -1.532, 1.668 ]]
    --------------------
    Output: 
    [ 0.9047110000272327, -0.4386256110555184, 0.6541196497144671 ]
```



### Differential Validation
Code from [LayerTestBase.java:139](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L139) executed in 0.00 seconds: 
```java
    getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Feedback for input 0
    Inputs: [ -0.512, -1.532, 1.668 ]
    Output: [ 0.9047110000272327, -0.4386256110555184, 0.6541196497144671 ]
    Measured: [ [ 0.12712713148399502, 0.825790136729232, -0.5918167144003394 ], [ -0.8782129963025298, 0.12679003816407075, 0.6788692956138931 ], [ -0.22519317685265605, 0.10696719269620569, 0.8340151395813677 ] ]
    Implemented: [ [ 0.12712713148425284, 0.8257901367294009, -0.5918167144000283 ], [ -0.8782129963032719, 0.12679003816390902, 0.678869295613072 ], [ -0.2251931768521838, 0.10696719269606929, 0.8340151395808627 ] ]
    Error: [ [ -2.57821541893577E-13, -1.6897594434794883E-13, -3.111955138024314E-13 ], [ 7.420730696594546E-13, 1.6173173911226968E-13, 8.211209490127658E-13 ], [ -4.72261119099926E-13, 1.364047763630083E-13, 5.050404539019837E-13 ] ]
    Learning Gradient for weight set 0
    Inputs: [ -0.512, -1.532, 1.668 ]
    Outputs: [ 0.9047110000272327, -0.4386256110555184, 0.6541196497144671 ]
    Measured Gradient: [ [ -0.5120000000014002, 0.0, 0.0 ], [ 0.0, -0.5119999999991798, 0.0 ], [ 0.0, 0.0, -0.51200000000029 ], [ -1.5320000000018652, 0.0, 0.0 ], [ 0.0, -1.5319999999996448, 0.0 ], [ 0.0, 0.0, -1.5319999999996448 ], [ 1.6679999999991146, 0.0, 0.0 ], [ 0.0, 1.6680000000002249, 0.0 ], [ 0.0, 0.0, 1.6679999999991146 ] ]
    Implemented Gradient: [ [ -0.512, 0.0, 0.0 ], [ 0.0, -0.512, 0.0 ], [ 0.0, 0.0, -0.512 ], [ -1.532, 0.0, 0.0 ], [ 0.0, -1.532, 0.0 ], [ 0.0, 0.0, -1.532 ], [ 1.668, 0.0, 0.0 ], [ 0.0, 1.668, 0.0 ], [ 0.0, 0.0, 1.668 ] ]
    Error: [ [ -1.4002132786572474E-12, 0.0, 0.0 ], [ 0.0, 8.202327705930657E-13, 0.0 ], [ 0.0, 0.0, -2.899902540320909E-13 ], [ -1.865174681370263E-12, 0.0, 0.0 ], [ 0.0, 3.552713678800501E-13, 0.0 ], [ 0.0, 0.0, 3.552713678800501E-13 ], [ -8.852918398360998E-13, 0.0, 0.0 ], [ 0.0, 2.2493118478905672E-13, 0.0 ], [ 0.0, 0.0, -8.852918398360998E-13 ] ]
    Finite-Difference Derivative Accuracy:
    absoluteTol: 2.9606e-13 +- 4.3624e-13 [0.0000e+00 - 1.8652e-12] (36#)
    relativeTol: 4.9576e-13 +- 3.6263e-13 [6.7425e-14 - 1.3674e-12] (18#)
    
```

Returns: 

```
    java.lang.RuntimeException: java.lang.RuntimeException: java.util.concurrent.ExecutionException: java.lang.AssertionError: Nonfrozen component not listed in delta. Deltas: []
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
    	at org.junit.runners
```
...[skipping 796 bytes](etc/1.txt)...
```
    com.intellij.rt.execution.junit.JUnitStarter.main(JUnitStarter.java:70)
    Caused by: java.lang.RuntimeException: java.util.concurrent.ExecutionException: java.lang.AssertionError: Nonfrozen component not listed in delta. Deltas: []
    	at com.simiacryptus.mindseye.layers.cudnn.GpuController.lambda$run$8(GpuController.java:215)
    	at com.simiacryptus.util.lang.StaticResourcePool.apply(StaticResourcePool.java:88)
    	at com.simiacryptus.mindseye.layers.cudnn.GpuController.run(GpuController.java:211)
    	at com.simiacryptus.mindseye.layers.DerivativeTester.testUnFrozen(DerivativeTester.java:125)
    	at com.simiacryptus.mindseye.layers.DerivativeTester.test(DerivativeTester.java:92)
    	at com.simiacryptus.mindseye.layers.LayerTestBase.lambda$test$15(LayerTestBase.java:140)
    	at com.simiacryptus.util.io.NotebookOutput.lambda$code$1(NotebookOutput.java:157)
    	at com.simiacryptus.util.io.MarkdownNotebookOutput.lambda$null$1(MarkdownNotebookOutput.java:136)
    	at com.simiacryptus.util.lang.TimedResult.time(TimedResult.java:59)
    	... 35 more
    Caused by: java.util.concurrent.ExecutionException: java.lang.AssertionError: Nonfrozen component not listed in delta. Deltas: []
    	at java.util.concurrent.FutureTask.report(FutureTask.java:122)
    	at java.util.concurrent.FutureTask.get(FutureTask.java:192)
    	at com.simiacryptus.mindseye.layers.cudnn.GpuController.lambda$run$8(GpuController.java:213)
    	... 43 more
    Caused by: java.lang.AssertionError: Nonfrozen component not listed in delta. Deltas: []
    	at com.simiacryptus.mindseye.layers.DerivativeTester.lambda$testUnFrozen$17(DerivativeTester.java:142)
    	at com.simiacryptus.mindseye.layers.cudnn.GpuController.lambda$null$7(GpuController.java:213)
    	at java.util.concurrent.Executors$RunnableAdapter.call(Executors.java:511)
    	at java.util.concurrent.FutureTask.run(FutureTask.java:266)
    	at java.util.concurrent.ThreadPoolExecutor.runWorker(ThreadPoolExecutor.java:1149)
    	at java.util.concurrent.ThreadPoolExecutor$Worker.run(ThreadPoolExecutor.java:624)
    	at java.lang.Thread.run(Thread.java:748)
    
```



